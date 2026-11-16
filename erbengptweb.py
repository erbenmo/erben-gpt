import argparse
import random
import time
from statistics import mean

import numpy as np
import torch
from tokenizers import Tokenizer
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    LRScheduler,
    SequentialLR,
)

from erbengpt import LanguageModel, ModelConfig
from erbentokenizer import SEP, build_input_and_target


class Metrics:
    def __init__(self, completed_updates, last_lr, processed_tokens):
        self.completed_updates = completed_updates # number of times we complete entire forward + backward training
        self.last_lr = last_lr
        self.processed_tokens = processed_tokens
        self.training_loss = []
        self.validation_loss = 0

    def update(self, train_data_batch, losses, last_lr):
        self.completed_updates += 1
        self.last_lr = last_lr
        self.training_loss.extend(losses)
        for (input, _) in train_data_batch:
            B, N = input.shape
            self.processed_tokens += B * N

    def log_event(self, elapsed_ms):
        mean_loss = mean(self.training_loss) if len(self.training_loss) > 0 else float("nan")
        print(
            f"completed_updates: {self.completed_updates}, "
            f"training_loss: {mean_loss}, "
            f"validation_loss: {self.validation_loss}, "
            f"learning_rate: {self.last_lr}, "
            f"processed tokens: {self.processed_tokens}, "
            f"elapsed time(ms): {elapsed_ms}"
        )

@torch.no_grad()
def validate(model: LanguageModel, input, target, use_bf16):
    model.eval()
    return model._generate_loss(input, target, use_bf16)

@torch.no_grad()
def generate(model: LanguageModel, tokenizer: Tokenizer, prompt: str, device, max_generation = 64):
    model.eval()
    if prompt == "":
        prompt = SEP

    encoded = tokenizer.encode(prompt).ids
    sep_id = tokenizer.encode(SEP).ids[0]
    for _ in range(max_generation):
        # bound the context window size, and create a batch of 1
        input = torch.tensor(encoded[-model.config.max_sequence:], device = device).unsqueeze(0) 

        # (B, N, vocab_size)
        logits = model(input)
        # first list in batch, last element in the list
        logits = logits[0][-1]
        id = LanguageModel.select_next_token(logits, greedy = False, temperature = 1.0)
        if id == sep_id:
            break

        encoded.append(id.item())

    decoded = tokenizer.decode(encoded)
    return decoded

def create_model(tokenizer, max_sequence, feature_count, head_count, layer_count, device):
    attention_mode="sdpa"
    config = ModelConfig(tokenizer.get_vocab_size(), max_sequence, feature_count, head_count, layer_count, attention_mode)
    model = LanguageModel(config)
    model = model.to(device)
    return model
    
def create_adamw_optimizer(model, lr) -> AdamW:
    # optimizer: AdamW - with weight decay for weight-only parameters
    matrices = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    vectors = [p for p in model.parameters() if p.requires_grad and p.dim() <= 1]
    optimizer = AdamW([
        {'params': matrices, 'weight_decay': 0.1},
        {'params': vectors, 'weight_decay': 0.0}
    ], lr)

    return optimizer

def create_lr_scheduler(optimizer: AdamW, peak_lr, total_iters) -> LRScheduler:
    # warm up: linear scale up from min to peak
    start_factor = 0.1
    min_lr = start_factor * peak_lr
    end_factor = 1.0
    warmup_iters = min(200, total_iters - 1)
    linear_lr = LinearLR(optimizer, start_factor, end_factor, warmup_iters)

    # cosine decay from peak to min
    cosine_lr = CosineAnnealingLR(optimizer, total_iters - warmup_iters, min_lr)

    # combine both
    milestones=[warmup_iters]
    return SequentialLR(optimizer, [linear_lr, cosine_lr], milestones)

def save_checkpoint(ckpt_file, model: LanguageModel, optimizer: AdamW,
                    lr_scheduler: LRScheduler, completed_steps: int, processed_tokens: int):
    ckpt = model.save()
    ckpt.update({
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'python' : random.getstate(),
        'torch' : torch.get_rng_state(),
        'numpy' : np.random.get_state(),
    })
    if torch.cuda.is_available():
        ckpt['cuda'] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        ckpt['mps'] = torch.mps.get_rng_state()

    ckpt['completed-steps'] = completed_steps
    ckpt['processed_tokens'] = processed_tokens

    torch.save(ckpt, ckpt_file)
    print(f"Saved checkpoint: {ckpt_file}")

def load_checkpoint(model_ckpt, device, lr, total_iters) -> tuple[LanguageModel, AdamW, LRScheduler, int, int]:
    ckpt = torch.load(model_ckpt, 'cpu', weights_only = False)
    model = LanguageModel.load(ckpt).to(device)
    optimizer = create_adamw_optimizer(model, lr)
    # creating scheduler will overwrite the learning rate in optimizer
    # so do it first, then reload the optimizer learning rate from checkpoint
    lr_scheduler = create_lr_scheduler(optimizer, lr, total_iters)
    optimizer.load_state_dict(ckpt['optimizer'])
    lr_scheduler.load_state_dict(ckpt['lr_scheduler'])

    # load random state last, so that random state doesn't get updated again by
    # things like creating a new model with random weights
    random.setstate(ckpt['python'])
    torch.set_rng_state(ckpt['torch'].cpu())
    np.random.set_state(ckpt['numpy'])
    if "cuda" in ckpt:
        torch.cuda.set_rng_state_all([s.cpu() for s in ckpt["cuda"]])
    if "mps" in ckpt:
        torch.mps.set_rng_state(ckpt["mps"].cpu())

    completed_steps = ckpt['completed-steps']
    processed_tokens = ckpt['processed_tokens']

    print(f"Loaded checkpoint: {model_ckpt}")
    return model, optimizer, lr_scheduler, completed_steps, processed_tokens

def print_metrics(metrics: Metrics, start, new_time, val_loss):
    metrics.validation_loss = val_loss.item()
    metrics.log_event((new_time - start) * 1000)
    metrics.training_loss = []

def train_web_model(max_steps, model: LanguageModel, optimizer: AdamW, lr_scheduler: LRScheduler,
                    train_ids, val_input, val_target, device, metrics: Metrics, use_bf16):
    accumulate_batch = 4      # we accumulate 4 forward steps & gradients before running optimizer backward
    train_batch = 8           # B, batch size of input training tensor

    # train the actual models
    start = time.perf_counter()
    for step in range(max_steps):
        # accumulate multiple forward runs, then run gradient descent once
        train_data_batch = []
        for _ in range(accumulate_batch):
            train_data_batch.append(build_input_and_target(train_ids, train_batch, model.config.max_sequence, device))
        losses = model.train_batch(optimizer, train_data_batch, use_bf16)
        last_lr = lr_scheduler.get_last_lr()
        lr_scheduler.step()
        metrics.update(train_data_batch, losses, last_lr)

        if metrics.completed_updates % 100 == 0 or step == max_steps - 1:
            val_loss = validate(model, val_input, val_target, use_bf16)
            print_metrics(metrics, start, time.perf_counter(), val_loss)


def train_job(path, total_steps, current_steps, lr, device, max_sequence, feature_count, head_count, layer_count,
              load, skip_training, use_bf16, model_ckpt_path = None):

    random.seed(42)
    torch.manual_seed(42)

    # get tokenizer
    file = path + "/" + "tokenizer.json"
    tokenizer = Tokenizer.from_file(file)

    if model_ckpt_path == None:
        model_ckpt_path = "checkpoints" + "/" + "model.web.ckpt"

    # get the tokenized train dataset
    train_file = path + "/" + "train.tokenized.npy"
    train_ids = np.load(train_file)

    if load:
        model, optimizer, lr_scheduler, completed_steps, processed_tokens = load_checkpoint(model_ckpt_path, device, lr, total_steps)
        metrics = Metrics(completed_steps, lr_scheduler.get_last_lr(), processed_tokens)
    else:
        model = create_model(tokenizer, max_sequence, feature_count, head_count, layer_count, device)
        optimizer = create_adamw_optimizer(model, lr)
        lr_scheduler = create_lr_scheduler(optimizer, lr, total_steps)
        metrics = Metrics(0, lr_scheduler.get_last_lr(), 0)

    # print model stats
    print(f"Model Config: {model.config.save()}")
    print(f"Model total parameter count: {model.get_total_parameter()}")

    # get tokenized val dataset
    val_file = path + "/" + "val.tokenized.npy"
    val_ids = np.load(val_file)
    val_batch = 64
    val_input, val_target = build_input_and_target(val_ids, val_batch, model.config.max_sequence, device,
                                                   random.Random(42))

    if not skip_training:
        # train and save the initial model
        train_web_model(current_steps, model, optimizer, lr_scheduler, train_ids, val_input, val_target,
                        device, metrics, use_bf16)
        save_checkpoint(model_ckpt_path, model, optimizer, lr_scheduler, metrics.completed_updates, metrics.processed_tokens)

    return model, tokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type= str, default = "data/processed/fineweb-2000", help = "path to tokenized training data")
    parser.add_argument("--total-steps", type = int, default = 500, help = "total steps we aim for across all checkpoints")
    parser.add_argument("--current-steps", type = int, default = 500, help = "number of training steps for this job")

    parser.add_argument("--load", action = "store_true", help = "load from checkpoint instead of starting from scratch")
    parser.add_argument("--gen-only", action = "store_true", help = "skip training, just generation")
    parser.add_argument("--use-bf16", action = "store_true", help = "use bf16 for forward pass")

    parser.add_argument("--max-sequence", type = int, default = 64, help = "max token length in a training row")
    parser.add_argument("--feature-count", type = int, default = 64, help = "number of feature for each token")
    parser.add_argument("--head-count", type = int, default = 4, help = "number of attention head")
    parser.add_argument("--layer-count", type = int, default = 2, help = "number of transformer block in LLM")

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    lr = 1e-3
    load = args.load or args.gen_only

    model, tokenizer = train_job(args.data_path, args.total_steps, args.current_steps, lr, device, args.max_sequence, args.feature_count, 
                                 args.head_count, args.layer_count, load, args.gen_only, args.use_bf16)

    # generate some stuff from this model
    prompt = "Baby squirrel is"
    print(f"\nGenerating from {prompt}:")
    output = generate(model, tokenizer, prompt, device, max_generation = 120)
    print(output)
