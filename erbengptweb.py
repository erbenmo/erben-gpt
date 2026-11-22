import argparse
import json
import os
import random
import time
from statistics import mean

import numpy as np
import torch
from erbengpt import LanguageModel, ModelConfig
from erbentokenizer import SEP, build_input_and_target
from tokenizers import Tokenizer
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    LRScheduler,
    SequentialLR,
)


class TrainingConfig:
    def __init__(self, data_dir, run_dir, total_steps, job_steps, 
                 descent_batch = 4, training_batch_size = 8, use_bf16 = False):
        self.data_dir = data_dir
        self.run_dir = run_dir
        self.total_steps = total_steps
        self.job_steps = job_steps
        self.descent_batch = descent_batch               # how many training batch to accumulate before running gradient descent
        self.training_batch_size = training_batch_size   # size of training batch, the "B"
        self.use_bf16 = use_bf16                         # use bf16 for computation (weight is still stored as fp32)

    def checkpoint_file(self):
        return self.run_dir + "/" + "model.web.ckpt"

class TrainingMetrics:
    def __init__(self, completed_updates, last_lr, processed_tokens, training_duration_ms, val_tokens, val_loss):
        self.completed_updates = completed_updates # number of times we complete entire forward + backward training
        self.last_lr = last_lr
        self.processed_tokens = processed_tokens
        self.training_loss = []
        self.validation_loss = 0
        self.training_duration_ms = training_duration_ms
        self.val_tokens = val_tokens
        self.val_loss = val_loss
        self.start_time = 0

    def start(self):
        self.start_time = time.perf_counter()

    def update_duration(self):
        current = time.perf_counter()
        self.training_duration_ms += (current - self.start_time) * 1000
        self.start_time = current

    def update(self, train_data_batch, losses, last_lr):
        self.completed_updates += 1
        self.last_lr = last_lr
        self.training_loss.extend(losses)
        for (input, _) in train_data_batch:
            B, N = input.shape
            self.processed_tokens += B * N
        self.update_duration()

    def log_event(self):
        mean_loss = mean(self.training_loss) if len(self.training_loss) > 0 else float("nan")
        print(
            f"completed_updates: {self.completed_updates}, "
            f"training_loss: {mean_loss}, "
            f"validation_loss: {self.validation_loss}, "
            f"learning_rate: {self.last_lr}, "
            f"processed tokens: {self.processed_tokens}, "
            f"training duration(ms): {self.training_duration_ms}"
        )

@torch.no_grad()
def validate(model: LanguageModel, input, target, use_bf16, reduction = "mean") -> torch.Tensor:
    model.eval()
    return model._generate_loss(input, target, use_bf16, reduction)

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

def save_checkpoint(ckpt_file, model: LanguageModel, optimizer: AdamW, lr_scheduler: LRScheduler, metrics: TrainingMetrics):
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

    ckpt['completed_updates'] = metrics.completed_updates
    ckpt['processed_tokens'] = metrics.processed_tokens
    ckpt['training_duration_ms'] = metrics.training_duration_ms

    ckpt_tmp = f"{ckpt_file}.tmp"
    torch.save(ckpt, ckpt_tmp)
    os.replace(ckpt_tmp, ckpt_file)
    print(f"Saved checkpoint: {ckpt_file}")

def load_checkpoint(config: TrainingConfig, device, lr, total_iters) -> tuple[LanguageModel, AdamW, LRScheduler, TrainingMetrics]:
    ckpt = torch.load(config.checkpoint_file(), 'cpu', weights_only = False)
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

    completed_steps = ckpt['completed_updates']
    processed_tokens = ckpt['processed_tokens']
    training_duration_ms = ckpt['training_duration_ms']

    metrics = TrainingMetrics(completed_steps, lr_scheduler.get_last_lr(), processed_tokens, training_duration_ms, 0, 0)

    print(f"Loaded checkpoint: {config.checkpoint_file()}")
    return model, optimizer, lr_scheduler, metrics

def print_metrics(metrics: TrainingMetrics, val_loss):
    metrics.validation_loss = val_loss.item()
    metrics.log_event()
    metrics.training_loss = []

def train_web_model(config: TrainingConfig, model: LanguageModel, optimizer: AdamW, lr_scheduler: LRScheduler,
                    train_ids, val_input, val_target, device, metrics: TrainingMetrics):
    # train the actual models
    metrics.start()
    for step in range(config.job_steps):
        # accumulate multiple forward runs, then run gradient descent once
        train_data_batch = []
        for _ in range(config.descent_batch):
            train_data_batch.append(build_input_and_target(train_ids, config.training_batch_size,
                                                           model.config.max_sequence, device))
        losses = model.train_batch(optimizer, train_data_batch, config.use_bf16)
        last_lr = lr_scheduler.get_last_lr()
        lr_scheduler.step()
        metrics.update(train_data_batch, losses, last_lr)

        if metrics.completed_updates % 100 == 0 or step == config.job_steps - 1:
            val_loss = validate(model, val_input, val_target, config.use_bf16)
            print_metrics(metrics, val_loss)

        # checkpoint periodically
        if metrics.completed_updates % 1000 == 0 or step == config.job_steps - 1:
            save_checkpoint(config.checkpoint_file(), model, optimizer, lr_scheduler, metrics)

    return metrics

def save_config(training_config: TrainingConfig, model: LanguageModel, load):
    config = {"training_config" : vars(training_config),
              "model_config" : vars(model.config),
              "total_parameter" : model.get_total_parameter()}
    file = training_config.run_dir + "/config.json"

    # for resume training: verify model config. training config may differ.
    if load:
        with open(file, "r") as f:
            existing_config = json.load(f)
            existing_model_config = ModelConfig.load(existing_config["model_config"])
            if vars(existing_model_config) != vars(model.config):
                raise ValueError(f"Resume Model Config doesn't match original config: {vars(model.config)}, existing: {vars(existing_model_config)}")
            existing_training_config = TrainingConfig(**existing_config["training_config"])
            if existing_training_config.data_dir != training_config.data_dir or existing_training_config.total_steps != training_config.total_steps:
                raise ValueError(f"Resume Training Config doesn't match original config: {vars(training_config)}, existing: {vars(existing_training_config)}")
            return
    
    print(f"config: {json.dumps(config, indent=2)}")
    with open(file, "w") as f:
        json.dump(config, f, indent=2)

# large scale validation on 1M tokens after training
def post_training_validation(training_config: TrainingConfig, model: LanguageModel, val_ids, val_batch, device):

    rng = random.Random(42)

    print("Start post-training validation")
    # aggregated
    val_tokens = 0
    val_losses = 0

    # validate a new batch until we validate 1M tokens
    while val_tokens < 1000000:
        val_input, val_target = build_input_and_target(val_ids, val_batch, model.config.max_sequence, device, rng)
        loss = validate(model, val_input, val_target, training_config.use_bf16, reduction = "sum")
        val_losses += loss.item()
        val_tokens += val_input.numel()

    average_loss_per_token = val_losses / val_tokens
    print(f"Completed post-training validation, loss: {average_loss_per_token}")
    return (average_loss_per_token, val_tokens)

def save_training_result(training_config: TrainingConfig, model: LanguageModel, metrics: TrainingMetrics):
    if metrics.completed_updates >= training_config.total_steps and metrics.val_tokens > 0:
        status = "completed"
    else:
        status = "in_progress"
    result = {
        "status": status,
        "parameter_count": model.get_total_parameter(),
        "metrics": vars(metrics)
    }
    file = training_config.run_dir + "/result.json"
    with open(file, "w") as file:
        json.dump(result, file, indent=2)
    print(f"Training result: {json.dumps(result, indent=2)}")

def train_job(config: TrainingConfig, lr, device, max_sequence, feature_count, head_count, layer_count,
              load, training, final_validation=True):

    random.seed(42)
    torch.manual_seed(42)
    os.makedirs(config.run_dir, exist_ok = True)

    # get tokenizer
    file = config.data_dir + "/" + "tokenizer.json"
    tokenizer = Tokenizer.from_file(file)

    # get the tokenized train dataset
    train_file = config.data_dir + "/" + "train.tokenized.npy"
    train_ids = np.load(train_file)

    if load:
        model, optimizer, lr_scheduler, metrics = load_checkpoint(config, device, lr, config.total_steps)
    else:
        model = create_model(tokenizer, max_sequence, feature_count, head_count, layer_count, device)
        optimizer = create_adamw_optimizer(model, lr)
        lr_scheduler = create_lr_scheduler(optimizer, lr, config.total_steps)
        metrics = TrainingMetrics(0, lr_scheduler.get_last_lr(), 0, 0, 0, 0)

    if training:
        save_config(config, model, load)

    # get tokenized val dataset
    val_file = config.data_dir + "/" + "val.tokenized.npy"
    val_ids = np.load(val_file)
    val_batch = 64
    val_input, val_target = build_input_and_target(val_ids, val_batch, model.config.max_sequence, device,
                                                   random.Random(42))

    if training:
        # train and save the initial model
        metrics = train_web_model(config, model, optimizer, lr_scheduler, train_ids, val_input, val_target, 
                                  device, metrics)
        if final_validation:
            metrics.val_loss, metrics.val_tokens = post_training_validation(config, model, val_ids, val_batch, device)
        save_training_result(config, model, metrics)

    return model, tokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type= str, default = "data/processed/fineweb-2000", help = "path to tokenized training data")
    parser.add_argument("--run-path", type= str, default = "./experiments/", help = "path to run directory")

    parser.add_argument("--load", action = "store_true", help = "load from checkpoint instead of starting from scratch")
    parser.add_argument("--gen-only", action = "store_true", help = "skip training, just generation")

    # model config
    parser.add_argument("--max-sequence", type = int, default = 64, help = "max token length in a training row")
    parser.add_argument("--feature-count", type = int, default = 64, help = "number of feature for each token")
    parser.add_argument("--head-count", type = int, default = 4, help = "number of attention head")
    parser.add_argument("--layer-count", type = int, default = 2, help = "number of transformer block in LLM")

    # training config
    parser.add_argument("--total-steps", type = int, default = 1000000, help = "total steps we aim for across all checkpoints")
    parser.add_argument("--job-steps", type = int, default = 500, help = "number of training steps for this job")
    parser.add_argument("--descent-batch", type = int, default = 4, help = "how many training batch to accumulate before running gradient descent")
    parser.add_argument("--training-batch-size", type = int, default = 8, help = "size of training batch, the B")
    parser.add_argument("--use-bf16", action = "store_true", help = "use bf16 for forward pass")

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    lr = 1e-3
    load = args.load or args.gen_only

    config = TrainingConfig(args.data_path, args.run_path, args.total_steps, args.job_steps, 
                            args.descent_batch, args.training_batch_size, args.use_bf16)
    model, tokenizer = train_job(config, lr, device, args.max_sequence, args.feature_count, 
                                 args.head_count, args.layer_count, load, not args.gen_only)

    # generate some stuff from this model
    prompt = "Baby squirrel is"
    print(f"\nGenerating from {prompt}:")
    output = generate(model, tokenizer, prompt, device, max_generation = 120)
    print(output)
