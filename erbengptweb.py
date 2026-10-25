import argparse
import random

import torch
from torch.optim.adamw import AdamW

from tokenizers import Tokenizer
import numpy as np

from erbengpt import LanguageModel
from erbentokenizer import build_input_and_target, SEP

from torch.optim.lr_scheduler import LRScheduler, CosineAnnealingLR, LinearLR, SequentialLR

@torch.no_grad()
def validate(model: LanguageModel, input, target):
    model.eval()
    return model._generate_loss(input, target)

@torch.no_grad()
def generate(model: LanguageModel, tokenizer: Tokenizer, prompt: str, device, max_generation = 64):
    model.eval()
    if prompt == "":
        prompt = SEP

    encoded = tokenizer.encode(prompt).ids
    sep_id = tokenizer.encode(SEP).ids[0]
    for _ in range(max_generation):
        # bound the context window size, and create a batch of 1
        input = torch.tensor(encoded[-model.max_sequence:], device = device).unsqueeze(0) 

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

def create_model(vocab_size, max_sequence, device) -> LanguageModel:
    attention_mode="sdpa"
    feature_count = 64
    head_count = 4
    layer_count = 2
    model = LanguageModel(vocab_size, max_sequence, feature_count, head_count, layer_count, attention_mode)
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

def create_lr_scheduler(optimizer: AdamW, total_iters, peak_lr) -> LRScheduler:
    # warm up: linear scale up from min to peak
    start_factor = 0.1
    min_lr = start_factor * peak_lr
    end_factor = 1.0
    warmup_iters = 20
    linear_lr = LinearLR(optimizer, start_factor, end_factor, warmup_iters)

    # cosine decay from peak to min
    cosine_lr = CosineAnnealingLR(optimizer, total_iters - warmup_iters, min_lr)

    # combine both
    milestones=[warmup_iters]
    return SequentialLR(optimizer, [linear_lr, cosine_lr], milestones)

def save_checkpoint(ckpt_file, model: LanguageModel, optimizer: AdamW,
                    lr_scheduler: LRScheduler):
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

    torch.save(ckpt, ckpt_file)
    print(f"Saved checkpoint: {ckpt_file}")

def load_checkpoint(model_ckpt, device, total_steps, lr) -> tuple[LanguageModel, AdamW, LRScheduler]:
    ckpt = torch.load(model_ckpt, 'cpu', weights_only = False)
    model = LanguageModel.load(ckpt).to(device)
    optimizer = create_adamw_optimizer(model, lr)
    # creating scheduler will overwrite the learning rate in optimizer
    # so do it first, then reload the optimizer learning rate from checkpoint
    lr_scheduler = create_lr_scheduler(optimizer, total_steps, lr)
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

    print(f"Loaded checkpoint: {model_ckpt}")
    return model, optimizer, lr_scheduler

def train_web_model(steps, model: LanguageModel, optimizer: AdamW, lr_scheduler: LRScheduler,
                    max_sequence, train_ids, val_input, val_target, device):
    # dynamic learning rate
    train_batch = 8
    accumulation_steps = 4

    # train the actual models
    for step in range(steps):
        if step % 100 == 0:
            loss = validate(model, val_input, val_target)
            print(f"step: {step}, val loss: {loss.item()}")

        # accumulate multiple forward runs, then run gradient descent once
        train_data_batch = []
        for _ in range(accumulation_steps):
            train_data_batch.append(build_input_and_target(train_ids, train_batch, max_sequence, device))
        model.train_batch(optimizer, train_data_batch)
        lr_scheduler.step()

def train_job(device, load, val_only):
    total_steps = 10000
    lr = 1e-3

    random.seed(42)
    torch.manual_seed(42)

    path = "data-2000"
    # get tokenizer
    file = path + "/" + "tokenizer.json"
    tokenizer = Tokenizer.from_file(file)

    model_ckpt = path + "/" + "model.web.ckpt"
    max_sequence = 64

    # get tokenized val dataset
    val_file = path + "/" + "val.tokenized.npy"
    val_ids = np.load(val_file)
    val_batch = 64
    val_input, val_target = build_input_and_target(val_ids, val_batch, max_sequence, device,
                                                   random.Random(42))

    # get the tokenized train dataset
    train_file = path + "/" + "train.tokenized.npy"
    train_ids = np.load(train_file)

    if load:
        model, optimizer, lr_scheduler = load_checkpoint(model_ckpt, device, total_steps, lr)
    else:
        model = create_model(tokenizer.get_vocab_size(), max_sequence, device)
        optimizer = create_adamw_optimizer(model, lr)
        lr_scheduler = create_lr_scheduler(optimizer, total_steps, lr)

    # train and save the initial model
    if not val_only:
        steps = 500
        train_web_model(steps, model, optimizer, lr_scheduler, max_sequence, train_ids, val_input, val_target, device)
        save_checkpoint(model_ckpt, model, optimizer, lr_scheduler)

    # validate model
    loss = validate(model, val_input, val_target)
    print(f"Final loss: {loss.item()}")

    return model, tokenizer, device

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", action = "store_true", help = "load checkpoint instead of training")
    parser.add_argument("--val-only", action = "store_true", help = "skip training, just validation")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model, tokenizer, device = train_job(device, args.load, args.val_only)

    # generate some stuff from this model
    prompt = "Baby squirrel is"
    print(f"\nGenerating from {prompt}:")
    output = generate(model, tokenizer, prompt, device, max_generation = 120)
    print(output)