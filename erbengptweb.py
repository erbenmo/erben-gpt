import argparse
import random

import torch
from torch.optim.adamw import AdamW

from tokenizers import Tokenizer
import numpy as np

from erbengpt import LanguageModel
from erbentokenizer import build_input_and_target, SEP

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

if __name__ == '__main__':
    random.seed(42)
    torch.manual_seed(42)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("--load", action = "store_true", help = "load checkpoint instead of training")
    args = parser.parse_args()

    path = "data-2000"
    # get tokenizer
    file = path + "/" + "tokenizer.json"
    tokenizer = Tokenizer.from_file(file)

    model_ckpt = path + "/" + "model.web.ckpt"
    attention_mode="sdpa"
    max_sequence = 64
    val_batch = 64
    val_file = path + "/" + "val.tokenized.npy"
    val_ids = np.load(val_file)
    val_input, val_target = build_input_and_target(val_ids, val_batch, max_sequence, device)

    if args.load:
        ckpt = torch.load(model_ckpt, 'cpu')
        model = LanguageModel.load(ckpt, attention_mode).to(device)
        model_size = sum(p.numel() for p in model.parameters())
        print(f"Loaded model from checkpoint - model size: {model_size}")
    else:
        # get the tokenized dataset
        train_file = path + "/" + "train.tokenized.npy"
        train_ids = np.load(train_file)

        vocab_size = tokenizer.get_vocab_size()
        feature_count = 64
        head_count = 4
        layer_count = 2
        steps = 500
        train_batch = 8

        model = LanguageModel(vocab_size, max_sequence, feature_count, head_count, layer_count, attention_mode)
        model = model.to(device)

        # optimizer: AdamW - with weight decay for weight-only parameters
        matrices = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
        vectors = [p for p in model.parameters() if p.requires_grad and p.dim() <= 1]
        optimizer = AdamW([
            {'params': matrices, 'weight_decay': 0.1},
            {'params': vectors, 'weight_decay': 0.0}
        ], lr=1e-3)

        for step in range(steps):
            if step % 100 == 0:
                loss = validate(model, val_input, val_target)
                print(f"step: {step}, val loss: {loss.item()}")
            inputs, target = build_input_and_target(train_ids, train_batch, max_sequence, device)
            model.train_step(optimizer, inputs, target)

        # saving model
        ckpt = model.save()
        print(f"Saving checkpoint: {model_ckpt}")
        torch.save(ckpt, model_ckpt)

    # validate model
    loss = validate(model, val_input, val_target)
    print(f"Final loss: {loss.item()}")

    # generate some stuff from this model
    prompt = "Baby squirrel is"
    print(f"\nGenerating from {prompt}:")
    output = generate(model, tokenizer, prompt, device, max_generation = 120)
    print(output)
