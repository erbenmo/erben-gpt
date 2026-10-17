import argparse
import random

import torch
from torch.optim.adam import Adam

from tokenizers import Tokenizer
import numpy as np

from erbengpt import LanguageModel
from erbentokenizer import build_input_and_target, SEP

@torch.no_grad()
def validate(model: LanguageModel, input, target):
    model.eval()
    return model._generate_loss(input, target)

@torch.no_grad()
def generate(model: LanguageModel, tokenizer: Tokenizer, prompt: str, max_generation = 64):
    model.eval()
    if prompt == "":
        prompt = SEP

    encoded = tokenizer.encode(prompt).ids
    sep_id = tokenizer.encode(SEP).ids[0]
    for _ in range(max_generation):
        # bound the context window size, and create a batch of 1
        input = torch.tensor(encoded[-model.max_sequence:]).unsqueeze(0) 

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

    parser = argparse.ArgumentParser()
    parser.add_argument("--load", action = "store_true", help = "load checkpoint instead of training")
    args = parser.parse_args()

    # get tokenizer
    file = "tokenizer.json"
    tokenizer = Tokenizer.from_file(file)


    model_ckpt = "model.web.ckpt"
    attention_mode="sdpa"
    max_sequence = 64
    val_batch = 64
    val_file = "val.tokenized.npy"
    val_ids = np.load(val_file)
    val_input, val_target = build_input_and_target(val_ids, val_batch, max_sequence)

    if args.load:
        ckpt = torch.load(model_ckpt)
        model = LanguageModel.load(ckpt, attention_mode)
        model_size = sum(p.numel() for p in model.parameters())
        print(f"Loaded model from checkpoint - model size: {model_size}")
    else:
        # get the tokenized dataset
        train_file = "train.tokenized.npy"
        train_ids = np.load(train_file)

        vocab_size = tokenizer.get_vocab_size()
        feature_count = 64
        head_count = 4
        layer_count = 2
        steps = 500
        train_batch = 8

        model = LanguageModel(vocab_size, max_sequence, feature_count, head_count, layer_count, attention_mode)
        optimizer = Adam(model.parameters(), lr=1e-3)

        for step in range(steps):
            if step % 100 == 0:
                loss = validate(model, val_input, val_target)
                print(f"step: {step}, val loss: {loss.item()}")
            inputs, target = build_input_and_target(train_ids, train_batch, max_sequence)
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
    output = generate(model, tokenizer, prompt, max_generation = 120)
    print(output)
