import argparse
import random
import urllib.request
from pathlib import Path

import torch
from torch.optim.adam import Adam

from erbentokenizer import CharacterTokenizer
from erbengpt import LanguageModel
from itertools import batched
from torch.nn.utils.rnn import pad_sequence


# download dataset
def download_dataset():
    input_file = "input.txt"
    if not Path(input_file).exists():
        url = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"
        urllib.request.urlretrieve(url, input_file)
        print(f"downloaded: {input_file}")
    else:
        print(f"{input_file} already exists: ")

    with open(input_file) as f:
        dataset = [line.strip() for line in f]
    dataset = sorted(set(dataset)) # input may have duplicate
    random.seed(42)
    random.shuffle(dataset)
    return dataset

def split_dataset(dataset):
    l = len(dataset)
    split = int(l * 0.9)
    return dataset[0:split], dataset[split:]

def prepare_batch(names, tokenizer):
    # input:   SEP, E, M, M, A
    # target:  E,   M, M, A, SEP
    input = [tokenizer.encode(tokenizer.SEP + name) for name in names]
    target = [tokenizer.encode(name + tokenizer.SEP) for name in names]
    # padding: SEP pad with SEP and target pad with -100 which is ignored later
    input = pad_sequence(input, batch_first=True, padding_value=torch.tensor(tokenizer.sep_id))
    target = pad_sequence(target, batch_first=True, padding_value=torch.tensor(-100))

    return (input, target)

@torch.no_grad()
def validate(model, val_input, val_target):
    return model._generate_loss(val_input, val_target)

@torch.no_grad()
def validate_all(model, tokenizer, val_data, val_batch):
    total_loss = 0
    total_tokens = 0

    for names in batched(val_data, val_batch):
        (input, target) = prepare_batch(names, tokenizer)
        loss = model._generate_loss(input, target)
        loss_avg = loss.item()
        tokens = sum(len(name) + 1 for name in names) # +1 for SEP
        total_tokens += tokens
        total_loss += tokens * loss_avg

    return total_loss / total_tokens

def train_naming(model: LanguageModel, optimizer, tokenizer, train_data, val_data, steps):
    train_batch = 32
    val_batch = 256
    name_idx = 0  # cursor into dataset, advances across steps                        

    # validation dataset
    val_names = [val_data[i % len(val_data)] for i in range(val_batch)]
    (val_input, val_target) = prepare_batch(val_names, tokenizer)

    # start the training steps
    for step in range(steps):
        # training batch
        names = []
        for _ in range(train_batch):
            names.append(train_data[name_idx % len(train_data)])
            name_idx += 1

        # forward pass - run model on prefix, generate loss against target
        (input, target) = prepare_batch(names, tokenizer)
        loss = model.train_step(optimizer, input, target)

        if step % 1000 == 0 or step == steps - 1:
            val_loss = validate(model, val_input, val_target)
            print(f"step: {step}, training loss: {loss.item():.4f}, val loss: {val_loss.item():.4f}")

if __name__ == '__main__':
    steps = 10000

    parser = argparse.ArgumentParser()
    parser.add_argument("--load", action = "store_true", help = "load checkpoint instead of training")
    parser.add_argument("--experiment", action = "store_true", help = "experiment with different feature & layer count")
    parser.add_argument("--ckpt", default = "./ckpt-file")
    parser.add_argument("--greedy", action = "store_true", help = "use greedy sampling")
    parser.add_argument("--temperature", type = float, default=1.0, help = "sampling temperature")
    parser.add_argument("--attention-mode", default = "manual", help = "manual attention vs sdpa")

    args = parser.parse_args()

    if args.experiment:
        feature_layers = [(16, 2), (32, 2), (32, 4)]
    else:
        feature_layers = [(16, 2)]

    if not args.load:
        dataset = download_dataset()
        chars = sorted(list(set("".join(dataset))))
        tokenizer = CharacterTokenizer(chars)
        max_sequence = max(len(name) for name in dataset) + 1
        train_data, val_data = split_dataset(dataset)

        for (feature, layer) in feature_layers:
            print(f"Feature: {feature}, Layer: {layer}")
            model = LanguageModel(tokenizer, max_sequence, feature_count = feature, head_count = 4, layer_count = layer, attention_mode = args.attention_mode)
            optimizer = Adam(model.parameters(), lr=1e-3)

            train_naming(model, optimizer, tokenizer, train_data, val_data, steps)

            parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Total parameter: {parameters}")

            loss = validate_all(model, tokenizer, val_data, 256)
            print(f"Final loss: {loss}")

            model.save(f"{args.ckpt}.f_{feature}.l_{layer}")
            print("----------------------------------------")

            model.generate(args.greedy, args.temperature)
    else:
        for (feature, layer) in feature_layers:
            model = LanguageModel.load(f"{args.ckpt}.f_{feature}.l_{layer}", args.attention_mode)
            model.generate(args.greedy, args.temperature)
