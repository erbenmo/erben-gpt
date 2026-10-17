import random

import torch
from torch.optim.adam import Adam

from tokenizers import Tokenizer
import numpy as np

from erbengpt import LanguageModel
from erbentokenizer import build_input_and_target

@torch.no_grad()
def validate(model: LanguageModel, input, target):
    model.eval()
    return model._generate_loss(input, target)

if __name__ == '__main__':
    random.seed(42)
    torch.manual_seed(42)

    # get tokenizer
    file = "tokenizer.json"
    tokenizer = Tokenizer.from_file(file)

    # get the tokenized dataset
    train_file = "train.tokenized.npy"
    train_ids = np.load(train_file)
    val_file = "val.tokenized.npy"
    val_ids = np.load(val_file)

    vocab_size = tokenizer.get_vocab_size()
    max_sequence = 64
    feature_count = 64
    head_count = 4
    layer_count = 2
    steps = 500
    train_batch = 8
    val_batch = 64

    model = LanguageModel(vocab_size, max_sequence, feature_count, head_count, layer_count, attention_mode = "sdpa")
    optimizer = Adam(model.parameters(), lr=1e-3)

    val_input, val_target = build_input_and_target(val_ids, val_batch, max_sequence)
    for step in range(steps):
        if step % 100 == 0:
            loss = validate(model, val_input, val_target)
            print(f"step: {step}, val loss: {loss.item()}")
        inputs, target = build_input_and_target(train_ids, train_batch, max_sequence)
        model.train_step(optimizer, inputs, target)
    
    loss = validate(model, val_input, val_target)
    print(f"Final loss: {loss.item()}")

