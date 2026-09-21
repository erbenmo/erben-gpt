import urllib.request
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.adam import Adam
from torch.nn.utils.rnn import pad_sequence
import torch
import random
import math


# download dataset
def download_dataset():
    input_file = "input.txt"
    if not Path(input_file).exists():
        url = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"
        urllib.request.urlretrieve(url, input_file)
        print(f"downloaded: {input_file}")
    else:
        print(f"already exists: {input_file}")

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

dataset = download_dataset()
train_data, val_data = split_dataset(dataset)
max_sequence = max(len(name) for name in dataset) + 1
chars = sorted(list(set("".join(dataset))))
SEP = '\n'
chars.append(SEP)
print(chars)
vocab_size = len(chars)
ctoi = lambda c: chars.index(c)
itoc = lambda i: chars[i]

# train a bigram look up table, basically for each token store the probability of all next token
class BigramModel(nn.Embedding):
    def __init__(self, vocab_size):
        super().__init__(vocab_size, vocab_size)

class Transformer(nn.Module):
    def __init__(self, feature_count, head_count):
        super().__init__()

        if feature_count % head_count > 0:
            raise ValueError(f"feature count {feature_count} must be multiplier of head_count {head_count}")

        self.ln = nn.LayerNorm(feature_count)

        # shared weight for self attention
        self.q = nn.ModuleList()
        self.k = nn.ModuleList()
        self.v = nn.ModuleList()
        self.head_count = head_count
        self.head_size = int(feature_count / head_count)
        for _ in range(head_count):
            self.q.append(nn.Linear(feature_count, self.head_size, bias = False))
            self.k.append(nn.Linear(feature_count, self.head_size, bias = False))
            self.v.append(nn.Linear(feature_count, self.head_size, bias = False))

        self.output_proj = nn.Linear(feature_count, feature_count, bias = False)

        # mlp layer
        self.ln2 = nn.LayerNorm(feature_count)
        self.mlp = nn.Sequential(
            nn.Linear(feature_count, feature_count * 4),
            nn.GELU(),
            nn.Linear(feature_count * 4, feature_count)
        )

    # features:
    def forward(self, features):
        normalized_features = self.ln(features)

        # multi-head attention
        outputs = []
        for i in range(self.head_count):
            # generate q, k, v -> (B, N, H)
            q = self.q[i](normalized_features)
            k = self.k[i](normalized_features)
            v = self.v[i](normalized_features)

            # each q should attend to every k, (B, N, H) @ (B, H, N) -> (B, N, N)
            logits = q @ k.transpose(1, 2) / math.sqrt(self.head_size)

            # then mask attention for (q, later k) to be -inf to ignore them (self attention)
            n = logits.shape[1]
            mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
            logits = logits.masked_fill(mask, float('-inf'))

            # finally calculate weight and multiple with v and sum
            # (B, N, N) @ (B, N, H) -> (B, N, H)
            weight = torch.softmax(logits, dim=-1)
            attention = weight @ v
            outputs.append(attention)

        # merge all (B, N, H) into (B, N, 16)
        attentions = torch.cat(outputs, dim=-1)
        attentions = self.output_proj(attentions)
        # update original embeddings with attentions
        features = features + attentions

        # compute layer - MLP
        normalized_features = self.ln2(features)
        mlp = self.mlp(normalized_features)
        features = features + mlp
        return features

# train a language model. token id is converted into 16 features and we use a linear layer to predict next token
class LanguageModel(nn.Module):
    def __init__(self, vocab_size, max_sequence, feature_count, head_count, layer_count):
        super().__init__()

        if feature_count % head_count > 0:
            raise ValueError(f"feature count {feature_count} must be multiplier of head_count {head_count}")

        # embedding each token -> features
        self.token_embeddings = nn.Embedding(vocab_size, feature_count)

        # embed position -> feature
        self.position_embedding = nn.Embedding(max_sequence, feature_count)

        # transformer: feature -> feature
        self.transformers = nn.ModuleList()
        for _ in range(layer_count):
            self.transformers.append(Transformer(feature_count, head_count))

        # final LayerNorm before feature -> token mapping
        self.ln_final = nn.LayerNorm(feature_count)

        # simple mapping: turn features -> next token prediction
        self.linear = nn.Linear(feature_count, vocab_size)

    # tokens.shape = (batch, max_sequence (of this batch))
    def forward(self, tokens):
        # features: [B, N, 16]
        # token embeddings
        features = self.token_embeddings(tokens)

        # position embedding, positions is just a batch of [0, 1, 2, 3,...]
        batch_size, max_sequence = tokens.shape[0], tokens.shape[1]
        positions = torch.stack([torch.arange(max_sequence) for _ in range(batch_size)])
        features += self.position_embedding(positions)

        # transformer blocks
        for transformer in self.transformers:
            features = transformer(features)
        features = self.ln_final(features)

        # final output: (B, N, vocab_size)
        return self.linear(features)

# model = BigramModel(vocab_size)

feature_count = 16
head_count = 4
layer_count = 2
steps = 10000
batch = 32
val_batch = 256

def generate_loss(model, names):
    # input:   SEP, E, M, M, A
    # target:  E,   M, M, A, SEP
    input = [torch.tensor([ctoi(c) for c in list(SEP + name)])
            for name in names]
    target = [torch.tensor([ctoi(c) for c in list(name + SEP)])
            for name in names]
    # padding: SEP -> -100 (and target == -100 is ignored for training)
    input = pad_sequence(input, batch_first=True, padding_value=torch.tensor(ctoi(SEP)))
    target = pad_sequence(target, batch_first=True, padding_value=torch.tensor(-100))

    # forward pass - train on this input
    logits = model(input) # shape: [batch, max(len(name))+1, vocab_size]
    # cross entropy expects shape [batch, vocab_size, max(len(name))+1]
    logits = logits.transpose(1, 2)
    loss = F.cross_entropy(logits, target, ignore_index = -100)
    return loss

@torch.no_grad()
def validate(model):
    # validation batch
    names = []
    for i in range(val_batch):
        names.append(val_data[i % len(val_data)])

    loss = generate_loss(model, names)
    print(f"val loss: {loss.item():.4f}")
    return loss

def train(model, optimizer):
    name_idx = 0  # cursor into dataset, advances across steps
    for step in range(steps):
        # training batch
        names = []
        for _ in range(batch):
            names.append(train_data[name_idx % len(train_data)])
            name_idx += 1

        # forward pass - run model on prefix, generate loss against target
        loss = generate_loss(model, names)

        # backward pass - find gradient, apply loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 1000 == 0:
            print(f"step: {step}, training loss: {loss.item():.4f}")
            validate(model)

def generate(model):
    print("\nGeneration samples:")
    # generate a few samples
    samples = 10
    with torch.no_grad():
        for _ in range(samples):
            name = ""
            tokens = torch.tensor([ctoi(SEP)])
            while True:
                logits = model(tokens.unsqueeze(0)) # unsqueeze for batching
                logits = logits[0][-1] # only care about the last sampled token
                new_token = torch.distributions.Categorical(logits=logits).sample()
                tokens = torch.cat([tokens, torch.tensor([new_token])])
                char = itoc(new_token.item())
                if char == SEP:
                    break
                name += char
                if len(name) >= max_sequence - 1:
                    break
            print(name)

if __name__ == '__main__':
    model = LanguageModel(vocab_size, max_sequence, feature_count, head_count, layer_count)
    optimizer = Adam(model.parameters(), lr=1e-3)

    train(model, optimizer)
    generate(model)
