import urllib.request
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.adam import Adam
from torch.nn.utils.rnn import pad_sequence
import torch
import random
import math
from itertools import batched
import argparse

steps = 10000
batch = 32
val_batch = 256

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
    SEP = "\n"

    def __init__(self, chars, max_sequence, feature_count = 16, head_count = 4, layer_count = 2):
        super().__init__()

        self.chars = chars.copy()
        self.chars.append(self.SEP)
        print(self.chars)

        self.ctoi = lambda c: self.chars.index(c)
        self.itoc = lambda i: self.chars[i]

        vocab_size = len(self.chars)
        self.max_sequence = max_sequence
        self.feature_count = feature_count
        self.head_count = head_count
        self.layer_count = layer_count

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

    def save(self, ckpt_path):
        ckpt = {'weight' : self.state_dict(),
                'chars' : self.chars[:len(self.chars) - 1], # remove SEP
                'max_sequence' : self.max_sequence,
                'feature_count' : self.feature_count,
                'head_count' : self.head_count,
                "layer_count" : self.layer_count}
        torch.save(ckpt, ckpt_path)

    @staticmethod
    def load(ckpt_path):
        ckpt = torch.load(ckpt_path)
        model = LanguageModel(ckpt['chars'],
                              ckpt['max_sequence'],
                              ckpt['feature_count'],
                              ckpt['head_count'],
                              ckpt['layer_count'])
        model.load_state_dict(ckpt['weight'])
        return model


    def _generate_loss(self, names):
        # input:   SEP, E, M, M, A
        # target:  E,   M, M, A, SEP
        input = [torch.tensor([self.ctoi(c) for c in list(self.SEP + name)])
                for name in names]
        target = [torch.tensor([self.ctoi(c) for c in list(name + self.SEP)])
                for name in names]
        # padding: SEP -> -100 (and target == -100 is ignored for training)
        input = pad_sequence(input, batch_first=True, padding_value=torch.tensor(self.ctoi(self.SEP)))
        target = pad_sequence(target, batch_first=True, padding_value=torch.tensor(-100))

        # forward pass - train on this input
        logits = self(input) # shape: [batch, max(len(name))+1, vocab_size]
        # cross entropy expects shape [batch, vocab_size, max(len(name))+1]
        logits = logits.transpose(1, 2)
        loss = F.cross_entropy(logits, target, ignore_index = -100)
        return loss

    @torch.no_grad()
    def validate(self):
        # validation batch
        names = []
        for i in range(val_batch):
            names.append(val_data[i % len(val_data)])

        return self._generate_loss(names)

    @torch.no_grad()
    def validate_all(self):
        total_loss = 0
        total_tokens = 0

        for names in batched(val_data, val_batch):
            loss_avg = self._generate_loss(names).item()
            tokens = sum(len(name) + 1 for name in names) # +1 for SEP
            total_tokens += tokens
            total_loss += tokens * loss_avg

        return total_loss / total_tokens


    def train_mode(self, optimizer):
        name_idx = 0  # cursor into dataset, advances across steps
        for step in range(steps):
            # training batch
            names = []
            for _ in range(batch):
                names.append(train_data[name_idx % len(train_data)])
                name_idx += 1

            # forward pass - run model on prefix, generate loss against target
            loss = self._generate_loss(names)

            # backward pass - find gradient, apply loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 1000 == 0 or step == steps - 1:
                val_loss = self.validate()
                print(f"step: {step}, training loss: {loss.item():.4f}, val loss: {val_loss.item():.4f}")

    @torch.no_grad()
    def generate(self):
        print("\nGeneration samples:")
        # generate a few samples
        samples = 10
        for _ in range(samples):
            name = ""
            tokens = torch.tensor([self.ctoi(self.SEP)])
            while True:
                logits = self(tokens.unsqueeze(0)) # unsqueeze for batching
                logits = logits[0][-1] # only care about the last sampled token
                new_token = torch.distributions.Categorical(logits=logits).sample()
                tokens = torch.cat([tokens, torch.tensor([new_token])])
                char = self.itoc(new_token.item())
                if char == self.SEP:
                    break
                name += char
                if len(name) >= self.max_sequence - 1:
                    break
            print(name)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--load", action="store_true", help="load checkpoint instead of training")
    parser.add_argument("--experiment", action="store_true", help="experiment with different feature & layer count")
    parser.add_argument("--ckpt", default="./ckpt-file")
    args = parser.parse_args()

    if args.experiment:
        feature_layers = [(16, 2), (32, 2), (32, 4)]
    else:
        feature_layers = [(16, 2)]

    if not args.load:
        dataset = download_dataset()
        train_data, val_data = split_dataset(dataset)
        chars = sorted(list(set("".join(dataset))))
        max_sequence = max(len(name) for name in dataset) + 1

        for (feature, layer) in feature_layers:
            print(f"Feature: {feature}, Layer: {layer}")
            model = LanguageModel(chars, max_sequence, feature_count = feature, head_count = 4, layer_count = layer)
            optimizer = Adam(model.parameters(), lr=1e-3)
            model.train_mode(optimizer)

            parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Total parameter: {parameters}")

            loss = model.validate_all()
            print(f"Final loss: {loss}")

            model.save(f"{args.ckpt}.f_{feature}.l_{layer}")
            print("----------------------------------------")
    else:
        for (feature, layer) in feature_layers:
            model = LanguageModel.load(f"{args.ckpt}.f_{feature}.l_{layer}")
            model.generate()
