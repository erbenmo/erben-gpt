import urllib.request
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.adam import Adam
from torch.nn.utils.rnn import pad_sequence
import torch
from torch import Tensor
import random
import math
from itertools import batched
import argparse
import math
import heapq
from itertools import pairwise


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


class BPETokenizer:
    def __init__(self, chars, merge_table=None):
        self.chars = chars.copy()

        self.idx = len(self.chars)

        # lower idx pair may get merged into higher idx, but not vice versa
        # Fresh dict per instance; copy caller's table so we don't mutate it.
        self.merge_table = dict(merge_table) if merge_table is not None else {}
        self.reverse_table = {}
        for pair in self.merge_table:
            value = self.merge_table[pair]
            self.reverse_table[value] = pair

    def _merge(self, pair):
        new_idx = self.idx
        self.idx += 1

        assert pair not in self.merge_table
        self.merge_table[pair] = new_idx
        self.reverse_table[new_idx] = pair
        return new_idx

    """
    first encode every char -> idx using self.chars
    
    then generate pair (idx, idx) -> list of pairs
        find most common pair
        merge the pair -> update list of pairs
        ... repeat until reaching vocab_size or list has only 1 pair
    """
    def train(self, documents, vocab_size):

        # encode char -> idx
        inputs = [] # list of list of index
        for document in documents:
            inputs.append([self.chars.index(c) for c in document])

        # generate pairs
        pair_list = []
        for indexes in inputs:
            pair_list.append([(prev, next) for prev, next in zip(indexes, indexes[1:])])

        while len(self.merge_table) + len(self.chars) < vocab_size:
            # count every pair, find the most common pair
            counter = {} # (prev, next) -> counter
            most_common = None
            for pairs in pair_list:
                for pair in pairs:
                    if pair not in counter:
                        counter[pair] = 0
                    counter[pair] += 1
                    if most_common == None or counter[most_common] < counter[pair]:
                        most_common = pair
            # no more pair left
            if most_common == None:
                break

            # register this pair
            merge_idx = self._merge(most_common)

            # merge pair in pair list
            for pairs in pair_list:
                while most_common in pairs:
                    idx = pairs.index(most_common)
                    # merge left and right pairs
                    if idx > 0:
                        pairs[idx-1] = (pairs[idx-1][0], merge_idx)
                    if idx + 1 < len(pairs):
                        pairs[idx+1] = (merge_idx, pairs[idx+1][1])

                    # remove the merged pair
                    pairs.pop(idx)


    class Token:
        def __init__(self, value, pos, prev, next, merge_rank = math.inf):
            self.value = value # encoded value
            self.pos = pos     # for breaking tie
            self.merge_rank = merge_rank
            self.prev = prev
            self.next = next

        def is_mergeable(self):
            return self.merge_rank != math.inf

        def __repr__(self):
          return f"Token(pos: {self.pos}, value: {self.value})"

    def refresh_merge_rank(self, token):
        if token == None:
            return
        
        # default to no rank
        token.merge_rank = math.inf
        if token.next == None:
            return
        
        pair = (token.value, token.next.value)
        if pair in self.merge_table:
            token.merge_rank = self.merge_table[pair]

    # merge a token with its right sibling
    def merge(self, token: Token):
        assert token.is_mergeable()
        assert token.next != None
        pair = (token.value, token.next.value)
        assert pair in self.merge_table
        new_value = self.merge_table[pair]

        # neighbors of the merged tokens
        left = token.prev
        sibling = token.next
        right = token.next.next

        # update token's value and neighbor
        token.value = new_value
        token.next = right
        if right != None:
            right.prev = token

        # left and token both have new sibling - so might become merge-able now
        self.refresh_merge_rank(left)
        self.refresh_merge_rank(token)
        
        # sibling is already merged so it can't be merged with its sibling now
        sibling.merge_rank = math.inf

        return (left, token)

    # string -> value
    def encode(self, document):
        if len(document) == 0:
            return torch.tensor([])
        
        # first create tokens, no rank, no neighbor
        tokens = []
        for pos in range(len(document)):
            c = document[pos]
            tokens.append(BPETokenizer.Token(self.chars.index(c), pos, prev = None, next = None))

        # update neighbors
        for l, r in pairwise(tokens):
            l.next = r
            r.prev = l

        # refreshs and sort by merge priority
        pq = [] # (rank, token)
        for token in tokens:
            self.refresh_merge_rank(token)
            if token.is_mergeable():
                heapq.heappush(pq, (token.merge_rank, token.pos, token))
        head = tokens[0]
        
        # now keep merging everything thats mergeable, similar to a DFS
        while pq:
            (rank, _, token) = heapq.heappop(pq)

            # the rank of the token has been updated / invalidated
            # the old rank is stale, so remove it. a new item will appear anyway
            if rank != token.merge_rank:
                continue

            # merge the token with its sibling
            (left, current) = self.merge(token)
            # left neighbor and current token has a new rank now - may trigger more merge
            if left != None and left.is_mergeable():
                heapq.heappush(pq, (left.merge_rank, left.pos, left))
            if current.is_mergeable():
                heapq.heappush(pq, (current.merge_rank, current.pos, current))
            if current.prev == None:
                head = current

        # finally collect all Tokens from head
        cur = head
        ret = []
        while cur != None:
            ret.append(cur.value)
            cur = cur.next

        return torch.tensor(ret)

    # idea is to keep expanding merged token until it is down to a single character
    def decode_single(self, value) -> str:
        # single char - termination
        if value not in self.reverse_table:
            return self.chars[value]

        # merged
        left, right = self.reverse_table[value]
        return self.decode_single(left) + self.decode_single(right)

    # tensor -> string
    def decode(self, encoded):
        s = ""
        for value in encoded:
            s += self.decode_single(value.item())
        return s

class CharacterTokenizer:
    SEP = "\n"

    def __init__(self, chars):
        self.chars = chars.copy()
        self.chars.append(self.SEP)
        self.ctoi = lambda c: self.chars.index(c)
        self.itoc = lambda i: self.chars[i]
        self.vocab_size = len(self.chars)
        self.sep_id = self.ctoi(self.SEP)

    def encode(self, name: str) -> Tensor:
        return torch.tensor([self.ctoi(c) for c in list(name)], dtype = torch.long)

    def decode(self, encoded: Tensor) -> str:
        assert encoded.ndim == 1
        return "".join([self.itoc(i) for i in encoded])

def prepare_batch(names, tokenizer):
    # input:   SEP, E, M, M, A
    # target:  E,   M, M, A, SEP
    input = [tokenizer.encode(tokenizer.SEP + name) for name in names]
    target = [tokenizer.encode(name + tokenizer.SEP) for name in names]
    # padding: SEP pad with SEP and target pad with -100 which is ignored later
    input = pad_sequence(input, batch_first=True, padding_value=torch.tensor(tokenizer.sep_id))
    target = pad_sequence(target, batch_first=True, padding_value=torch.tensor(-100))

    return (input, target)

def causal_attention_multi_head(q, k, v):
    # q, k, v: (B, HC, N, HS)
    assert k.shape == v.shape and v.shape == q.shape
    head_size = k.shape[-1]

    # each q should attend to every k, (B, HC, N, HS) @ (B, HC, HS, N) -> (B, HC, N, N)
    logits = q @ k.transpose(-1, -2) / math.sqrt(head_size)

    # then mask attention for (q, later k) to be -inf to ignore them (self attention)
    n = logits.shape[-2]
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    logits = logits.masked_fill(mask, float('-inf'))

    # finally calculate weight and multiple with v and sum
    # (B, HC, N, N) @ (B, HC, N, HS) -> (B, HC, N, HS)
    weight = torch.softmax(logits, dim=-1)
    attention = weight @ v
    return attention

class Transformer(nn.Module):
    def __init__(self, feature_count, head_count, attention_mode):
        super().__init__()

        if feature_count % head_count > 0:
            raise ValueError(f"feature count {feature_count} must be multiplier of head_count {head_count}")

        if attention_mode != "sdpa" and attention_mode != "manual":
            raise ValueError(f"invalid attention_mode: {attention_mode} must be either manual or sdpa")
        self.attention_mode = attention_mode

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
        q = []
        k = []
        v = []
        for i in range(self.head_count):
            # self-attention: generate q, k, v -> (B, N, HS)
            q.append(self.q[i](normalized_features))
            k.append(self.k[i](normalized_features))
            v.append(self.v[i](normalized_features))

        q = torch.stack(q, dim=1) # (B, HC, N, HS)
        k = torch.stack(k, dim=1)
        v = torch.stack(v, dim=1)

        if self.attention_mode == "sdpa":
            attentions = F.scaled_dot_product_attention(q, k, v, is_causal = True)
        else:
            attentions = causal_attention_multi_head(q, k, v)
        # attentions: (B, HC, N, HS)

        # merge all heads (B, HC, N, HS) into (B, N, 16)
        attentions = attentions.transpose(1, 2).flatten(-2)
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
    def __init__(self, tokenizer, max_sequence, feature_count = 16, head_count = 4, layer_count = 2, attention_mode = "manual"):
        super().__init__()

        self.tokenizer = tokenizer
        vocab_size = self.tokenizer.vocab_size

        self.max_sequence = max_sequence
        self.feature_count = feature_count
        self.head_count = head_count
        self.layer_count = layer_count
        self.attention_mode = attention_mode

        if feature_count % head_count > 0:
            raise ValueError(f"feature count {feature_count} must be multiplier of head_count {head_count}")

        # embedding each token -> features
        self.token_embeddings = nn.Embedding(vocab_size, feature_count)

        # embed position -> feature
        self.position_embedding = nn.Embedding(max_sequence, feature_count)

        # transformer: feature -> feature
        self.transformers = nn.ModuleList()
        for _ in range(layer_count):
            self.transformers.append(Transformer(feature_count, head_count, attention_mode))

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
                'chars' : self.tokenizer.chars[:len(self.tokenizer.chars) - 1], # remove SEP
                'max_sequence' : self.max_sequence,
                'feature_count' : self.feature_count,
                'head_count' : self.head_count,
                "layer_count" : self.layer_count}
        torch.save(ckpt, ckpt_path)

    @staticmethod
    def load(ckpt_path, attention_mode):
        ckpt = torch.load(ckpt_path)
        tokenizer = CharacterTokenizer(ckpt['chars'])
        model = LanguageModel(tokenizer,
                              ckpt['max_sequence'],
                              ckpt['feature_count'],
                              ckpt['head_count'],
                              ckpt['layer_count'],
                              attention_mode)
        model.load_state_dict(ckpt['weight'])
        return model

    def _generate_loss(self, names):
        (input, target) = prepare_batch(names, self.tokenizer)

        # forward pass - train on this input
        logits = self(input) # shape: [batch, max(len(name))+1, vocab_size]
        # cross entropy expects shape [batch, vocab_size, max(len(name))+1]
        logits = logits.transpose(1, 2)
        loss = F.cross_entropy(logits, target, ignore_index = -100)
        return loss

    @torch.no_grad()
    def validate(self, val_data, val_batch):
        # validation batch
        names = []
        for i in range(val_batch):
            names.append(val_data[i % len(val_data)])

        return self._generate_loss(names)

    @torch.no_grad()
    def validate_all(self, val_data, val_batch):
        total_loss = 0
        total_tokens = 0

        for names in batched(val_data, val_batch):
            loss_avg = self._generate_loss(names).item()
            tokens = sum(len(name) + 1 for name in names) # +1 for SEP
            total_tokens += tokens
            total_loss += tokens * loss_avg

        return total_loss / total_tokens


    def train_mode(self, optimizer, train_data, val_data, steps, batch = 32, val_batch = 256):
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
                val_loss = self.validate(val_data, val_batch)
                print(f"step: {step}, training loss: {loss.item():.4f}, val loss: {val_loss.item():.4f}")

    @staticmethod
    def select_next_token(logits, greedy, temperature):
        # logits.shape = (vocab_size)

        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError(f"temperature {temperature} is invalid")

        if greedy:
            return torch.argmax(logits)

        # sample by probability
        logits = logits / temperature
        probability = torch.softmax(logits, dim = 0)
        # sample 1 token
        return torch.multinomial(probability, num_samples=1)[0]

    @torch.no_grad()
    def generate(self, greedy = False, temperature = 1):
        names = []
        print("\nGeneration samples:")
        # generate a few samples
        samples = 10
        for _ in range(samples):
            name = ""
            tokens = torch.tensor([self.tokenizer.sep_id])
            while True:
                logits = self(tokens.unsqueeze(0)) # unsqueeze for batching
                logits = logits[0][-1] # only care about the last sampled token
                new_token = LanguageModel.select_next_token(logits, greedy, temperature)
                tokens = torch.cat([tokens, torch.tensor([new_token])])
                char = self.tokenizer.decode(torch.tensor([new_token]))
                if char == self.tokenizer.SEP:
                    break
                name += char
                if len(name) >= self.max_sequence - 1:
                    break
            names.append(name)
            print(name)
        return names


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
            model.train_mode(optimizer, train_data, val_data, steps)

            parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Total parameter: {parameters}")

            loss = model.validate_all(val_data, 256)
            print(f"Final loss: {loss}")

            model.save(f"{args.ckpt}.f_{feature}.l_{layer}")
            print("----------------------------------------")

            model.generate(args.greedy, args.temperature)
    else:
        for (feature, layer) in feature_layers:
            model = LanguageModel.load(f"{args.ckpt}.f_{feature}.l_{layer}", args.attention_mode)
            model.generate(args.greedy, args.temperature)
