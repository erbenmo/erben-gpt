import math

import torch
import torch.nn.functional as F
from torch import nn


def causal_attention_multi_head(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    # q, k, v: (B, HC, N, HS)
    assert k.shape == v.shape and v.shape == q.shape
    head_size = k.shape[-1]

    # each q should attend to every k, (B, HC, N, HS) @ (B, HC, HS, N) -> (B, HC, N, N)
    logits = q @ k.transpose(-1, -2) / math.sqrt(head_size)

    # then mask attention for (q, later k) to be -inf to ignore them (self attention)
    n = logits.shape[-2]
    device = q.device
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=device), diagonal=1)
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

    # features: (B, N, F)
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

        # attentions: (B, HC, N, HS)
        if self.attention_mode == "sdpa":
            attentions = F.scaled_dot_product_attention(q, k, v, is_causal = True)
        else:
            attentions = causal_attention_multi_head(q, k, v)

        # merge all heads (B, HC, N, HS) into (B, N, F)
        attentions = attentions.transpose(1, 2).flatten(-2)
        attentions = self.output_proj(attentions)
        # update original embeddings with attentions
        features = features + attentions

        # compute layer - MLP
        normalized_features = self.ln2(features)
        mlp = self.mlp(normalized_features)
        features = features + mlp
        return features

# train a language model. token id is converted into F features and we use a linear layer to predict next token
class LanguageModel(nn.Module):
    def __init__(self, vocab_size, max_sequence, feature_count = 16, head_count = 4, layer_count = 2, attention_mode = "manual"):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_sequence = max_sequence
        self.feature_count = feature_count
        self.head_count = head_count
        self.layer_count = layer_count
        self.attention_mode = attention_mode

        if feature_count % head_count > 0:
            raise ValueError(f"feature count {feature_count} must be multiplier of head_count {head_count}")

        # embedding each token -> features
        self.token_embeddings = nn.Embedding(self.vocab_size, feature_count)

        # embed position -> feature
        self.position_embedding = nn.Embedding(max_sequence, feature_count)

        # transformer: feature -> feature
        self.transformers = nn.ModuleList()
        for _ in range(layer_count):
            self.transformers.append(Transformer(feature_count, head_count, attention_mode))

        # final LayerNorm before feature -> token mapping
        self.ln_final = nn.LayerNorm(feature_count)

        # simple mapping: turn features -> next token prediction
        self.linear = nn.Linear(feature_count, self.vocab_size)

    # tokens.shape = (batch, max_sequence (of this batch))
    def forward(self, tokens: torch.Tensor):
        # features: [B, N, F]
        # token embeddings
        features = self.token_embeddings(tokens)

        # position embedding, positions is just a batch of [0, 1, 2, 3,...]
        batch_size, max_sequence = tokens.shape[0], tokens.shape[1]
        positions = torch.stack([torch.arange(max_sequence, device = tokens.device) for _ in range(batch_size)])
        features += self.position_embedding(positions)

        # transformer blocks
        for transformer in self.transformers:
            features = transformer(features)
        features = self.ln_final(features)

        # final output: (B, N, vocab_size)
        return self.linear(features)

    def save(self) -> dict:
        ckpt = {'vocab_size' : self.vocab_size,
                'weight' : self.state_dict(),
                'max_sequence' : self.max_sequence,
                'feature_count' : self.feature_count,
                'head_count' : self.head_count,
                "layer_count" : self.layer_count,
                "attention_mode" : self.attention_mode}
        return ckpt

    @staticmethod
    def load(ckpt):
        model = LanguageModel(ckpt['vocab_size'],
                              ckpt['max_sequence'],
                              ckpt['feature_count'],
                              ckpt['head_count'],
                              ckpt['layer_count'],
                              ckpt['attention_mode'])
        model.load_state_dict(ckpt['weight'])
        return model

    def _generate_loss(self, input: torch.Tensor, target: torch.Tensor):
        # forward pass - train on this input
        logits = self(input) # shape: [batch, N, vocab_size]
        # cross entropy expects shape [batch, vocab_size, N]
        logits = logits.transpose(1, 2)
        loss = F.cross_entropy(logits, target, ignore_index = -100)
        return loss

    # run multiple forward run and accumulate gradients, then run gradient descent once
    def train_batch(self, optimizer, train_data_batch):
        self.train()
        optimizer.zero_grad()

        losses = []
        # forward run + accumulate gradient
        for (input, target) in train_data_batch:
            loss = self._generate_loss(input, target)
            loss.backward()
            losses.append(loss.item())

        # now average the accumulated gradient
        batch_size = len(train_data_batch)
        for p in self.parameters():
            if p.requires_grad:
                p.grad /= batch_size

        # finally run optimizer 
        # clip gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer.step()

        return losses

    def train_step(self, optimizer, input, target) -> torch.Tensor:
        # enter train mode
        self.train()

        # forward pass - run model on prefix, generate loss against target
        loss = self._generate_loss(input, target)

        # backward pass - find gradient, apply loss
        optimizer.zero_grad()
        loss.backward()

        # clip gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer.step()

        return loss

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