import urllib.request
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.adam as adam
import torch
import random


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
        return [line.strip() for line in f]

dataset = download_dataset()
random.seed(42)
random.shuffle(dataset)
chars = sorted(list(set("".join(dataset))))
SEP = '\n'
chars.append(SEP)
print(chars)
vocab_size = len(chars)
ctoi = lambda c: chars.index(c)
itoc = lambda i: chars[i]

# train a bigram look up table, basically for each token store the probability of all next token
model = nn.Embedding(vocab_size, vocab_size)
optimizer = adam.Adam(model.parameters(), lr=1e-3)
steps = 100000
for step in range(steps):
    name = dataset[step % len(dataset)]

    # input:   SEP, E, M, M, A
    # target:  E,   M, M, A, SEP
    input = torch.tensor([ctoi(c) for c in list(SEP + name)])
    target = torch.tensor([ctoi(c) for c in list(name + SEP)])

    # forward pass - train on this input
    logits = model(input) # shape: [len(input), vocab_size]
    loss = F.cross_entropy(logits, target)

    # backward pass - find gradient, apply loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 1000 == 0:
        print(f"step: {step}, loss: {loss.item():.4f}, name: {name}")

print("Generation samples:")
# generate a few samples
samples = 10
with torch.no_grad():
    for step in range(samples):
        name = ""
        token = torch.tensor(ctoi(SEP))
        while True:
            logits = model(token)
            token = torch.distributions.Categorical(logits=logits).sample()
            char = itoc(token.item())
            if char == SEP:
                break
            name += char
        print(name)