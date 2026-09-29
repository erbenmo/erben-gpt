from erbengpt import download_dataset, split_dataset, causal_attention
from erbengpt import LanguageModel
from torch.optim.adam import Adam
import torch.nn.functional as F
import torch
import pytest

@pytest.fixture
def dataset():
    dataset = download_dataset()
    chars = sorted(list(set("".join(dataset))))
    max_sequence = max(len(name) for name in dataset) + 1

    train_data, val_data = split_dataset(dataset)
    yield (chars, max_sequence, train_data, val_data)

def test_small_model(dataset):
    (chars, max_sequence, train_data, val_data) = dataset

    steps = 10000

    # train_data with 8 names
    train_data = ['aukai', 'ellanore', 'liem', 'aquarius', 'joangel', 'wryn', 'isabela', 'astryd']
    print(train_data)

    # train on 8 dataset, validate all
    model = LanguageModel(chars, max_sequence)
    optimizer = Adam(model.parameters(), lr=1e-3)
    model.train_mode(optimizer, train_data, val_data, steps)

    loss = model.validate_all(val_data, 256)
    print(f"Final loss: {loss}")

    loss = model.validate_all(train_data, 8)
    assert 0.291 <= loss and loss <= 0.292

def test_causal_mask():
    # compare 2 name with same prefix
    names = ["emma", "emmy"]
    chars = ["e", "m", "a", "y"]
    max_sequence = 5

    with(torch.no_grad()):
        model = LanguageModel(chars, max_sequence)
        input = [torch.tensor([model.ctoi(c) for c in list(model.SEP + name)])
                    for name in names]
        input = torch.stack(input)
        logits = model(input)  # [batch, max(len(name))+1, vocab_size]

    # SEP, e, m, m should match
    for i in range(4):
        torch.testing.assert_close(logits[0][i], logits[1][i])

def test_save_load(tmp_path):
    name = "emma"
    chars = ["e", "m", "a"]
    max_sequence = 5

    model1 = LanguageModel(chars, max_sequence, 8, 2, 1)
    input = torch.tensor([model1.ctoi(c) for c in list(model1.SEP + name)])
    input = torch.stack([input])
    with torch.no_grad():
        logits_old = model1(input)

    # save & load
    file = tmp_path / "ckpt.file"
    model1.save(file)
    model2 = LanguageModel.load(file, "sdpa")

    # verify model metadata are the same
    assert model1.chars == model2.chars
    assert model1.max_sequence == model2.max_sequence
    assert model1.feature_count == model2.feature_count
    assert model1.head_count == model2.head_count
    assert model1.layer_count == model2.layer_count

    # verify logits are the same
    with torch.no_grad():
        logits_new = model2(input)
    torch.testing.assert_close(logits_old, logits_new)

def test_gradient():
    chars = ["e", "m", "a"]
    names = ["emma", "em"]
    max_sequence = 5

    # train a small model with 1 step
    model = LanguageModel(chars, max_sequence, 4, 1, 1)
    loss = model._generate_loss(names)
    loss.backward()

    # verify the gradient is not None and is finite
    for (name, params) in model.named_parameters():
        assert params.grad is not None, f"grad is None for {name}"
        assert torch.isfinite(params.grad).all(), f"grad is not finite for {name}, grad: {params.grad}"

def causal_attention_helper(B, N, H):
    q = torch.randn(B, N, H, requires_grad = True)
    k = torch.randn(B, N, H, requires_grad = True)
    v = torch.randn(B, N, H, requires_grad = True)
    q2 = q.detach().clone().requires_grad_(True)
    k2 = k.detach().clone().requires_grad_(True)
    v2 = v.detach().clone().requires_grad_(True)

    # calculate attention using our utility vs pytorch native
    attention1 = causal_attention(q, k, v)
    attention2 = F.scaled_dot_product_attention(q2.unsqueeze(1), k2.unsqueeze(1), v2.unsqueeze(1), is_causal = True)
    torch.testing.assert_close(attention1, attention2.squeeze(1), msg = f"attention doesn't match: {attention1}, {attention2.squeeze(1)}")

    # do a backward run and verify gradient also match
    attention1.sum().backward()
    attention2.sum().backward()
    torch.testing.assert_close(q.grad, q2.grad, msg = f"q gradient doesn't match: {q.grad}, {q2.grad}")
    torch.testing.assert_close(k.grad, k2.grad, msg = f"k gradient doesn't match: {k.grad}, {k2.grad}")
    torch.testing.assert_close(v.grad, v2.grad, msg = f"v gradient doesn't match: {v.grad}, {v2.grad}")

def test_causal_attention():
    causal_attention_helper(B=10, N=10, H=16)
    causal_attention_helper(B=10, N=1, H=16)
    causal_attention_helper(B=1, N=1, H=16)

def model_shape_helper(names):
    chars = list(set([c for name in names for c in name]))
    max_sequence = max(len(name) for name in names) + 1
    B = len(names)
    N = max_sequence
    V = len(chars) + 1 # +1 for SEP

    # assert input shape (skip padding for now)
    model = LanguageModel(chars, max_sequence)
    input = [torch.tensor([model.ctoi(c) for c in list(model.SEP + name)])
             for name in names]
    input = torch.stack(input)
    assert input.shape == torch.Size((B, N))

    # assert logits shape
    with torch.no_grad():
        logits = model(input)
    assert logits.shape == torch.Size((B, N, V))

def test_model_shape():
    # create input of 2 names
    model_shape_helper(["emma", "tony"])

    # empty string, batch of 1
    model_shape_helper([""])

def test_parameter_count():
    names = ["emma", "tony"]
    chars = list(set([c for name in names for c in name]))
    max_sequence = max(len(name) for name in names) + 1
    V = len(chars) + 1 # +1 for SEP

    model = LanguageModel(chars, max_sequence, feature_count = 24, head_count = 3, layer_count = 3)
    D = model.feature_count
    hc = model.head_count
    H = int(D / hc)
    L = model.layer_count

    # token embedding: each V embeds to D
    token_embedding = V * D

    # positional embedding: max_sequence embeds -> D
    position_embedding = max_sequence * D

    # Transformer
    transformer = 0

    # 0. layer norm
    transformer += 2 * D

    # 1. single head (Q, K, V), each one is a linear layer that converts D -> H,
    # so D * H for weight (no bias)
    attention = (D * H) * 3

    # 2. multiple heads
    attention *= hc
    transformer += attention

    # 3. output linear projection: feature count -> feature count
    output = D * D
    transformer += output

    # 4. MLP (linear up + linear down) w/ layer norm
    mlp = 0
    mlp += 2 * D # layer norm
    mlp += D * D * 4 + 4 * D # D -> 4 * D
    mlp += D * D * 4 + D # 4 * D -> D
    transformer += mlp

    # 5. layer count multiplier
    transformer *= L

    # Output linear w/ layer norm
    output_processing = 2 * D
    # Linear projection from D -> vocab
    output_processing += D * V + V

    # get final parameter count, and assert it matches
    count = token_embedding + position_embedding + transformer + output_processing
    assert count == sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_generation():

    logits = torch.tensor([1, 5, 2])
    new_token = LanguageModel.select_next_token(logits, True, 1.0)
    assert new_token == 1

    chars = [chr(c) for c in range(ord('a'), ord('z') + 1)]
    max_sequence = 10

    model = LanguageModel(chars, max_sequence)
    names = model.generate(greedy=True)

    assert len(set(names)) == 1

def test_invalid_generation_temperature():
    chars = [chr(c) for c in range(ord('a'), ord('z') + 1)]
    max_sequence = 10

    model = LanguageModel(chars, max_sequence)
    with pytest.raises(ValueError):
        model.generate(temperature = 0.0)
