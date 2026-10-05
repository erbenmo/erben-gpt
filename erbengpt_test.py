from erbengpt import download_dataset, split_dataset, causal_attention_multi_head, prepare_batch
from erbengpt import LanguageModel
from erbentokenizer import BPETokenizer, CharacterTokenizer
from torch.optim.adam import Adam
import torch.nn.functional as F
import torch
import pytest
import copy
import time

@pytest.fixture
def dataset():
    dataset = download_dataset()
    chars = sorted(list(set("".join(dataset))))
    max_sequence = max(len(name) for name in dataset) + 1

    train_data, val_data = split_dataset(dataset)
    yield (chars, max_sequence, train_data, val_data)

@pytest.fixture
def tokenizer():
    chars = [chr(c) for c in range(ord('a'), ord('z') + 1)]
    tokenizer = CharacterTokenizer(chars)
    yield tokenizer

@pytest.fixture
def bpe():
    chars = [chr(c) for c in range(ord('a'), ord('z') + 1)]
    tokenizer = BPETokenizer(chars)
    yield tokenizer

def test_small_model(dataset):
    (chars, max_sequence, train_data, val_data) = dataset

    steps = 10000

    # train_data with 8 names
    train_data = ['aukai', 'ellanore', 'liem', 'aquarius', 'joangel', 'wryn', 'isabela', 'astryd']
    print(train_data)

    # train on 8 dataset, validate all
    tokenizer = CharacterTokenizer(chars)
    model = LanguageModel(tokenizer, max_sequence)
    optimizer = Adam(model.parameters(), lr=1e-3)
    model.train_mode(optimizer, train_data, val_data, steps)

    loss = model.validate_all(val_data, 256)
    print(f"Final loss: {loss}")

    loss = model.validate_all(train_data, 8)
    assert 0.291 <= loss and loss <= 0.292

def test_causal_mask(tokenizer):
    # compare 2 name with same prefix
    names = ["emma", "emmy"]
    max_sequence = 5

    with(torch.no_grad()):
        model = LanguageModel(tokenizer, max_sequence)
        input = [model.tokenizer.encode(model.tokenizer.SEP + name) for name in names]
        input = torch.stack(input)
        logits = model(input)  # [batch, max(len(name))+1, vocab_size]

    # SEP, e, m, m should match
    for i in range(4):
        torch.testing.assert_close(logits[0][i], logits[1][i])

def test_save_load(tmp_path, tokenizer):
    name = "emma"
    max_sequence = 5

    model1 = LanguageModel(tokenizer, max_sequence, 8, 2, 1)
    input = model1.tokenizer.encode(model1.tokenizer.SEP + name)
    input = torch.stack([input])
    with torch.no_grad():
        logits_old = model1(input)

    # save & load
    file = tmp_path / "ckpt.file"
    model1.save(file)
    model2 = LanguageModel.load(file, "sdpa")

    # verify model metadata are the same
    assert model1.tokenizer.chars == model2.tokenizer.chars
    assert model1.max_sequence == model2.max_sequence
    assert model1.feature_count == model2.feature_count
    assert model1.head_count == model2.head_count
    assert model1.layer_count == model2.layer_count

    # verify logits are the same
    with torch.no_grad():
        logits_new = model2(input)
    torch.testing.assert_close(logits_old, logits_new)

def test_gradient(tokenizer):
    names = ["emma", "em"]
    max_sequence = 5

    # train a small model with 1 step
    model = LanguageModel(tokenizer, max_sequence, 4, 1, 1)
    loss = model._generate_loss(names)
    loss.backward()

    # verify the gradient is not None and is finite
    for (name, params) in model.named_parameters():
        assert params.grad is not None, f"grad is None for {name}"
        assert torch.isfinite(params.grad).all(), f"grad is not finite for {name}, grad: {params.grad}"

def causal_attention_helper(B, HC, N, HS):
    q = torch.randn(B, HC, N, HS, requires_grad = True)
    k = torch.randn(B, HC, N, HS, requires_grad = True)
    v = torch.randn(B, HC, N, HS, requires_grad = True)
    q2 = q.detach().clone().requires_grad_(True)
    k2 = k.detach().clone().requires_grad_(True)
    v2 = v.detach().clone().requires_grad_(True)

    # calculate attention using our utility vs pytorch native
    attention1 = causal_attention_multi_head(q, k, v)
    attention2 = F.scaled_dot_product_attention(q2, k2, v2, is_causal = True)
    torch.testing.assert_close(attention1, attention2, msg = f"attention doesn't match: {attention1}, {attention2}")

    # do a backward run and verify gradient also match
    attention1.sum().backward()
    attention2.sum().backward()
    torch.testing.assert_close(q.grad, q2.grad, msg = f"q gradient doesn't match: {q.grad}, {q2.grad}")
    torch.testing.assert_close(k.grad, k2.grad, msg = f"k gradient doesn't match: {k.grad}, {k2.grad}")
    torch.testing.assert_close(v.grad, v2.grad, msg = f"v gradient doesn't match: {v.grad}, {v2.grad}")

def test_causal_attention():
    causal_attention_helper(B=10, HC = 4, N=10, HS=16)
    causal_attention_helper(B=10, HC = 4, N=1, HS=16)
    causal_attention_helper(B=1, HC = 4, N=1, HS=16)

def model_shape_helper(names, tokenizer):
    max_sequence = max(len(name) for name in names) + 1
    B = len(names)
    N = max_sequence
    V = tokenizer.vocab_size

    # assert input shape (skip padding for now)
    model = LanguageModel(tokenizer, max_sequence)
    input = [model.tokenizer.encode(model.tokenizer.SEP + name) for name in names]
    input = torch.stack(input)
    assert input.shape == torch.Size((B, N))

    # assert logits shape
    with torch.no_grad():
        logits = model(input)
    assert logits.shape == torch.Size((B, N, V))

def test_model_shape(tokenizer):
    # create input of 2 names
    model_shape_helper(["emma", "tony"], tokenizer)

    # empty string, batch of 1
    model_shape_helper([""], tokenizer)

def test_parameter_count(tokenizer):
    names = ["emma", "tony"]
    max_sequence = max(len(name) for name in names) + 1
    V = tokenizer.vocab_size

    model = LanguageModel(tokenizer, max_sequence, feature_count = 24, head_count = 3, layer_count = 3)
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

def test_generation(tokenizer):

    logits = torch.tensor([1, 5, 2])
    new_token = LanguageModel.select_next_token(logits, True, 1.0)
    assert new_token == 1

    max_sequence = 10

    model = LanguageModel(tokenizer, max_sequence)
    names = model.generate(greedy=True)

    assert len(set(names)) == 1

def test_invalid_generation_temperature(tokenizer):
    max_sequence = 10

    model = LanguageModel(tokenizer, max_sequence)
    with pytest.raises(ValueError):
        model.generate(temperature = 0.0)

def test_manual_vs_sdpa_attention(tokenizer):
    names = ["emma", "henry", "natalie"]
    max_sequence = max(len(name) for name in names) + 1

    # create model 2 to be a clone of model1, but use sdpa
    model1 = LanguageModel(tokenizer, max_sequence, attention_mode = "manual")
    model2 = LanguageModel(tokenizer, max_sequence, attention_mode = "sdpa")
    model2.load_state_dict(copy.deepcopy(model1.state_dict()))

    # generate loss on both models - should look similar
    loss1 = model1._generate_loss(names)
    loss2 = model2._generate_loss(names)
    torch.testing.assert_close(loss1, loss2, msg = f"loss should be similar: loss1: {loss1}, loss2: {loss2}")

    # calculate gradient for loss - should be similar
    loss1.backward()
    loss2.backward()
    parameters1 = {name : p.grad for name, p in model1.named_parameters()}
    parameters2 = {name : p.grad for name, p in model2.named_parameters()}

    assert parameters1.keys() == parameters2.keys(), f"parameters key not equal)"
    for (name, p1) in parameters1.items():
        p2 = parameters2[name]
        assert p1 is not None
        assert p2 is not None
        torch.testing.assert_close(p1, p2, msg = f"grad should be similar for {name}: grad1: {p1}, grad2: {p2}")

def attention_benchmark_helper(tokenizer, N):
    # create 2 array with different attention (manual vs sdpa) but same weight
    model1 = LanguageModel(tokenizer, N, feature_count = 32, head_count = 4, layer_count = 2, attention_mode = "manual")
    model2 = LanguageModel(tokenizer, N, feature_count = 32, head_count = 4, layer_count = 2, attention_mode = "sdpa")
    model2.load_state_dict(copy.deepcopy(model1.state_dict()))

    # prepare dataset and warm up
    B = 32
    input = torch.randint(0, tokenizer.vocab_size, (B, N))
    target = torch.randint(0, tokenizer.vocab_size, (B, N))
    for model in [model1, model2]:
        for _ in range(10):
            model.zero_grad(set_to_none=True)
            logits = model(input)
            logits = logits.transpose(1, 2)
            loss = F.cross_entropy(logits, target)
            loss.backward()

    # now start the full run
    for model in [model1, model2, model1, model2]:
        times = []
        for _ in range(100):
            model.zero_grad(set_to_none=True)
            start = time.perf_counter()
            logits = model(input)
            logits = logits.transpose(1, 2)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        print(f"N = {N}, {model.attention_mode} avg latency: {sum(times) / len(times) * 1000} ms")


def test_attention_benchmark(tokenizer):
    attention_benchmark_helper(tokenizer, N = 16)
    attention_benchmark_helper(tokenizer, N = 128)

def test_tokenizer(tokenizer):
    input = "emma"
    encoded = tokenizer.encode(input)
    assert encoded.dtype == torch.long
    decoded = tokenizer.decode(encoded)
    assert input == decoded

def test_prepare_batch(tokenizer):
    names = ["emma", "al"]
    (input, target) = prepare_batch(names, tokenizer)
    
    expected_input = torch.tensor([[26, 4, 12, 12, 0], [26, 0, 11, 26, 26]])
    expected_target = torch.tensor([[4, 12, 12, 0, 26], [0, 11, 26, -100, -100]])
    torch.testing.assert_close(input, expected_input)
    torch.testing.assert_close(target, expected_target)    


def test_bpe(bpe):
    doc = ["ama", "amy", "samy"]
    # input: (0, 12, 0), (0, 12, 24), (18, 0, 12, 24)
    # mapping:
    # (0, 12) -> 26
    #   input: (26, 0), (26, 24), (18, 26, 24)
    # (26, 24) -> 27
    #   input: (26, 0), (27), (18, 27)
    # (26, 0) -> 28
    # (18, 27) -> 29
    bpe.train(doc, 26)
    assert len(bpe.merge_table) == 0

    bpe = BPETokenizer(bpe.chars)
    bpe.train(doc, 27)
    assert bpe.merge_table == {(0, 12) : 26}

    bpe = BPETokenizer(bpe.chars)
    bpe.train(doc, 42)
    expected = {(0, 12) : 26,
                (26, 24) : 27,
                (26, 0) : 28,
                (18, 27) : 29}
    assert(bpe.merge_table == expected)

    doc = ["aaa", "aaaa"]
    # -> (26, 0), (26, 26)
    bpe = BPETokenizer(bpe.chars)
    bpe.train(doc, 42)
    assert(bpe.merge_table == {(0, 0) : 26,
                               (26, 0) : 27,
                               (26, 26) : 28})

    doc = [""]
    bpe = BPETokenizer(bpe.chars)
    bpe.train(doc, 42)
    assert(len(bpe.merge_table) == 0)

    doc = ["s"]
    bpe = BPETokenizer(bpe.chars)
    bpe.train(doc, 42)
    assert(len(bpe.merge_table) == 0)

    doc = ["a", "b"]
    bpe = BPETokenizer(bpe.chars)
    bpe.train(doc, 42)
    assert(len(bpe.merge_table) == 0)

def bpe_encode_helper(merge_table, input, expected):
    chars = [chr(c) for c in range(ord('a'), ord('z') + 1)]
    tokenizer = BPETokenizer(chars, merge_table)
    ret = tokenizer.encode(input)
    assert torch.equal(ret, expected)

    decoded = tokenizer.decode(ret)
    assert input == decoded

def test_bpe_encode():
    bpe_encode_helper({
        (0, 0): 42, # (a, a)
        (42, 0): 43 # (aa, a)
    }, "", torch.tensor([]))

    bpe_encode_helper({
            (0, 0): 42, # (a, a)
            (42, 0): 43 # (aa, a)
        }, "aaa", torch.tensor([43]))

    bpe_encode_helper({
            (0, 0): 42, # (a, a)
            (42, 0): 43 # (aa, a)
        }, "abaa", torch.tensor([0, 1, 42]))

    bpe_encode_helper({
            (0, 0): 42, # (a, a)
            (42, 0): 43, # (aa, a)
            (0, 1): 44, # (a, b)
        }, "abaa", torch.tensor([44, 42]))

    bpe_encode_helper({
            (0, 0): 42, # (a, a)
            (42, 1): 43, # (aa, a)
            (43, 2): 44, # (a, b)
        }, "aabcaa", torch.tensor([44, 42]))

    bpe_encode_helper({
            (2, 0): 42, # (a, a)
            (1, 42): 43, # (b, ca)
            (0, 0): 44, # (a, a)
            (44, 43): 45 # (aa, bca)
        }, "aabcaa", torch.tensor([45, 0]))

    bpe_encode_helper({
            (0, 0): 42,
            (0, 42): 43
    }, "aaa", torch.tensor([42, 0]))

    bpe_encode_helper({
            (2, 2): 42,
            (2, 0): 43
    }, "ccca", torch.tensor([42, 43]))

    # After b+c merges, the old a+b candidate must not merge a+bc
    # ahead of the higher-priority bc+d rule. Expected: a | bcd.
    bpe_encode_helper({
            (1, 2): 42,  # b + c -> bc
            (0, 1): 43,  # a + b -> ab (stale after the first merge)
            (42, 3): 44, # bc + d -> bcd
            (0, 42): 45, # a + bc -> abc
    }, "abcd", torch.tensor([0, 44], dtype=torch.long))

def test_bpe_heap_tie_order():
    tokenizer = BPETokenizer(["a", "b", "c"], {
        (0, 2): 3,  # a + c -> ac
        (2, 2): 5,  # c + c -> cc
        (1, 0): 6,  # b + a -> ba
        (2, 0): 8,  # c + a -> ca
    })
    # Apply ac first, then cc occurrences left to right, then ca.
    # The final segmentation must be cc | ac | cc | ca.
    encoded = tokenizer.encode("ccacccca")
    expected = torch.tensor([5, 3, 5, 8], dtype=torch.long)
    torch.testing.assert_close(encoded, expected)
    assert tokenizer.decode(encoded) == "ccacccca"

def test_bpe_save_load(bpe, tmp_path):
    doc = ["ama", "amy", "samy"]
    bpe.train(doc, 42)

    file = tmp_path / "tokenizer.file"
    bpe.save(file)
    new_bpe = BPETokenizer.load(file)

    assert bpe.merge_table == new_bpe.merge_table
    assert bpe.chars == new_bpe.chars

