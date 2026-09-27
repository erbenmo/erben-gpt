from erbengpt import download_dataset, split_dataset
from erbengpt import LanguageModel
from torch.optim.adam import Adam
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
    model2 = LanguageModel.load(file)

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
