from erbengpt import download_dataset, split_dataset
from erbengpt import LanguageModel
from torch.optim.adam import Adam


def test_small_model():
    steps = 10000
    dataset = download_dataset()
    chars = sorted(list(set("".join(dataset))))
    max_sequence = max(len(name) for name in dataset) + 1

    _, val_data = split_dataset(dataset)

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