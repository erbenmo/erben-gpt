import random
import numpy as np

from erbentokenizer import download_documents, split_documents, train_bpe_tokenizer, tokenize_and_save_documents

def test_tokenizer(tmp_path):
    random.seed(42)

    # load dataset
    documents = download_documents()
    random.shuffle(documents)
    train, _ = split_documents(documents)

    # train tokenizer
    tokenizer = train_bpe_tokenizer(train)

    # verify invariant
    input = "Hello World"
    encoded = tokenizer.encode(input)
    decoded = tokenizer.decode(encoded.ids)
    assert decoded == input

    # tokenize some documents
    documents = ["Hello World", "Life is for living"]
    file = tmp_path / "tokenizer.npy"
    ids = tokenize_and_save_documents(file, tokenizer, documents)

    # load and verify round trip
    loaded_ids = np.load(file)
    assert np.array_equal(ids, loaded_ids)