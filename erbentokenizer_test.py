import random

import numpy as np
import pytest
import torch

from erbentokenizer import (
    BPETokenizer,
    download_documents,
    split_documents,
    tokenize_and_save_documents,
    train_bpe_tokenizer,
)


@pytest.fixture
def bpe():
    chars = [chr(c) for c in range(ord('a'), ord('z') + 1)]
    tokenizer = BPETokenizer(chars)
    yield tokenizer

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


def test_tokenizer(tmp_path):
    random.seed(42)

    # load dataset
    documents = download_documents()
    random.shuffle(documents)
    train, _, _ = split_documents(documents)

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