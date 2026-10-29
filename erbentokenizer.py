import hashlib
import heapq
import json
import os
import pickle
import random
from itertools import pairwise
from pathlib import Path

import numpy as np
import tokenizers.decoders
import torch
from datasets import Dataset, load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from torch import Tensor

SEP = "[SEP]"

class BPETokenizer:
    def __init__(self, chars, merge_table=None):
        self.chars = chars.copy()
        self.idx = len(self.chars)
        self.ctoi = {}
        for i in range(self.idx):
            self.ctoi[self.chars[i]] = i

        # lower idx pair may get merged into higher idx, but not vice versa
        # Fresh dict per instance; copy caller's table so we don't mutate it.
        self.merge_table = dict(merge_table) if merge_table is not None else {}
        self.reverse_table = {}
        for pair in self.merge_table:
            value = self.merge_table[pair]
            self.reverse_table[value] = pair

    def _register_pair(self, pair):
        new_idx = self.idx
        self.idx += 1

        assert pair not in self.merge_table
        self.merge_table[pair] = new_idx
        self.reverse_table[new_idx] = pair
        return new_idx


    """
    1. encode into [1, 7, 3, 1, 7, 3, 1, 7]
    2. try to find common pattern like [1, 7]
    3. replace into [42, 3, 42, 3, 42]
    4. repeat until nothing left to merge or we reach vocab_size
    """
    def train(self, documents, vocab_size):
        # encode char -> idx
        encoded_list = [] # list of list of index
        for document in documents:
            encoded_list.append([self.ctoi[c] for c in document])

        steps = 0
        while len(self.merge_table) + len(self.chars) < vocab_size:
            # loop the string to find most common pair
            counter = {}
            common = None

            for encoded in encoded_list:
                for pair in pairwise(encoded):
                    if pair not in counter:
                        counter[pair] = 0
                    counter[pair] += 1
                    if common == None or counter[common] < counter[pair]:
                        common = pair

            # no adjacent pairs remain
            if common == None:
                break

            if steps % 10 == 0:
                print(f"train: {steps}")
            steps += 1
            
            # found the common pair, register it
            merge_idx = self._register_pair(common)
            
            # then replace original encoded string
            merged_list = []
            for encoded in encoded_list:
                merged = []
                i = 0
                while i < len(encoded):
                    # found a merge
                    if i+1 < len(encoded) and encoded[i] == common[0] and encoded[i+1] == common[1]:
                        merged.append(merge_idx)
                        i += 2
                    else:
                        merged.append(encoded[i])
                        i += 1
                merged_list.append(merged)
            encoded_list = merged_list

    class Token:
        def __init__(self, value, pos, prev, next, merged = False):
            self.value = value  # encoded value
            self.pos = pos      # for breaking tie
            self.prev = prev
            self.next = next
            self.merged = merged 

        def __repr__(self):
          return f"Token(pos: {self.pos}, value: {self.value})"

    # merge a token with its right sibling -> return merged token
    def merge(self, token: Token):
        new_value = self.lookup_merge(token)
        assert new_value != None

        # neighbors of the merged tokens
        left = token.prev
        sibling = token.next
        right = token.next.next

        # create a new merged token and update its neighbors
        merged = BPETokenizer.Token(new_value, token.pos, left, right)
        if left != None:
            left.next = merged
        if right != None:
            right.prev = merged

        # mark the 2 merged tokens as merged
        token.merged = True
        sibling.merged = True

        return merged

    # return rank if mergeable, else None
    def lookup_merge(self, token: Token):
        if token == None or token.next == None:
            return None
        if token.merged or token.next.merged:
            return None

        pair = (token.value, token.next.value)
        if pair not in self.merge_table:
            return None
        return self.merge_table[pair]

    def attempt_merge(self, pq, token):
        rank = self.lookup_merge(token)
        if rank != None:
            heapq.heappush(pq, (rank, token.pos, token, token.next))

    # string -> value
    def encode(self, document) -> Tensor:
        if len(document) == 0:
            return torch.tensor([])
        
        # first create tokens
        tokens = []
        for pos in range(len(document)):
            c = document[pos]
            tokens.append(BPETokenizer.Token(self.ctoi[c], pos, prev = None, next = None))
        for l, r in pairwise(tokens):
            l.next = r
            r.prev = l

        # refreshs and sort by merge priority
        pq = [] # (rank, token)
        for token in tokens:
            self.attempt_merge(pq, token)
        head = tokens[0]
        
        # now keep merging everything thats mergeable, similar to a DFS
        while pq:
            (_, _, token, expected_next) = heapq.heappop(pq)

            # either token got merged, or token.next got merged
            if token.merged or expected_next.merged:
                continue
            
            # merge the token with its sibling
            merged = self.merge(token)

            # the current merge may trigger left neighbor and merged token to become mergeable
            left = merged.prev
            self.attempt_merge(pq, left)
            self.attempt_merge(pq, merged)

            # merged token may be the head
            if merged.prev == None:
                head = merged

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
    def decode(self, encoded: Tensor) -> str:
        s = ""
        for value in encoded:
            s += self.decode_single(value.item())
        return s

    def save(self, tokenizer_path):
        with open(tokenizer_path, 'wb') as f:
            content = {"chars" : self.chars,
                       "merge_table" : self.merge_table}
            pickle.dump(content, f)

    @staticmethod
    def load(tokenizer_path):
        with open(tokenizer_path, 'rb') as f:
            content = pickle.load(f)
            return BPETokenizer(content["chars"], content["merge_table"])

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

def download_documents(doc_count = 1000):
    # download 1000 documents dataset from fineweb-edu
    file = "dataset/" + "fineweb-" + str(doc_count)
    if not Path(file).exists():
        dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        subset = Dataset.from_list(list(dataset.take(doc_count)))
        subset.save_to_disk(file)
    else:
        print(f"{file} already exists")
    dataset = Dataset.load_from_disk(file)
    return [doc['text'] for doc in dataset]

def split_documents(documents):
    l = len(documents)
    train, val = int(l * 0.9), int(l * 0.95)
    return documents[0:train], documents[train:val], documents[val:]

def train_bpe_tokenizer(train):
    # train Tokenizers
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space = False)
    tokenizer.decoder = tokenizers.decoders.ByteLevel()
    vocab_size = 8192
    trainer = BpeTrainer(vocab_size = vocab_size, initial_alphabet = ByteLevel.alphabet(), special_tokens=[SEP])
    tokenizer.train_from_iterator(train, trainer)
    return tokenizer

def tokenize_and_save_documents(file, tokenizer, documents):
    sep_ids = tokenizer.encode(SEP).ids
    ids = [token for doc in documents for token in tokenizer.encode(doc).ids + sep_ids]
    ids = np.array(ids, dtype = np.uint16)
    np.save(file, ids)
    return ids

def get_tokenized_documents(file, tokenizer, documents):
    if Path(file).exists():
        print(f"{file} exists")
        return np.load(file)
    else:
        return tokenize_and_save_documents(file, tokenizer, documents)

def build_input_and_target(ids, B, N, device, rng = None):
    inputs = []
    target = []
    for _ in range(B):
        if rng:
            pos = rng.randrange(len(ids) - N)
        else:
            pos = random.randrange(len(ids) - N)
        inputs.append(torch.tensor(ids[pos:pos+N], dtype = torch.long, device = device))
        target.append(torch.tensor(ids[pos+1:pos+1+N], dtype = torch.long, device = device))
    return torch.stack(inputs), torch.stack(target)

def build_manifest(tokenizer: Tokenizer, tokenizer_file, train, val, test, train_ids, val_ids, test_ids, D, seed, path):
    # get tokenizer file sha256
    with open(tokenizer_file, "rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    manifest = {
        "path" : "HuggingFaceFW/fineweb-edu",
        "name" : "sample-10BT",
        "sample" : D,
        "split" : [90, 5, 5],
        "split_seed" : seed,
        "train doc size" : len(train),
        "val doc size" : len(val),
        "test doc size" : len(test),
        "train token size" : len(train_ids),
        "val token size" : len(val_ids),
        "test token size" : len(test_ids),
        "vocab size" : tokenizer.get_vocab_size(),
        "SEP" : SEP, 
        "SEP id" : tokenizer.encode(SEP).ids[0],
        "tokenizer-sha-256" : digest
    }
    manifest_file = path + "/" + "manifest.json"

    if Path(manifest_file).exists():
        with open(manifest_file, encoding="utf-8") as f:
            existing = json.load(f)
            if manifest != existing:
                raise ValueError(f"Found existing manifest with mismatch. existing: {existing}, new: {manifest}")
    else:
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    seed = 42
    random.seed(seed)

    # load dataset
    D = 2000
    path = "data-" + str(D)
    os.makedirs(path, exist_ok=True)
    documents = download_documents(D)
    random.shuffle(documents)
    train, val, test = split_documents(documents)
    
    # train the tokenizer
    file = path + "/" + "tokenizer.json"
    if not Path(file).exists():
        tokenizer = train_bpe_tokenizer(train)
        tokenizer.save(file)
    else:
        print("tokenizer.json already exists")
    tokenizer = Tokenizer.from_file(file)

    # now encode the train and val dataset
    train_file = path + "/" + "train.tokenized.npy"
    val_file = path + "/" + "val.tokenized.npy"
    test_file = path + "/" + "test.tokenized.npy"
    train_ids = get_tokenized_documents(train_file, tokenizer, train)
    val_ids = get_tokenized_documents(val_file, tokenizer, val)
    test_ids = get_tokenized_documents(test_file, tokenizer, test)
    print(f"train_ids length: {len(train_ids)}, val_ids length: {len(val_ids)}, test_ids length: {len(test_ids)}")

    build_manifest(tokenizer, file, train, val, test, train_ids, val_ids, test_ids, D, seed, path)

    # build dataset
    B = 5
    N = 5
    inputs, target = build_input_and_target(train_ids, B, N, torch.device('cpu'))
    print(inputs)
    print(target)