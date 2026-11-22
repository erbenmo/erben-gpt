import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer

SEP = "[SEP]"

def build_manifest(tokenizer: Tokenizer, tokenizer_file, train_token, train_doc, val_token, val_doc, holdout_token, holdout_doc, path):
    # get tokenizer file sha256
    with open(tokenizer_file, "rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    manifest = {
        "path" : "HuggingFaceFW/fineweb-edu",
        "name" : "sample-10BT",
        "train doc size" : train_doc,
        "val doc size" : val_doc,
        "holdout doc size": holdout_doc,
        "train token size" : train_token,
        "val token size" : val_token,
        "holdout token size": holdout_token,
        "vocab size" : tokenizer.get_vocab_size(),
        "SEP" : SEP, 
        "SEP id" : tokenizer.encode(SEP).ids[0],
        "tokenizer-sha-256" : digest
    }
    manifest_file = path + "/" + "manifest.json"

    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(manifest)

# keep tokenizing more documents until we reach the num_tokens limit
def tokenize_and_save(tokenizer, data_iter, num_tokens, tokenized_file):
    print(f"Start tokenizing: {tokenized_file}")
    sep_ids = tokenizer.encode(SEP).ids

    token_counter = 0
    doc_counter = 0
    with open(tokenized_file, "ab") as file:
        while token_counter < num_tokens:
            doc = next(data_iter)['text']
            tokens = tokenizer.encode(doc).ids + sep_ids
            tokens = np.array(tokens, dtype = np.uint16)
            tokens.tofile(file)
            token_counter += len(tokens)
            doc_counter += 1

    print(f"Tokenized {token_counter} tokens and saved to {file}")
    return token_counter, doc_counter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=str, required = True, help = "path to store the final output")
    parser.add_argument("--tokenizer-file", type=str, required = True, help = "copy existing tokenizer from the given path")
    parser.add_argument("--training-tokens", type=int, default=2000, help = "how many tokens used for training")
    parser.add_argument("--validation-tokens", type=int, default=2000, help = "how many tokens used for validation")
    parser.add_argument("--holdout-tokens", type=int, default=2000, help = "how many tokens used for holdout")

    args = parser.parse_args()

    # create dataset path
    path = args.output_path
    os.makedirs(path, exist_ok=False)

    # check invariants for loading existing tokenizer
    if not Path(args.tokenizer_file).exists():
        raise ValueError(f"Given tokenizer file doesn't exist: {args.tokenizer_file}")
    tokenizer_file = path + "/" + "tokenizer.json"
    shutil.copy(args.tokenizer_file, tokenizer_file)
    # load/train the tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_file)

    # load raw dataset
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    data_iterator = iter(dataset)

    try: 
        # now encode the train and val dataset
        train_file = path + "/" + "train.tokenized.bin"
        val_file = path + "/" + "val.tokenized.bin"
        holdout_file = path + "/" + "holdout.tokenized.bin"

        train_token, train_doc = tokenize_and_save(tokenizer, data_iterator, args.training_tokens,  train_file)
        val_token, val_doc = tokenize_and_save(tokenizer, data_iterator, args.validation_tokens,  val_file)
        holdout_token, holdout_doc = tokenize_and_save(tokenizer, data_iterator, args.holdout_tokens,  holdout_file)

    finally:
        data_iterator.close()
    
    build_manifest(tokenizer, tokenizer_file, train_token, train_doc, val_token, val_doc, holdout_token, holdout_doc, path)
    os._exit(0)