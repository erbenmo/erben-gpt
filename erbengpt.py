import urllib.request
from pathlib import Path
import torch.nn as nn

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
chars = set("".join(dataset))
NULL = '\n'
chars.add(NULL)
print(chars)
vocab_size = len(chars)


# bigram look up table, basically for each token store the probability of all next token
# shape: [vocab_size, vocab_size]
# model = 

# given a string like emma
# transform it to pairs: NULL -> e, e -> m, m -> m, m -> a, a -> NULL




