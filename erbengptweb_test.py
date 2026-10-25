import torch
from tokenizers import Tokenizer


from erbengptweb import create_model, create_adamw_optimizer, create_lr_scheduler, save_checkpoint, load_checkpoint, train_web_model

def test_checkpoint_resume_correctness():
    device = torch.device("cpu")

    path = "data-2000"
    # get tokenizer
    file = path + "/" + "tokenizer.json"
    tokenizer = Tokenizer.from_file(file)

    # first train a model without pause
    total_steps = 1000

