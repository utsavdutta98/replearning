import transformers
from datasets import load_dataset
import wandb
import numpy as np
import os
import torch
import random

import sys
sys.set_int_max_str_digits(0)

"""
Load huggingface datasets
"""
def load_hf_dataset(dataset_name="yelp_review_full"):

    dataset = load_dataset(dataset_name)
    return dataset

"""
Convert datasets to torch format
"""
def prepare_datasets(dataset,dataset_frac):

    for split in dataset.keys():
        if dataset[split].format['type'] != 'torch':

            print(f"Converting dataset format to torch for split : {split}")

            dataset[split].set_format('torch')
            dataset[split] = dataset[split].shuffle().select(range(int(dataset_frac * len(dataset[split]))))

    return dataset

def prepare_dataloaders(dataset,batch_size,num_workers):

    from torch.utils.data import DataLoader

    train_loader = None
    val_loader = None
    test_loader = None

    for split in dataset.keys():

        if split == 'train':
            train_loader = DataLoader(dataset[split], shuffle=True, batch_size=batch_size,num_workers=num_workers)
    
        elif split == 'test':
            test_loader = DataLoader(dataset[split], shuffle=True, batch_size=batch_size,num_workers=num_workers)

        elif split == 'val':
            test_loader = DataLoader(dataset[split], shuffle=True, batch_size=batch_size,num_workers=num_workers)

        else:
            raise ValueError("Split must be in 'train', 'test', 'val' \n")

    return train_loader, val_loader, test_loader

"""
Get tokenizer
Add padding_id 
"""
def get_tokenizer(model_name='GPT2'):

    if model_name == 'GPT2':
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    elif model_name == 'distilGPT2':
        
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    
    else:
        raise ValueError("Model name not supported \n")

    # If no padding_id is present
    if tokenizer.pad_token_id is None:
        print("Adding pad_token to tokenizer\n")
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def str_to_bool(str):
    if str == 'False':
        return False
    elif str == 'True':
        return True
    else:
        raise ValueError("Input must be a boolean")