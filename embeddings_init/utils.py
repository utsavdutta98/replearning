import transformers
from datasets import load_dataset
import wandb

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

def prepare_dataloaders(dataset,batch_size):

    from torch.utils.data import DataLoader

    train_loader = None
    val_loader = None
    test_loader = None

    for split in dataset.keys():

        if split == 'train':
            train_loader = DataLoader(dataset[split], shuffle=True, batch_size=batch_size)
    
        elif split == 'test':
            test_loader = DataLoader(dataset[split], shuffle=True, batch_size=batch_size)

        elif split == 'val':
            test_loader = DataLoader(dataset[split], shuffle=True, batch_size=batch_size)

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