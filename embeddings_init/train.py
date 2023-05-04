import tqdm
from tqdm import tqdm
import wandb

"""
Function to train dictionary of models and log
"""

def train_models(models,train_loader,args):

    # store losses in dict
    epoch_loss = {key:0 for key in models.keys()}

    # set models to train
    for model in models:
        models[model].train() 

    # iterate over dataloader
    for batch in tqdm(train_loader):

        # compute loss for each model and log
        for model in models:

            loss = models[model].step(batch)
            epoch_loss[model] += loss.item()

    # log in central dict, with average train loss
    for model in models:
        models[model].losses['train_loss'].append(epoch_loss[model]/len(train_loader))

def evaluate_models(models,valid_loader,args):

    # store losses in dict
    epoch_loss = {key:0 for key in models.keys()}

    # set models to eval
    for model in models:
        models[model].eval() 

    for batch in tqdm(valid_loader):

        for model in models:

            loss = models[model].step(batch)
            epoch_loss[model] += loss.item()

    for model in models:
        models[model].losses['val_loss'].append(epoch_loss[model]/len(valid_loader))