# ---------------------------------------------------------------------------- #
#                                  DEPRECATED                                  #
# ---------------------------------------------------------------------------- #


# import tqdm
# from tqdm import tqdm
# import wandb
# import torch

# """
# Function to train dictionary of models and log
# """

# def train_models(Model,train_loader,args):

#     # store losses in dict
#     epoch_loss = 0

#     # set models to train
#     Model.train() 

#     # iterate over dataloader
#     for i,batch in tqdm(enumerate(train_loader)):

#         # compute loss for each model and log
#         loss = Model.step(batch)
#         epoch_loss += loss

#     # take scheduler step
#     if Model.scheduler_flag:
#         Model.scheduler.step()

#     # log in central dict, with average train loss
#     Model.losses['train_loss'].append(epoch_loss.item()/len(train_loader))

#     # empty cache
#     torch.cuda.empty_cache()

# def evaluate_models(Model,valid_loader,args):

#     # store losses in dict
#     epoch_loss = 0

#     # set models to train
#     Model.eval() 

#     # iterate over dataloader
#     for batch in tqdm(valid_loader):

#         with torch.no_grad():

#             # compute loss for each model and log
#             loss = Model.step(batch)
#             epoch_loss += loss

#     # log in central dict, with average train loss
#     Model.losses['val_loss'].append(epoch_loss.item()/len(valid_loader))

#     # empty cache
#     torch.cuda.empty_cache()