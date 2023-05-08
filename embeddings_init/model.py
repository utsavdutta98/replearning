# DEPRECATED

import transformers
import torch
import math
import wandb
import tqdm 
from tqdm import tqdm
import copy

class ConsolidatedModelClass:

    def __init__(self,
                model_name,
                num_layers,
                use_pretrained_embeddings,
                freeze_pretrained_embeddings,
                optimizer,
                lr,
                tokenizer,
                scheduler,
                args,
                device
                ):

        # Get model, optimizer, scheduler

        self.device = device
        self.model = self.build_model(model_name,
                                      num_layers,
                                      use_pretrained_embeddings,
                                      freeze_pretrained_embeddings)

        self.optimizer = self.build_optimizer(optimizer,lr)

        self.scheduler_flag = scheduler
        if self.scheduler_flag:
            self.scheduler = self.build_scheduler(warmup_steps=args.warmup_steps)
        
        self.tokenizer = tokenizer

        # Resize token embeddings, in case there is a disparity with the optimizer
        self.model.resize_token_embeddings(len(self.tokenizer))

        assert self.model.transformer.wte.weight.shape[0] == len(self.tokenizer), "Model's embeddings should be the same as tokenizer's embeddings"

        self.args = args

        self.losses = {
            'train_loss' : [],
            'val_loss' : []
        }

        self.scaler = torch.cuda.amp.GradScaler()

        self.init_embeddings = copy.deepcopy(self.model.transformer.wte.weight)

    def build_model(self,model_name,num_layers,use_pretrained_embeddings,freeze_pretrained_embeddings):

        if model_name == 'GPT2':

            from transformers import GPT2LMHeadModel,GPT2Config

            configuration = GPT2Config(n_layer=num_layers)
            model = GPT2LMHeadModel(configuration)

            if use_pretrained_embeddings:
     
                # Get pre_trained embeddings            
                pretrained_model = GPT2LMHeadModel.from_pretrained('gpt2')

                # Assign pre-trained embeddings to blank model
                model.transformer.wte = pretrained_model.transformer.wte

                # Freeze pre-trained embeddings
                if freeze_pretrained_embeddings:
                    model.transformer.wte.weight.requires_grad = False
        
        else:
            raise ValueError("Model name must be in ['GPT2']")
        
        model = model.to(self.device)

        return model

    ""
    def build_optimizer(self,optimizer,lr):

        if optimizer == 'AdamW':

            optim = torch.optim.AdamW(self.model.parameters(),lr)

        else:
            raise ValueError("Optimizer must be in ['AdamW']")

        return optim

    """
    Build scheduler, based on original paper
    """
    def build_scheduler(self,warmup_steps=2000,total_steps=10000,max_lr=2.5e-4,initial_lr=0):

        # LR scheduler from GPT2 paper (mentioned on Wikipedia)
        # don't ask how I got this, chatgpt generated it and I verified it works
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, 
                    lr_lambda=lambda step: 
                    ((max_lr) / warmup_steps) * step if step < warmup_steps 
                    else 
                    0.5 * (max_lr) * (1 + math.cos((step - warmup_steps) / (total_steps - warmup_steps) * math.pi))
                    )
        
        return lr_scheduler

    """
    The forward pass of the class.
    """
    def __call__(self, tokenized_batch):
        
        out = self.model(
            input_ids=tokenized_batch['input_ids'].to(self.device),
            attention_mask=tokenized_batch['attention_mask'].to(self.device),
            labels=tokenized_batch['input_ids'].to(self.device)
            )
        
        return out

    """
    Performs zero_grad, loss backprop and step for optimizer + scheduler
    """
    def step(self,batch):

        tokenized_batch = self.tokenizer(batch['text'],
                                        return_tensors='pt',
                                        padding='max_length',
                                        truncation=True,
                                        max_length=self.args.tokenizer_max_length)

        if self.model.training:

            self.optimizer.zero_grad()
            
            with torch.autocast(device_type='cuda',dtype=torch.float16):
                
                outputs = self.__call__(tokenized_batch)
                loss = outputs.loss
                self.scaler.scale(loss).backward()

                # loss.backward()

            # self.optimizer.step()
            self.scaler.step(self.optimizer)
            self.scaler.update()

        else:
            
            outputs = self.__call__(tokenized_batch)
            loss = outputs.loss

        return loss

    def train_model(self,train_loader):

        self.model.train()

        # store losses in dict
        epoch_loss = 0

        # iterate over dataloader
        for i,batch in tqdm(enumerate(train_loader)):

            # compute loss for each model and log
            loss = self.step(batch)
            epoch_loss += loss

        # take scheduler step
        if self.scheduler_flag:
            self.scheduler.step()

        # log in central dict, with average train loss
        self.losses['train_loss'].append(epoch_loss.item()/len(train_loader))

        # empty cache
        torch.cuda.empty_cache()

    def evaluate_model(self,valid_loader):

        epoch_loss = 0

        # set models to train
        self.model.eval() 

        # iterate over dataloader
        for batch in tqdm(valid_loader):

            with torch.no_grad():

                # compute loss for each model and log
                loss = self.step(batch)
                epoch_loss += loss

        # log in central dict, with average train loss
        self.losses['val_loss'].append(epoch_loss.item()/len(valid_loader))

        # empty cache
        torch.cuda.empty_cache()

    # return changes to embeddings for bookkeeping
    def get_embedding_updates(self):

        new_embeddings = self.model.transformer.wte.weight
        diff_embeddings = torch.norm(self.init_embeddings-new_embeddings).item()

        if new_embeddings.grad is not None:
            embeddings_gradient = torch.norm(new_embeddings.grad).item()
        else: 
            embeddings_gradient = 0

        return diff_embeddings,embeddings_gradient