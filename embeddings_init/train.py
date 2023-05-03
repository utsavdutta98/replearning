import transformers
import torch
import math

class ConsolidatedModelClass:

    def __init__(self,
                model_name,
                use_pretrained_embeddings,
                optimizer,
                lr,
                tokenizer,
                scheduler):

        # Get model, optimizer, scheduler
        self.model = self.build_model(model_name,use_pretrained_embeddings)
        self.optimizer = self.build_optimizer(optimizer,lr)

        self.scheduler_flag = scheduler
        if self.scheduler_flag:
            self.scheduler = self.build_scheduler()
        
        self.tokenizer = tokenizer

    def build_model(self,model_name,use_pretrained_embeddings=False):

        if model_name == 'GPT2':

            from transformers import GPT2Model,GPT2LMHeadModel,GPT2Config

            if use_pretrained_embeddings == False:
                
                configuration = GPT2Config()
                model = GPT2LMHeadModel(configuration)

            else:

                configuration = GPT2Config()
                model = GPT2LMHeadModel(configuration)

                # Get pre_trained embeddings            
                pretrained_model = GPT2LMHeadModel.from_pretrained('gpt2')

                # Assign pre-trained embeddings to blank model
                model.transformer.wte = pretrained_model.transformer.wte
        
        else:
            raise ValueError("Model name must be in ['GPT2']")
        
        return model

    def build_optimizer(self,optimizer,lr):

        if optimizer == 'AdamW':

            optim = torch.optim.AdamW(self.model.parameters(),lr)

        else:
            raise ValueError("Optimizer must be in ['AdamW']")

        return optim

    def build_scheduler(self,warmup_steps=2000,total_steps=10000,max_lr=2.5e-4,initial_lr=0):

        # LR scheduler from GPT2 paper (mentioned on Wikipedia)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, 
                    lr_lambda=lambda step: 
                    ((max_lr) / warmup_steps) * step if step < warmup_steps 
                    else 
                    0.5 * (max_lr) * (1 + math.cos((step - warmup_steps) / (total_steps - warmup_steps) * math.pi))
                    )
        
        return lr_scheduler

    def __call__(self, tokenized_batch):
        
        out = self.model(
            input_ids=tokenized_batch['input_ids'],
            attention_mask=tokenized_batch['attention_mask'],
            labels=tokenized_batch['input_ids'])
        
        return out

    def step(self,tokenized_batch):

        self.optimizer.zero_grad()
        
        outputs = self.__call__(tokenized_batch)
        loss = outputs.loss
        loss.backward()

        self.optimizer.step()

        if self.scheduler_flag:
            self.scheduler.step()

    def train(self):

        self.model.train()

    def eval(self):

        self.model.eval()

"""
Get Model

If use_pretrained_embeddings is True, then extract pre-trained embeddings and overwrite word-embeddings layer
"""
def get_model(model_name,use_pretrained_embeddings=False):

    if model_name == 'GPT2':

        from transformers import GPT2Model,GPT2LMHeadModel,GPT2Config

        if use_pretrained_embeddings == False:
            
            configuration = GPT2Config()
            model = GPT2LMHeadModel(configuration)
        
        else:

            configuration = GPT2Config()
            model = GPT2LMHeadModel(configuration)

            # Get pre_trained embeddings            
            pretrained_model = GPT2LMHeadModel.from_pretrained('gpt2')

            # Assign pre-trained embeddings to blank model
            model.transformer.wte = pretrained_model.transformer.wte

    else:
        raise ValueError("Model name must be in ['GPT2']")

    return model

"""
Get PyTorch optimizer
"""
def get_optimizer(model,optimizer,lr,**kwargs):

    if optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(),lr=lr)

    else:
        raise ValueError("Optimizer must be in ['AdamW']")

    return optimizer

def train(model,num_epochs,optimizer):
    return