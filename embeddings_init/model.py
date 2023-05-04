import transformers
import torch
import math
import wandb

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
                device):

        # Get model, optimizer, scheduler

        self.device = device
        self.model = self.build_model(model_name,num_layers,use_pretrained_embeddings,freeze_pretrained_embeddings)
        self.optimizer = self.build_optimizer(optimizer,lr)

        self.scheduler_flag = scheduler
        if self.scheduler_flag:
            self.scheduler = self.build_scheduler()
        
        self.tokenizer = tokenizer

        # Resize token embeddings, in case there is a disparity with the optimizer
        self.model.resize_token_embeddings(len(self.tokenizer))

        assert self.model.transformer.wte.weight.shape[0] == len(self.tokenizer), "Model's embeddings should be the same as tokenizer's embeddings"

        self.args = args

        self.losses = {
            'train_loss' : [],
            'val_loss' : []
        }

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
                    model.transformer.wte.requires_grad = False
        
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
            
            outputs = self.__call__(tokenized_batch)
            loss = outputs.loss
            loss.backward()

            self.optimizer.step()

            if self.scheduler_flag:
                self.scheduler.step()

        else:
            
            outputs = self.__call__(tokenized_batch)
            loss = outputs.loss

        return loss

    def train(self):

        self.model.train()

    def eval(self):

        self.model.eval()