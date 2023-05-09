import argparse
from utils import *
from model import *
from train import *
import tqdm
from tqdm import tqdm
import wandb
import os
import copy

os.environ["WANDB_SILENT"] = "true" # suppress wandb outputs
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" # set CUDA to device : 1

# git token : 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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Inputs for running files")

    parser.add_argument("--dataset_name",default='yelp_review_full')
    parser.add_argument("--model_name",default="GPT2")
    parser.add_argument("--optimizer",default="AdamW")
    parser.add_argument("--max_lr",default=3e-4,type=float)
    parser.add_argument("--use_pretrained_embeddings",default=True,type=str_to_bool)
    parser.add_argument("--freeze_pretrained_embeddings",default=True,type=str_to_bool)
    parser.add_argument("--num_epochs",default=200,type=int)
    parser.add_argument("--scheduler",default=True,type=str_to_bool)
    parser.add_argument("--tokenizer_max_length",default=64,type=int)
    parser.add_argument("--dataset_frac",default=0.2,type=float)
    parser.add_argument("--batch_size",default=128,type=int)
    parser.add_argument("--num_layers",default=6,type=int)
    parser.add_argument("--use_wandb",default=True,type=str_to_bool)
    parser.add_argument("--seed",default=42,type=int)
    parser.add_argument("--wandb_project_name",default="gptembeddings_with_monitoring")
    parser.add_argument("--num_workers",default=8,type=int)
    parser.add_argument("--run_name",default=None)
    parser.add_argument("--warmup_steps",default=100,type=int)
    parser.add_argument("--max_grad_norm",default=2.0,type=float)
    parser.add_argument("--early_stopping",default=True,type=str_to_bool)
    parser.add_argument("--patience",default=15,type=int)

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Print args
    for key,value in vars(args).items():
        print(key,"=>",value)

    if args.run_name:
        run_name = args.run_name
    else:
        run_name = f"batch_size : {args.batch_size}, \
                    pretrain : {args.use_pretrained_embeddings}, \
                    freeze : {args.freeze_pretrained_embeddings}, \
                    max_lr : {args.max_lr}, \
                    seed : {args.seed}, \
                    dataset : {args.dataset_name}, \
                    num_epochs : {args.num_epochs}, \
                    "
    
    print("Logging as run name:", run_name)

    if args.use_wandb:
        # Start wandb
        run = wandb.init(
            name=run_name,
            project=args.wandb_project_name,
            config=dict(vars(args).items())
        )

    # Only run if GPU is available
    assert torch.cuda.is_available(), "Cuda is not available"

    # ------------------------------- Load Datasets ------------------------------ #
    dataset = load_hf_dataset(args.dataset_name)
    dataset = prepare_datasets(dataset,args.dataset_frac)   

    print("Length of training dataset:", len(dataset['train']))
    print("Length of validation dataset:", len(dataset['test']))

    # ----------------------------- Load dataloaders ----------------------------- #
    train_loader, val_loader, test_loader = prepare_dataloaders(dataset,batch_size=args.batch_size,num_workers=args.num_workers)

    # ------------------------------- Get tokenizer ------------------------------ #
    tokenizer = get_tokenizer(args.model_name)

    # -------------------------------- Get device -------------------------------- #
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Device :",device)

    # ---------------------------------------------------------------------------- #
    #                                  Init Models                                 #
    # ---------------------------------------------------------------------------- #

    Model = ConsolidatedModelClass(
        model_name=args.model_name,
        num_layers=args.num_layers,
        use_pretrained_embeddings=args.use_pretrained_embeddings,
        freeze_pretrained_embeddings=args.freeze_pretrained_embeddings,
        optimizer='AdamW',
        max_lr=args.max_lr,
        tokenizer=tokenizer,
        scheduler=args.scheduler,
        args=args,
    )

    # ---------------------------------------------------------------------------- #
    #                                 Training loop                                #
    # ---------------------------------------------------------------------------- #
    Model.train(train_loader,test_loader)


## 3 separate runs, with same seed
## check workers, only dataloaders speed up, num_workers for gauatama max 16, (check)
## batch_size : can change? lower batch sizes might be better (generalization?)
## Separate sweeps for each model, find optimal and compare those
## tuning : lr, wdecay default (0.1) try higher?