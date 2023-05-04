import argparse
from utils import *
from model import *
from train import *
import tqdm
from tqdm import tqdm
import wandb

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Inputs for running files")

    parser.add_argument("--dataset_name",default='yelp_review_full')
    parser.add_argument("--model_name",default="GPT2")
    parser.add_argument("--optimizer",default="AdamW")
    parser.add_argument("--lr",default=3e-4)
    parser.add_argument("--use_pretrained_embeddings",default=True)
    parser.add_argument("--num_epochs",default=200,type=int)
    parser.add_argument("--scheduler",default=True)
    parser.add_argument("--tokenizer_max_length",default=64,type=int)
    parser.add_argument("--dataset_frac",default=0.2,type=float)
    parser.add_argument("--batch_size",default=128,type=int)
    parser.add_argument("--num_layers",default=6,type=int)

    args = parser.parse_args()

    print(args)

    assert torch.cuda.is_available(), "Cuda is not available"

    # Load dataset and convert to torch
    dataset = load_hf_dataset(args.dataset_name)
    dataset = prepare_datasets(dataset,args.dataset_frac)    

    # Load dataloaders
    train_loader, val_loader, test_loader = prepare_dataloaders(dataset,args.batch_size)

    # Get tokenizer
    tokenizer = get_tokenizer(args.model_name)

    # Get device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print("Device :",device)

    # Build Consolidate Model Classes
    BaseModel = ConsolidatedModelClass(
        model_name=args.model_name,
        num_layers=args.num_layers,
        use_pretrained_embeddings=False,
        optimizer='AdamW',
        lr=args.lr,
        tokenizer=tokenizer,
        scheduler=args.scheduler,
        args=args,
        device=device
    )

    BaseModelWithEmbeddings = ConsolidatedModelClass(
        model_name=args.model_name,
        num_layers=args.num_layers,
        use_pretrained_embeddings=True,
        optimizer='AdamW',
        lr=args.lr,
        tokenizer=tokenizer,
        scheduler=args.scheduler,
        args=args,
        device=device
    )

    models = {
        'base':BaseModel,
        'base_with_embeddings':BaseModelWithEmbeddings
        }

    for epoch in tqdm(range(args.num_epochs)):

        train_models(models,train_loader,args)
        evaluate_models(models,test_loader,args)

        for model in models:
            print(f"\n model {model} has train loss : {models[model].losses['train_loss'][-1]} \n \
                                  and test loss : {models[model].losses['val_loss'][-1]}")