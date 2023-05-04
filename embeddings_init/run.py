import argparse
from utils import *
from train import *
import tqdm
from tqdm import tqdm

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Inputs for running files")
    parser.add_argument("--dataset_name",default='yelp_review_full')
    parser.add_argument("--model_name",default="GPT2")
    parser.add_argument("--optimizer",default="AdamW")
    parser.add_argument("--lr",default=3e-4)
    parser.add_argument("--use_pretrained_embeddings",default=True)
    parser.add_argument("--num_epochs",default=200)
    parser.add_argument("--scheduler",default=True)

    args = parser.parse_args()

    # Load dataset and convert to torch
    dataset = load_hf_dataset(args.dataset_name)
    prepare_datasets(dataset)

    # Load dataloaders
    train_loader, val_loader, test_loader = prepare_dataloaders(dataset)

    # Get tokenizer
    tokenizer = get_tokenizer(args.model_name)

    # Get device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Build Consolidate Model Classes
    BaseModel = ConsolidatedModelClass(
        model_name=args.model_name,
        use_pretrained_embeddings=False,
        optimizer='AdamW',
        lr=args.lr,
        tokenizer=tokenizer,
        scheduler=args.scheduler,
        device=device
    )

    BaseModelWithEmbeddings = ConsolidatedModelClass(
        model_name=args.model_name,
        use_pretrained_embeddings=True,
        optimizer='AdamW',
        lr=args.lr,
        tokenizer=tokenizer,
        scheduler=args.scheduler,
        device=device
    )

    for epoch in tqdm(range(args.num_epochs)):

        BaseModel.train()
        BaseModelWithEmbeddings.train()

        for batch in tqdm(train_loader):

            tokenized_batch = tokenizer(batch['text'],return_tensors='pt',padding=True,max_length=1024)

            train_base_loss = BaseModel.step(tokenized_batch)
            train_base_with_embeddings_loss = BaseModelWithEmbeddings.step(tokenized_batch)

        BaseModel.eval()
        BaseModelWithEmbeddings.eval()

        train_val_losses = []
        train_val_with_embeddings_losses = []

        for batch in tqdm(test_loader):

            with torch.no_grad():
                tokenized_batch = tokenizer(batch['text'],return_tensors='pt',padding=True,max_length=1024)

                train_val_loss = BaseModel.step(tokenized_batch)
                train_val_with_embeddings_loss = BaseModelWithEmbeddings.step(tokenized_batch)

                train_val_losses.append(train_val_loss.item())
                train_val_with_embeddings_losses.append(train_val_with_embeddings_loss.item())

        print("Average Val Loss Base", train_val_losses.mean())
        print("Average Val Loss Base With Embeddings", train_val_with_embeddings_losses.mean())