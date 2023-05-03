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

    print(args)

    # Build Consolidate Model Classes
    BaseModel = ConsolidatedModelClass(
        model_name=args.model_name,
        use_pretrained_embeddings=False,
        optimizer='AdamW',
        lr=args.lr,
        tokenizer=tokenizer,
        scheduler=args.scheduler
    )

    BaseModelWithEmbeddings = ConsolidatedModelClass(
        model_name=args.model_name,
        use_pretrained_embeddings=True,
        optimizer='AdamW',
        lr=args.lr,
        tokenizer=tokenizer,
        scheduler=args.scheduler
    )

    for epoch in tqdm(range(args.num_epochs)):

        for batch in tqdm(train_loader):

            tokenized_batch = tokenizer(batch['text'],return_tensors='pt',padding=True,max_length=1024)

            BaseModel.step(tokenized_batch)
            BaseModelWithEmbeddings.step(tokenized_batch)