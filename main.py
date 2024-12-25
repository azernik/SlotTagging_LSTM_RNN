# src/main.py
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from gensim.downloader import load as load_glove
from datasets import SlotTaggingDataset, collate_fn, collate_fn_inference
from models import SlotTagger
from training import train_model
from inference import inference
from utils import set_seed, create_embedding_matrix

def main(train_data: str, test_data: str, output_file: str):
    set_seed(42)

    # load and prepare data
    print("Loading datasets...")
    df = pd.read_csv(train_data)
    df_train, df_val = df.sample(frac=0.9, random_state=42), df.drop(df.sample(frac=0.9, random_state=42).index)
    df_test = pd.read_csv(test_data)

    train_dataset = SlotTaggingDataset(df_train, training=True)
    val_dataset = SlotTaggingDataset(df_val, token_vocab=train_dataset.token_vocab, tag_vocab=train_dataset.tag_vocab, training=False)
    test_dataset = SlotTaggingDataset(df_test, token_vocab=train_dataset.token_vocab, tag_vocab=train_dataset.tag_vocab, training=False)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn_inference)

    # load GloVe embeddings
    print("Creating embedding matrix...")
    word_vectors = load_glove('glove-wiki-gigaword-300')
    embedding_matrix = create_embedding_matrix(train_dataset.token_vocab, word_vectors)

    # initialize model
    model = SlotTagger(
        vocab_size=len(train_dataset.token_vocab),
        tagset_size=len(train_dataset.tag_vocab),
        embedding_dim=300,
        hidden_dim=400,
        dropout_prob=0.5,
        pretrained_embeddings=embedding_matrix
    )

    # train the model
    print("Training the model...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    trained_model = train_model(model, train_loader, val_loader, optimizer, train_dataset.tag_vocab)

    # run inference
    print("Running inference...")
    inference(trained_model, test_loader, output_file, train_dataset.tag_vocab)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Slot Tagging Model")
    parser.add_argument("train_data", type=str, help="Path to training data CSV")
    parser.add_argument("test_data", type=str, help="Path to test data CSV")
    parser.add_argument("output", type=str, help="Path to output predictions")
    args = parser.parse_args()

    main(args.train_data, args.test_data, args.output)
