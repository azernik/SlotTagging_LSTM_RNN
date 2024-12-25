# training.py
import torch
from seqeval.metrics import f1_score
from typing import Dict, Any
from datasets import ids_to_tags

def train_model(model, train_loader, val_loader, optimizer, tag_vocab, num_epochs=200, patience=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    best_val_f1 = 0
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        for token_ids, tag_ids, lengths in train_loader:
            token_ids, tag_ids = token_ids.to(device), tag_ids.to(device)
            optimizer.zero_grad()
            loss = model(token_ids, lengths, tags=tag_ids)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        model.eval()
        total_val_loss = 0
        all_preds = []
        all_true_tags = []

        with torch.no_grad():
            for token_ids, tag_ids, lengths in val_loader:
                token_ids, tag_ids = token_ids.to(device), tag_ids.to(device)
                loss = model(token_ids, lengths, tags=tag_ids)
                total_val_loss += loss.item()
                predictions = model(token_ids, lengths)

                # process predictions and true tags
                for i, length in enumerate(lengths):
                    masked_preds = predictions[i][:length]
                    masked_tags = tag_ids[i][:length].cpu().tolist()
                    all_preds.append(masked_preds)
                    all_true_tags.append(masked_tags)

        # convert IDs to tags for evaluation
        all_preds = ids_to_tags(all_preds, tag_vocab)
        all_true_tags = ids_to_tags(all_true_tags, tag_vocab)

        avg_val_loss = total_val_loss / len(val_loader)
        val_f1 = f1_score(all_true_tags, all_preds)

        print(f"Epoch {epoch + 1}: Train Loss = {total_train_loss:.3f}, Val Loss = {avg_val_loss:.3f}, Val F1 = {val_f1:.3f}")

        # check for improvement
        if val_f1 > best_val_f1 or (val_f1 == best_val_f1 and avg_val_loss < best_val_loss):
            best_val_loss = avg_val_loss
            best_val_f1 = val_f1
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(best_model_state)
    return model