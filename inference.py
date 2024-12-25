# inference.py
import os
import torch
from datasets import ids_to_tags

def inference(model, test_loader, output_file, tag_vocab):
    """
    Runs inference using the trained model and saves predictions to a file.
    """
    # make sure output dir exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    all_preds = []

    with torch.no_grad():
        for token_ids, lengths in test_loader:
            token_ids = token_ids.to(device)
            predictions = model(token_ids, lengths)
            trimmed_predictions = [pred[:length] for pred, length in zip(predictions, lengths)]
            trimmed_tags = ids_to_tags(trimmed_predictions, tag_vocab)
            all_preds.extend(trimmed_tags)

    with open(output_file, 'w') as f:
        for tags in all_preds:
            f.write(" ".join(tags) + "\n")
    print(f"Inference complete. Predictions saved to {output_file}")