This project focuses on slot tagging in natural language utterances, a sequence labeling task that aims to identify entities mentioned in user queries in an automated system. The implemented model predicts IOB (Inside, Outside, Beginning) slot tags for each token in a sentence, using a BiLSTM-CRF architecture enhanced with pretrained GloVe embeddings.

This project was originally implemented as part of a Deep Learning master's course.
For a detailed description of the methodology, experiments, and results, please refer to the [project report](https://drive.google.com/file/d/1skIcxm2SmVEhci-tO3i1qoYztEfeF7XD/view?usp=drive_link).

### Introduction
For a detailed description of the methodology, experiments, and results, please refer to the [project report](https://drive.google.com/file/d/1skIcxm2SmVEhci-tO3i1qoYztEfeF7XD/view?usp=drive_link).

Slot tagging involves assigning a semantic label to each token in a natural language sentence. For example:
```
show me movies directed by Woody Allen recently.
```
The model should produce the following labels:
```
show    O
me      O
movies  O
directed O
by      O
Woody   B_director
Allen   I_director
recently B_release_year
```


### Dataset
The dataset comprises CSV files containing:
- **utterances**: Natural language sentences.
- **IOB Slot tags**: Corresponding slot labels for each token (optional in test data).

Key challenges include:
- **Class Imbalance**: Over 70% of tokens are labeled `O`, with rare labels like `B_release_year` making up less than 1%.
- **Sequence Length Variability**: Utterances range from 1 to 21 tokens, requiring careful handling of padding and batching.


### Model Architecture
The final model is a **BiLSTM-CRF**, designed to effectively handle sequence labeling:
- **Embedding Layer**: 300-dimensional GloVe embeddings, fine-tuned during training to adapt to the slot-tagging task.
- **Bidirectional LSTM Layer**: Captures contextual information in both forward and backward directions, with a hidden size of 400 per direction.
- **Dropout Layer**: Regularization with a dropout rate of 0.5 to prevent overfitting.
- **CRF Layer**: Ensures coherent sequence predictions by modeling dependencies between tags.

### Experiments and Techniques
Multiple models were developed during experimentation, culminating in the final BiLSTM-CRF model:
1. **Baseline RNN**:
   - Randomly initialized embeddings (100-dimensional).
   - Simple RNN architecture with a validation F1 score of ~0.733.
2. **RNN with Frozen GloVe Embeddings**:
   - Incorporated pretrained 100-dimensional GloVe embeddings, achieving a validation F1 score of ~0.771.
3. **Final Model (BiLSTM-CRF)**:
   - Trainable 300-dimensional GloVe embeddings.
   - Bidirectional LSTM with CRF, achieving a validation F1 score of ~0.917 and test F1 score of ~0.855.

Hyperparameter tuning included testing different learning rates, dropout rates, and optimizers (Adam, AdamW, RMSprop).

### Training and Evaluation
- **Metrics**: Evaluated using F1 score, computed with the seqeval library.
- **Early Stopping**: Applied with a patience of 10 epochs based on validation F1 score.
- **Challenges**: Addressed class imbalance through careful model and hyperparameter selection.

### Results
The final model demonstrated significant improvements over simpler architectures, particularly for rare slot tags:
- **Validation F1**: 0.917
- **Test F1**: 0.855

Tag-specific performance analysis highlighted strong results for frequent tags (`B_movie`, `B_director`) and improvements for rare tags like `B_mpaa_rating`.

### Code and Usage
To train and evaluate the model:
```
python src/main.py <train_data> <test_data> <output_file>
```
- `<train_data>`: Path to the training dataset.
- `<test_data>`: Path to the test dataset.
- `<output_file>`: Path to save predicted slot tags.
