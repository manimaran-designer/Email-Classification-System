# Email Classification with Transformer Attention Model

A complete PyTorch-based email classification system using transformer architecture with multi-head attention mechanism.

## Features

- **Custom Transformer Architecture**: Built from scratch with multi-head attention
- **7 Email Categories**: work, personal, promotions, spam, social, finance, newsletter
- **Complete Pipeline**: Data generation, preprocessing, training, and evaluation
- **Attention Visualization**: Store and access attention weights for interpretability
- **Comprehensive Metrics**: Accuracy, precision, recall, F1-score, confusion matrix

## Project Structure

```
section3/
├── generate_email_dataset.py      # Generate synthetic email dataset
├── email_classifier_model.py      # Transformer model with attention
├── train_email_classifier.py      # Complete training pipeline
├── email_dataset.csv              # Generated dataset (2000 emails)
├── best_model.pth                 # Trained model checkpoint
├── training_history.png           # Training/validation curves
└── confusion_matrix.png           # Classification results
```

## Installation

```bash
pip install -r requirements_email_classifier.txt
```

## Usage

### 1. Generate Dataset

```bash
python generate_email_dataset.py
```

This creates `email_dataset.csv` with 2000 synthetic emails across 7 categories.

### 2. Train Model

```bash
python train_email_classifier.py
```

This will:
- Load and split data (70% train, 15% val, 15% test)
- Build vocabulary from training data
- Train transformer model for 20 epochs
- Save best model based on validation accuracy
- Generate training history plots
- Evaluate on test set with detailed metrics
- Show sample predictions

### 3. Use Trained Model

```python
import torch
from email_classifier_model import EmailClassifier

# Load checkpoint
checkpoint = torch.load('best_model.pth', weights_only=False)
tokenizer = checkpoint['tokenizer']
idx2label = checkpoint['idx2label']

# Create model
model = EmailClassifier(
    vocab_size=len(tokenizer.word2idx),
    num_classes=len(idx2label),
    d_model=128,
    num_heads=8,
    num_layers=3
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Predict
email_text = "Meeting scheduled for tomorrow at 3 PM"
encoded = tokenizer.encode(email_text)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)

with torch.no_grad():
    output = model(encoded_tensor)
    pred_idx = output.argmax(dim=1).item()
    predicted_label = idx2label[pred_idx]
    
print(f"Predicted category: {predicted_label}")
```

## Model Architecture

### Transformer Components

1. **Embedding Layer**: Converts token indices to dense vectors (d_model=128)
2. **Positional Encoding**: Adds position information to embeddings
3. **Multi-Head Attention** (8 heads): Captures relationships between tokens
4. **Feed-Forward Network**: Position-wise transformations
5. **Layer Normalization**: Stabilizes training
6. **Classification Head**: Maps to output classes

### Key Parameters

- Vocabulary Size: 5000 tokens
- Max Sequence Length: 128 tokens
- Model Dimension: 128
- Number of Attention Heads: 8
- Number of Encoder Layers: 3
- Feed-Forward Dimension: 512
- Dropout: 0.1

### Model Size

- Total Parameters: ~800K
- Model Size: ~3 MB

## Training Details

- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: CrossEntropyLoss
- **Batch Size**: 32
- **Epochs**: 20
- **Data Split**: 70% train, 15% validation, 15% test
- **Early Stopping**: Best model saved based on validation accuracy

## Dataset

### Email Categories

1. **work**: Business emails, meetings, projects, deadlines
2. **personal**: Friends, family, casual conversations
3. **promotions**: Sales, discounts, marketing campaigns
4. **spam**: Phishing, scams, unsolicited offers
5. **social**: Social media notifications, mentions
6. **finance**: Banking, transactions, statements
7. **newsletter**: Subscriptions, updates, digests

### Dataset Statistics

- Total Samples: 2000
- Distribution: ~285 samples per category
- Train: 1400 samples (70%)
- Validation: 300 samples (15%)
- Test: 300 samples (15%)

## Expected Performance

With the default configuration, you should achieve:
- Training Accuracy: ~95-98%
- Validation Accuracy: ~90-93%
- Test Accuracy: ~90-93%

Performance varies by category:
- Spam: Highest accuracy (~98%)
- Finance: High accuracy (~95%)
- Work: High accuracy (~95%)
- Social: Moderate accuracy (~88%)

## Attention Mechanism

The model includes multi-head attention with 8 heads per layer. Attention weights are stored during forward pass and can be visualized:

```python
# Get attention weights
attention_weights = model.get_attention_weights()
# Shape: [num_layers, batch_size, num_heads, seq_len, seq_len]

# Visualize attention for first sample, first head, first layer
import matplotlib.pyplot as plt
import seaborn as sns

attn = attention_weights[0][0, 0].cpu().numpy()  # First layer, first sample, first head
sns.heatmap(attn, cmap='viridis')
plt.xlabel('Key Position')
plt.ylabel('Query Position')
plt.title('Attention Weights')
plt.show()
```

## Customization

### Change Model Size

```python
model = EmailClassifier(
    vocab_size=5000,
    num_classes=7,
    d_model=256,        # Larger model
    num_heads=16,       # More attention heads
    num_layers=6,       # Deeper network
    d_ff=1024,          # Larger feed-forward
    dropout=0.1
)
```

### Add New Categories

1. Update `CATEGORIES` dict in `generate_email_dataset.py`
2. Regenerate dataset
3. Retrain model (num_classes will auto-adjust)

### Fine-tune on Real Data

```python
# Load your real email data
df = pd.read_csv('real_emails.csv')  # Columns: email, label

# Use existing tokenizer from checkpoint
checkpoint = torch.load('best_model.pth', weights_only=False)
tokenizer = checkpoint['tokenizer']

# Continue training
# ... (follow training pipeline)
```

## Troubleshooting

### PyTorch 2.6+ Loading Error
If you get an error about `weights_only` when loading the model:
```python
# Use weights_only=False since we trust our own checkpoints
checkpoint = torch.load('best_model.pth', weights_only=False)
```
This is required in PyTorch 2.6+ which changed the default for security reasons. Since we're loading our own trained model (not from untrusted sources), it's safe to use `weights_only=False`.

### CUDA Out of Memory
Reduce batch size or model dimensions:
```python
BATCH_SIZE = 16  # Instead of 32
D_MODEL = 64     # Instead of 128
```

### Low Accuracy
- Increase training epochs
- Adjust learning rate
- Add more data
- Increase model capacity

### Overfitting
- Increase dropout rate
- Add weight decay to optimizer
- Use data augmentation
- Reduce model size

## References

- Attention Is All You Need (Vaswani et al., 2017)
- PyTorch Documentation: https://pytorch.org/docs/
- Transformer Architecture: https://arxiv.org/abs/1706.03762

## License

MIT License - Feel free to use and modify for your projects!

