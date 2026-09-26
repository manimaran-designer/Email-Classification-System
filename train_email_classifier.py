"""
Complete training pipeline for email classification with transformers
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import time
import os

from email_classifier_model import EmailClassifier, count_parameters

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class Tokenizer:
    """Simple tokenizer for text processing"""
    def __init__(self, vocab_size=5000, max_len=128):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.word_counts = Counter()
    
    def preprocess_text(self, text):
        """Basic text preprocessing"""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and numbers, keep only letters and spaces
        text = re.sub(r'[^a-z\s]', '', text)
        # Tokenize by whitespace
        tokens = text.split()
        return tokens
    
    def build_vocab(self, texts):
        """Build vocabulary from texts"""
        print("Building vocabulary...")
        for text in texts:
            tokens = self.preprocess_text(text)
            self.word_counts.update(tokens)
        
        # Keep most common words
        most_common = self.word_counts.most_common(self.vocab_size - 2)  # -2 for PAD and UNK
        
        for idx, (word, count) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        
        print(f"Vocabulary size: {len(self.word2idx)}")
    
    def encode(self, text):
        """Convert text to sequence of indices"""
        tokens = self.preprocess_text(text)
        indices = [self.word2idx.get(token, 1) for token in tokens]  # 1 is UNK
        
        # Pad or truncate to max_len
        if len(indices) < self.max_len:
            indices += [0] * (self.max_len - len(indices))  # 0 is PAD
        else:
            indices = indices[:self.max_len]
        
        return indices
    
    def decode(self, indices):
        """Convert sequence of indices back to text"""
        words = [self.idx2word.get(idx, '<UNK>') for idx in indices if idx != 0]
        return ' '.join(words)


class EmailDataset(Dataset):
    """PyTorch Dataset for email classification"""
    def __init__(self, emails, labels, tokenizer, label_encoder):
        self.emails = emails
        self.labels = labels
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
    
    def __len__(self):
        return len(self.emails)
    
    def __getitem__(self, idx):
        email = self.emails[idx]
        label = self.labels[idx]
        
        # Encode email text
        encoded = self.tokenizer.encode(email)
        encoded_tensor = torch.tensor(encoded, dtype=torch.long)
        
        # Encode label
        label_idx = self.label_encoder[label]
        label_tensor = torch.tensor(label_idx, dtype=torch.long)
        
        return encoded_tensor, label_tensor


def load_and_split_data(data_path, test_size=0.15, val_size=0.15):
    """Load data and split into train/val/test sets"""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    print(f"Total samples: {len(df)}")
    print(f"\nLabel distribution:")
    print(df['label'].value_counts())
    
    # First split: separate test set
    train_val_emails, test_emails, train_val_labels, test_labels = train_test_split(
        df['email'].values,
        df['label'].values,
        test_size=test_size,
        random_state=42,
        stratify=df['label'].values
    )
    
    # Second split: separate validation set from train
    val_size_adjusted = val_size / (1 - test_size)
    train_emails, val_emails, train_labels, val_labels = train_test_split(
        train_val_emails,
        train_val_labels,
        test_size=val_size_adjusted,
        random_state=42,
        stratify=train_val_labels
    )
    
    print(f"\nDataset splits:")
    print(f"Train: {len(train_emails)} samples ({len(train_emails)/len(df)*100:.1f}%)")
    print(f"Val: {len(val_emails)} samples ({len(val_emails)/len(df)*100:.1f}%)")
    print(f"Test: {len(test_emails)} samples ({len(test_emails)/len(df)*100:.1f}%)")
    
    return (train_emails, train_labels), (val_emails, val_labels), (test_emails, test_labels)


def create_label_encoder(labels):
    """Create label to index mapping"""
    unique_labels = sorted(set(labels))
    label2idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx2label = {idx: label for label, idx in label2idx.items()}
    return label2idx, idx2label


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch_idx, (emails, labels) in enumerate(dataloader):
        emails, labels = emails.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(emails)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for emails, labels in dataloader:
            emails, labels = emails.to(device), labels.to(device)
            
            outputs = model(emails)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy, all_preds, all_labels


def plot_training_history(history, save_path='training_history.png'):
    """Plot training and validation metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(history['val_loss'], label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Accuracy', marker='o')
    ax2.plot(history['val_acc'], label='Val Accuracy', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history saved to {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, labels, save_path='confusion_matrix.png'):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def print_classification_report(y_true, y_pred, idx2label):
    """Print detailed classification metrics"""
    labels = [idx2label[i] for i in sorted(idx2label.keys())]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )
    
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT")
    print("="*70)
    print(f"{'Label':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-"*70)
    
    for i, label in enumerate(labels):
        print(f"{label:<15} {precision[i]:>10.3f} {recall[i]:>10.3f} {f1[i]:>10.3f} {support[i]:>10.0f}")
    
    print("-"*70)
    avg_precision = precision.mean()
    avg_recall = recall.mean()
    avg_f1 = f1.mean()
    total_support = support.sum()
    
    print(f"{'Macro Avg':<15} {avg_precision:>10.3f} {avg_recall:>10.3f} {avg_f1:>10.3f} {total_support:>10.0f}")
    print("="*70)


def predict_single_email(model, tokenizer, label_encoder, idx2label, email_text, device):
    """Predict category for a single email"""
    model.eval()
    
    # Encode email
    encoded = tokenizer.encode(email_text)
    encoded_tensor = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(encoded_tensor)
        probabilities = torch.softmax(output, dim=1)
        pred_idx = output.argmax(dim=1).item()
        confidence = probabilities[0][pred_idx].item()
    
    predicted_label = idx2label[pred_idx]
    
    # Get top 3 predictions
    top3_probs, top3_indices = torch.topk(probabilities[0], 3)
    top3_labels = [(idx2label[idx.item()], prob.item()) for idx, prob in zip(top3_indices, top3_probs)]
    
    return predicted_label, confidence, top3_labels


def main():
    # Hyperparameters
    VOCAB_SIZE = 5000
    MAX_LEN = 128
    D_MODEL = 128
    NUM_HEADS = 8
    NUM_LAYERS = 3
    D_FF = 512
    DROPOUT = 0.1
    
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and split data
    data_path = 'email_dataset.csv'
    (train_emails, train_labels), (val_emails, val_labels), (test_emails, test_labels) = \
        load_and_split_data(data_path)
    
    # Create label encoder
    label2idx, idx2label = create_label_encoder(train_labels)
    num_classes = len(label2idx)
    print(f"\nNumber of classes: {num_classes}")
    print(f"Classes: {list(label2idx.keys())}")
    
    # Build tokenizer
    tokenizer = Tokenizer(vocab_size=VOCAB_SIZE, max_len=MAX_LEN)
    tokenizer.build_vocab(train_emails)
    
    # Create datasets
    train_dataset = EmailDataset(train_emails, train_labels, tokenizer, label2idx)
    val_dataset = EmailDataset(val_emails, val_labels, tokenizer, label2idx)
    test_dataset = EmailDataset(test_emails, test_labels, tokenizer, label2idx)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"\nDataLoader batches:")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Create model
    model = EmailClassifier(
        vocab_size=len(tokenizer.word2idx),
        num_classes=num_classes,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        max_len=MAX_LEN,
        dropout=DROPOUT,
        pad_idx=0
    ).to(device)
    
    print(f"\n{'='*70}")
    print("MODEL ARCHITECTURE")
    print(f"{'='*70}")
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Model size: {count_parameters(model) * 4 / 1024 / 1024:.2f} MB (float32)")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print(f"\n{'='*70}")
    print("TRAINING")
    print(f"{'='*70}")
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] ({epoch_time:.1f}s) - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'tokenizer': tokenizer,
                'label2idx': label2idx,
                'idx2label': idx2label,
            }, 'best_model.pth')
            print(f"  → Best model saved (Val Acc: {val_acc:.4f})")
    
    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time/60:.2f} minutes")
    
    # Plot training history
    plot_training_history(history)
    
    # Load best model for testing
    print(f"\n{'='*70}")
    print("TESTING")
    print(f"{'='*70}")
    checkpoint = torch.load('best_model.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']+1}")
    
    # Test
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Detailed metrics
    print_classification_report(test_labels, test_preds, idx2label)
    
    # Plot confusion matrix
    label_names = [idx2label[i] for i in sorted(idx2label.keys())]
    plot_confusion_matrix(test_labels, test_preds, label_names)
    
    # Test predictions on sample emails
    print(f"\n{'='*70}")
    print("SAMPLE PREDICTIONS")
    print(f"{'='*70}")
    
    sample_emails = [
        "Meeting scheduled for tomorrow at 3 PM. Please confirm your attendance.",
        "SALE: Up to 50% off on all items! Limited time offer. Shop now!",
        "Your monthly credit card statement is now available. Amount due: $250.",
        "Hey! How are you doing? Long time no see. Let's catch up soon!",
        "You've won $1,000,000! Claim your prize now by clicking here!!!",
        "Facebook: John commented on your post. See what they said.",
        "Weekly tech digest: Top stories you might have missed this week."
    ]
    
    for i, email in enumerate(sample_emails, 1):
        print(f"\n{i}. Email: {email[:80]}...")
        pred_label, confidence, top3 = predict_single_email(
            model, tokenizer, label2idx, idx2label, email, device
        )
        print(f"   Predicted: {pred_label} (confidence: {confidence:.3f})")
        print(f"   Top 3 predictions:")
        for label, prob in top3:
            print(f"     - {label}: {prob:.3f}")
    
    print(f"\n{'='*70}")
    print("Training completed successfully!")
    print(f"Model saved as: best_model.pth")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

