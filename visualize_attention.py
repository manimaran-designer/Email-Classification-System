"""
Visualize attention weights from trained model
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from email_classifier_model import EmailClassifier
from train_email_classifier import Tokenizer  # Import Tokenizer class for unpickling

def visualize_attention_heatmap(attention_weights, tokens, layer_idx=0, head_idx=0, 
                               save_path='attention_heatmap.png'):
    """
    Visualize attention weights as heatmap
    
    Args:
        attention_weights: Attention weights from model [num_layers, batch_size, num_heads, seq_len, seq_len]
        tokens: List of tokens for the sequence
        layer_idx: Which transformer layer to visualize
        head_idx: Which attention head to visualize
        save_path: Path to save the visualization
    """
    # Get attention for specific layer and head
    # Shape: [seq_len, seq_len]
    attn = attention_weights[layer_idx][0, head_idx].cpu().numpy()
    
    # Limit to actual tokens (remove padding)
    seq_len = min(len(tokens), attn.shape[0])
    attn = attn[:seq_len, :seq_len]
    tokens = tokens[:seq_len]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot heatmap
    im = ax.imshow(attn, cmap='viridis', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(seq_len))
    ax.set_yticks(np.arange(seq_len))
    ax.set_xticklabels(tokens, rotation=90, ha='right', fontsize=8)
    ax.set_yticklabels(tokens, fontsize=8)
    
    # Labels
    ax.set_xlabel('Key (Attended To)', fontsize=12)
    ax.set_ylabel('Query (Attending From)', fontsize=12)
    ax.set_title(f'Attention Weights - Layer {layer_idx+1}, Head {head_idx+1}', 
                 fontsize=14, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight', rotation=270, labelpad=20, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Attention heatmap saved to {save_path}")
    plt.close()


def visualize_attention_heads(attention_weights, tokens, layer_idx=0, num_heads=8,
                              save_path='attention_heads.png'):
    """
    Visualize all attention heads for a specific layer
    
    Args:
        attention_weights: Attention weights from model
        tokens: List of tokens for the sequence
        layer_idx: Which transformer layer to visualize
        num_heads: Number of attention heads
        save_path: Path to save the visualization
    """
    # Limit tokens to non-padding
    seq_len = min(len(tokens), attention_weights[layer_idx].shape[-1])
    tokens = tokens[:seq_len]
    
    # Create subplot grid
    cols = 4
    rows = (num_heads + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows*3))
    axes = axes.flatten() if num_heads > 1 else [axes]
    
    for head_idx in range(num_heads):
        attn = attention_weights[layer_idx][0, head_idx].cpu().numpy()
        attn = attn[:seq_len, :seq_len]
        
        ax = axes[head_idx]
        im = ax.imshow(attn, cmap='viridis', aspect='auto')
        ax.set_title(f'Head {head_idx+1}', fontsize=10, fontweight='bold')
        ax.set_xticks(np.arange(seq_len))
        ax.set_yticks(np.arange(seq_len))
        ax.set_xticklabels(tokens, rotation=90, ha='right', fontsize=6)
        ax.set_yticklabels(tokens, fontsize=6)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Hide unused subplots
    for idx in range(num_heads, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(f'All Attention Heads - Layer {layer_idx+1}', 
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Attention heads visualization saved to {save_path}")
    plt.close()


def visualize_attention_patterns(attention_weights, tokens, save_path='attention_patterns.png'):
    """
    Visualize average attention patterns across all layers
    
    Args:
        attention_weights: List of attention weights from all layers
        tokens: List of tokens for the sequence
        save_path: Path to save the visualization
    """
    num_layers = len(attention_weights)
    seq_len = min(len(tokens), attention_weights[0].shape[-1])
    tokens = tokens[:seq_len]
    
    # Average attention across all heads and layers
    all_attn = []
    for layer_attn in attention_weights:
        # Average across heads: [batch_size, num_heads, seq_len, seq_len] -> [seq_len, seq_len]
        layer_avg = layer_attn[0].mean(dim=0).cpu().numpy()
        all_attn.append(layer_avg[:seq_len, :seq_len])
    
    # Create figure
    fig, axes = plt.subplots(1, num_layers, figsize=(5*num_layers, 4))
    if num_layers == 1:
        axes = [axes]
    
    for idx, (attn, ax) in enumerate(zip(all_attn, axes)):
        im = ax.imshow(attn, cmap='viridis', aspect='auto')
        ax.set_title(f'Layer {idx+1}\n(Avg across heads)', fontsize=11, fontweight='bold')
        ax.set_xticks(np.arange(seq_len))
        ax.set_yticks(np.arange(seq_len))
        ax.set_xticklabels(tokens, rotation=90, ha='right', fontsize=8)
        ax.set_yticklabels(tokens, fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    fig.suptitle('Attention Patterns Across Layers', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Attention patterns visualization saved to {save_path}")
    plt.close()


def analyze_attention_example():
    """Full example of loading model and visualizing attention"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    print("Loading trained model...")
    checkpoint = torch.load('best_model.pth', map_location=device, weights_only=False)
    tokenizer = checkpoint['tokenizer']
    idx2label = checkpoint['idx2label']
    label2idx = checkpoint['label2idx']
    
    # Create model
    model = EmailClassifier(
        vocab_size=len(tokenizer.word2idx),
        num_classes=len(idx2label),
        d_model=128,
        num_heads=8,
        num_layers=3,
        d_ff=512,
        max_len=128,
        dropout=0.1,
        pad_idx=0
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully!")
    
    # Example emails to visualize
    example_emails = [
        "Meeting scheduled for tomorrow at 3 PM. Please confirm your attendance.",
        "SALE: Up to 50% off on all items! Limited time offer. Shop now!",
        "You've won $1,000,000! Claim your prize now by clicking here!!!",
    ]
    
    for email_idx, email_text in enumerate(example_emails):
        print(f"\n{'='*70}")
        print(f"Email {email_idx+1}: {email_text}")
        
        # Tokenize
        tokens = tokenizer.preprocess_text(email_text)
        encoded = tokenizer.encode(email_text)
        encoded_tensor = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(device)
        
        # Get prediction and attention weights
        with torch.no_grad():
            output = model(encoded_tensor)
            pred_idx = output.argmax(dim=1).item()
            predicted_label = idx2label[pred_idx]
            confidence = torch.softmax(output, dim=1)[0, pred_idx].item()
        
        print(f"Predicted: {predicted_label} (confidence: {confidence:.3f})")
        
        # Get attention weights
        attention_weights = model.get_attention_weights()
        
        # Visualize
        prefix = f'email_{email_idx+1}'
        
        # Single head heatmap
        visualize_attention_heatmap(
            attention_weights, tokens, layer_idx=0, head_idx=0,
            save_path=f'{prefix}_layer1_head1.png'
        )
        
        # All heads for layer 1
        visualize_attention_heads(
            attention_weights, tokens, layer_idx=0, num_heads=8,
            save_path=f'{prefix}_all_heads_layer1.png'
        )
        
        # Pattern across layers
        visualize_attention_patterns(
            attention_weights, tokens,
            save_path=f'{prefix}_patterns.png'
        )
    
    print(f"\n{'='*70}")
    print("Attention visualization completed!")
    print(f"{'='*70}")


if __name__ == "__main__":
    # Check if model exists
    import os
    if not os.path.exists('best_model.pth'):
        print("Error: best_model.pth not found!")
        print("Please train the model first using: python train_email_classifier.py")
    else:
        analyze_attention_example()

