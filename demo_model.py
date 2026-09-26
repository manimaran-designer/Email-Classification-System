"""
Demo script to showcase the email classifier model architecture
and explain how to use it
"""
import torch
import torch.nn as nn
from email_classifier_model import EmailClassifier, count_parameters

def demo_model_architecture():
    """Demonstrate the model architecture"""
    print("="*70)
    print("EMAIL CLASSIFICATION TRANSFORMER MODEL DEMO")
    print("="*70)
    
    # Model hyperparameters
    VOCAB_SIZE = 5000
    NUM_CLASSES = 7  # work, personal, promotions, spam, social, finance, newsletter
    D_MODEL = 128
    NUM_HEADS = 8
    NUM_LAYERS = 3
    D_FF = 512
    MAX_LEN = 128
    DROPOUT = 0.1
    
    print("\nModel Configuration:")
    print(f"  - Vocabulary Size: {VOCAB_SIZE:,}")
    print(f"  - Number of Classes: {NUM_CLASSES}")
    print(f"  - Model Dimension (d_model): {D_MODEL}")
    print(f"  - Attention Heads: {NUM_HEADS}")
    print(f"  - Encoder Layers: {NUM_LAYERS}")
    print(f"  - Feed-Forward Dimension: {D_FF}")
    print(f"  - Max Sequence Length: {MAX_LEN}")
    print(f"  - Dropout Rate: {DROPOUT}")
    
    # Create model
    print("\nInitializing model...")
    model = EmailClassifier(
        vocab_size=VOCAB_SIZE,
        num_classes=NUM_CLASSES,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        max_len=MAX_LEN,
        dropout=DROPOUT,
        pad_idx=0
    )
    
    # Model statistics
    total_params = count_parameters(model)
    model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
    
    print("\nModel Statistics:")
    print(f"  - Total Parameters: {total_params:,}")
    print(f"  - Model Size: {model_size_mb:.2f} MB (float32)")
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 4
    seq_len = 50
    
    # Create dummy input
    dummy_input = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len))
    print(f"  - Input shape: {dummy_input.shape} [batch_size, seq_len]")
    
    # Forward pass
    output = model(dummy_input)
    print(f"  - Output shape: {output.shape} [batch_size, num_classes]")
    
    # Get predictions
    probs = torch.softmax(output, dim=1)
    predictions = output.argmax(dim=1)
    
    print(f"\nSample Predictions:")
    categories = ['work', 'personal', 'promotions', 'spam', 'social', 'finance', 'newsletter']
    for i in range(batch_size):
        pred_class = predictions[i].item()
        confidence = probs[i, pred_class].item()
        print(f"  Sample {i+1}: Predicted '{categories[pred_class]}' (confidence: {confidence:.3f})")
    
    # Attention weights
    attention_weights = model.get_attention_weights()
    print(f"\nAttention Mechanism:")
    print(f"  - Number of layers with attention: {len(attention_weights)}")
    if attention_weights:
        attn_shape = attention_weights[0].shape
        print(f"  - Attention weights shape per layer: {attn_shape}")
        print(f"    [batch_size={attn_shape[0]}, num_heads={attn_shape[1]}, seq_len={attn_shape[2]}, seq_len={attn_shape[3]}]")
    
    print("\n" + "="*70)
    print("Model Architecture Details:")
    print("="*70)
    print("\nComponent Breakdown:")
    print("1. Embedding Layer")
    print(f"   - Converts token indices to {D_MODEL}-dimensional vectors")
    print(f"   - Parameters: {VOCAB_SIZE:,} x {D_MODEL} = {VOCAB_SIZE * D_MODEL:,}")
    
    print("\n2. Positional Encoding")
    print(f"   - Adds position information to embeddings")
    print(f"   - Max sequence length: {MAX_LEN}")
    
    print("\n3. Transformer Encoder Layers (x{})".format(NUM_LAYERS))
    print("   a) Multi-Head Self-Attention")
    print(f"      - {NUM_HEADS} attention heads")
    print(f"      - Each head dimension: {D_MODEL // NUM_HEADS}")
    print("      - Query, Key, Value projections")
    print("   b) Feed-Forward Network")
    print(f"      - Hidden dimension: {D_FF}")
    print(f"      - Activation: ReLU")
    print("   c) Layer Normalization (x2 per layer)")
    print("   d) Residual Connections")
    
    print("\n4. Classification Head")
    print(f"   - Global Average Pooling")
    print(f"   - FC1: {D_MODEL} -> {D_MODEL // 2}")
    print(f"   - FC2: {D_MODEL // 2} -> {NUM_CLASSES}")
    
    print("\n" + "="*70)
    print("How to Use:")
    print("="*70)
    print("\nStep 1: Generate Dataset")
    print("  $ python3 generate_email_dataset.py")
    print("\nStep 2: Train Model")
    print("  $ python3 train_email_classifier.py")
    print("\nStep 3: Visualize Attention")
    print("  $ python3 visualize_attention.py")
    
    print("\n" + "="*70)
    print("Expected Performance:")
    print("="*70)
    print("  - Training Time: ~5-10 minutes (CPU)")
    print("  - Expected Accuracy: 90-93%")
    print("  - Epochs: 20")
    print("  - Best metric: F1-score per category")
    
    print("\n" + "="*70)
    print("Key Features:")
    print("="*70)
    print("  ✓ Custom Transformer implementation")
    print("  ✓ Multi-head attention mechanism")
    print("  ✓ Positional encoding")
    print("  ✓ Layer normalization")
    print("  ✓ Residual connections")
    print("  ✓ Attention weight visualization")
    print("  ✓ Complete training pipeline")
    print("  ✓ Comprehensive evaluation metrics")
    
    print("\n" + "="*70)
    print("Demo completed successfully!")
    print("="*70)

def explain_attention_mechanism():
    """Explain how attention mechanism works"""
    print("\n" + "="*70)
    print("ATTENTION MECHANISM EXPLAINED")
    print("="*70)
    
    print("\nWhat is Attention?")
    print("-" * 70)
    print("Attention allows the model to focus on different parts of the input")
    print("when making predictions. For email classification, this means:")
    print("  - Identifying keywords (e.g., 'meeting', 'sale', 'urgent')")
    print("  - Understanding context relationships")
    print("  - Weighing importance of different tokens")
    
    print("\nMulti-Head Attention:")
    print("-" * 70)
    print("Instead of one attention mechanism, we use 8 parallel attention")
    print("heads. Each head can learn different aspects:")
    print("  - Head 1 might focus on nouns")
    print("  - Head 2 might focus on verbs")
    print("  - Head 3 might focus on special characters")
    print("  - etc.")
    
    print("\nAttention Calculation:")
    print("-" * 70)
    print("For each token (word), we calculate:")
    print("  1. Query (Q): What am I looking for?")
    print("  2. Key (K): What do I offer?")
    print("  3. Value (V): What information do I carry?")
    print("\nAttention Score = softmax(Q × K^T / √d_k) × V")
    print("  - Tokens with higher scores are more relevant")
    print("  - softmax ensures scores sum to 1")
    
    print("\nExample: 'Meeting scheduled for tomorrow'")
    print("-" * 70)
    print("When processing 'scheduled':")
    print("  - High attention to 'meeting' (subject)")
    print("  - High attention to 'tomorrow' (time)")
    print("  - Lower attention to 'for' (grammatical)")
    
    print("\n" + "="*70)

def show_usage_example():
    """Show practical usage example"""
    print("\n" + "="*70)
    print("PRACTICAL USAGE EXAMPLE")
    print("="*70)
    
    code_example = """
# Load trained model
import torch
from email_classifier_model import EmailClassifier

checkpoint = torch.load('best_model.pth', weights_only=False)
tokenizer = checkpoint['tokenizer']
idx2label = checkpoint['idx2label']

# Initialize model
model = EmailClassifier(
    vocab_size=len(tokenizer.word2idx),
    num_classes=len(idx2label),
    d_model=128,
    num_heads=8,
    num_layers=3
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Classify new email
email = "Meeting scheduled for tomorrow at 3 PM"
encoded = tokenizer.encode(email)
input_tensor = torch.tensor(encoded).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)
    pred_idx = output.argmax(dim=1).item()
    predicted_category = idx2label[pred_idx]
    confidence = torch.softmax(output, dim=1)[0, pred_idx].item()

print(f"Category: {predicted_category}")
print(f"Confidence: {confidence:.2%}")
"""
    
    print("\nCode Example:")
    print(code_example)
    
    print("="*70)

if __name__ == "__main__":
    # Run demos
    demo_model_architecture()
    explain_attention_mechanism()
    show_usage_example()
    
    print("\n" + "="*70)
    print("For more information, see README_email_classifier.md")
    print("="*70)

