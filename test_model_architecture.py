"""
Test script to verify model architecture without full training
This can be run to quickly verify everything is working correctly
"""
import torch
import torch.nn as nn
import sys

def test_imports():
    """Test that all required imports work"""
    print("Testing imports...")
    try:
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        import matplotlib.pyplot as plt
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("  Please install: pip install -r requirements_email_classifier.txt")
        return False

def test_model_creation():
    """Test model instantiation"""
    print("\nTesting model creation...")
    try:
        from email_classifier_model import EmailClassifier, count_parameters
        
        model = EmailClassifier(
            vocab_size=5000,
            num_classes=7,
            d_model=128,
            num_heads=8,
            num_layers=3,
            d_ff=512,
            max_len=128,
            dropout=0.1
        )
        
        total_params = count_parameters(model)
        print(f"✓ Model created successfully")
        print(f"  Total parameters: {total_params:,}")
        return True, model
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False, None

def test_forward_pass(model):
    """Test forward pass through model"""
    print("\nTesting forward pass...")
    try:
        batch_size = 4
        seq_len = 50
        vocab_size = 5000
        
        # Create dummy input
        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Forward pass
        output = model(x)
        
        # Check output shape
        expected_shape = (batch_size, 7)
        if output.shape == expected_shape:
            print(f"✓ Forward pass successful")
            print(f"  Input shape: {x.shape}")
            print(f"  Output shape: {output.shape}")
            return True
        else:
            print(f"✗ Unexpected output shape: {output.shape} (expected {expected_shape})")
            return False
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False

def test_attention_weights(model):
    """Test attention weight extraction"""
    print("\nTesting attention mechanism...")
    try:
        batch_size = 2
        seq_len = 30
        x = torch.randint(0, 5000, (batch_size, seq_len))
        
        # Forward pass
        output = model(x)
        
        # Get attention weights
        attention_weights = model.get_attention_weights()
        
        if len(attention_weights) == 3:  # 3 layers
            print(f"✓ Attention weights extracted successfully")
            print(f"  Number of layers: {len(attention_weights)}")
            print(f"  Attention shape per layer: {attention_weights[0].shape}")
            return True
        else:
            print(f"✗ Unexpected number of attention layers: {len(attention_weights)}")
            return False
    except Exception as e:
        print(f"✗ Attention test failed: {e}")
        return False

def test_tokenizer():
    """Test tokenizer functionality"""
    print("\nTesting tokenizer...")
    try:
        from train_email_classifier import Tokenizer
        
        tokenizer = Tokenizer(vocab_size=100, max_len=20)
        
        # Test text
        texts = [
            "Meeting scheduled for tomorrow",
            "SALE: 50% off now!",
            "Hey how are you doing"
        ]
        
        # Build vocab
        tokenizer.build_vocab(texts)
        
        # Test encoding
        test_text = "Meeting tomorrow"
        encoded = tokenizer.encode(test_text)
        
        if len(encoded) == 20:  # max_len
            print(f"✓ Tokenizer working correctly")
            print(f"  Vocab size: {len(tokenizer.word2idx)}")
            print(f"  Sample encoding: {encoded[:5]}...")
            return True
        else:
            print(f"✗ Unexpected encoding length: {len(encoded)}")
            return False
    except Exception as e:
        print(f"✗ Tokenizer test failed: {e}")
        return False

def test_dataset_loading():
    """Test dataset can be loaded"""
    print("\nTesting dataset loading...")
    try:
        import pandas as pd
        import os
        
        # Check if sample dataset exists
        if os.path.exists('sample_email_dataset.csv'):
            df = pd.read_csv('sample_email_dataset.csv')
            print(f"✓ Sample dataset loaded")
            print(f"  Samples: {len(df)}")
            print(f"  Categories: {df['label'].nunique()}")
            print(f"  Distribution:\n{df['label'].value_counts()}")
            return True
        else:
            print("⚠ sample_email_dataset.csv not found")
            print("  Run generate_email_dataset.py to create full dataset")
            return True  # Not a failure, just warning
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        return False

def test_gradient_flow(model):
    """Test that gradients flow through the model"""
    print("\nTesting gradient flow...")
    try:
        model.train()
        
        # Create dummy data
        x = torch.randint(0, 5000, (2, 20))
        target = torch.tensor([0, 1])
        
        # Forward pass
        output = model(x)
        
        # Compute loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Check if gradients exist
        has_grads = all(p.grad is not None for p in model.parameters() if p.requires_grad)
        
        if has_grads:
            print(f"✓ Gradients flowing correctly")
            print(f"  Loss value: {loss.item():.4f}")
            return True
        else:
            print(f"✗ Some parameters don't have gradients")
            return False
    except Exception as e:
        print(f"✗ Gradient test failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("="*70)
    print("EMAIL CLASSIFIER MODEL - ARCHITECTURE TEST")
    print("="*70)
    
    tests_passed = 0
    tests_total = 7
    
    # Test 1: Imports
    if test_imports():
        tests_passed += 1
    
    # Test 2: Model creation
    success, model = test_model_creation()
    if success:
        tests_passed += 1
    else:
        print("\n✗ Cannot continue without model")
        return
    
    # Test 3: Forward pass
    if test_forward_pass(model):
        tests_passed += 1
    
    # Test 4: Attention weights
    if test_attention_weights(model):
        tests_passed += 1
    
    # Test 5: Tokenizer
    if test_tokenizer():
        tests_passed += 1
    
    # Test 6: Dataset
    if test_dataset_loading():
        tests_passed += 1
    
    # Test 7: Gradients
    if test_gradient_flow(model):
        tests_passed += 1
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests passed: {tests_passed}/{tests_total}")
    
    if tests_passed == tests_total:
        print("\n✓ All tests passed! System is ready for training.")
        print("\nNext steps:")
        print("  1. Generate full dataset: python3 generate_email_dataset.py")
        print("  2. Train model: python3 train_email_classifier.py")
        print("  3. Visualize attention: python3 visualize_attention.py")
    else:
        print(f"\n⚠ {tests_total - tests_passed} test(s) failed")
        print("  Please check the errors above and install missing dependencies")
    
    print("="*70)

if __name__ == "__main__":
    try:
        run_all_tests()
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

