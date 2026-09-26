# Attention Visualization Fix - Summary

## Problem

The `visualize_attention.py` script was failing with an `AttributeError`:

```
AttributeError: Can't get attribute 'Tokenizer' on <module '__main__' from '/Users/manimaran/Desktop/Credo Systems AI Course/section3 2/visualize_attention.py'>
```

## Root Cause

When the model checkpoint (`best_model.pth`) was saved during training, it included a pickled `Tokenizer` object:

```python
# In train_email_classifier.py (line 408-416)
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'val_acc': val_acc,
    'tokenizer': tokenizer,  # ← Tokenizer object saved here
    'label2idx': label2idx,
    'idx2label': idx2label,
}, 'best_model.pth')
```

When loading this checkpoint in `visualize_attention.py`, PyTorch's unpickler tried to reconstruct the `Tokenizer` object, but it couldn't find the class definition because:

1. The `Tokenizer` class was defined in `train_email_classifier.py`
2. The `visualize_attention.py` script didn't import this class
3. Python's pickle module requires the class definition to be available when unpickling

## Solution

Added the missing import statement to `visualize_attention.py`:

```python
from train_email_classifier import Tokenizer  # Import Tokenizer class for unpickling
```

This allows Python to find the `Tokenizer` class definition when unpickling the checkpoint.

## Verification

After the fix, the script runs successfully and generates attention visualizations:

```
✓ Loading trained model...
✓ Model loaded successfully!

Email 1: Meeting scheduled for tomorrow at 3 PM...
  Predicted: work (confidence: 0.998)
  ✓ email_1_layer1_head1.png
  ✓ email_1_all_heads_layer1.png
  ✓ email_1_patterns.png

Email 2: SALE: Up to 50% off on all items...
  Predicted: promotions (confidence: 0.997)
  ✓ email_2_layer1_head1.png
  ✓ email_2_all_heads_layer1.png
  ✓ email_2_patterns.png

Email 3: You've won $1,000,000!...
  Predicted: spam (confidence: 0.998)
  ✓ email_3_layer1_head1.png
  ✓ email_3_all_heads_layer1.png
  ✓ email_3_patterns.png
```

## Generated Visualizations

The script successfully created 9 attention visualization images:

### For Each Email:
1. **Single Head Heatmap** (`email_X_layer1_head1.png`)
   - Shows attention weights for one specific attention head
   - Visualizes which tokens attend to which other tokens

2. **All Heads** (`email_X_all_heads_layer1.png`)
   - Shows all 8 attention heads for layer 1
   - Allows comparison of different attention patterns

3. **Patterns Across Layers** (`email_X_patterns.png`)
   - Shows average attention across all 3 transformer layers
   - Reveals how attention evolves through the network

## Key Takeaway

When saving PyTorch checkpoints that include custom objects (like the `Tokenizer` class), ensure that:

1. The class definition is importable when loading the checkpoint
2. Either import the class or define it in the loading script
3. Consider using `weights_only=True` if you only need model weights (though this wasn't an option here since we needed the tokenizer)

## Files Modified

- **visualize_attention.py**: Added import for `Tokenizer` class (line 9)

## Status

✅ **FIXED** - The attention visualization script now works correctly and generates all expected visualizations.
