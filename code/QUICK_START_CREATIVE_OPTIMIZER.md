# Creative Optimizer - Quick Start Guide

## 🚀 What is the Creative Optimizer?

A smart tool that analyzes your training logs and proposes **new model architectures** and **novel loss functions** to help you achieve lower validation loss values.

Unlike basic hyperparameter optimization, Creative Optimizer suggests genuinely new ideas based on your current performance.

## ⚡ Quick Usage (5 minutes)

### Step 1: Train Your Baseline Model
```bash
# Train with your current configuration
python CR_recon/train.py --config CR_recon/configs/default.yaml

# Check the best loss value
grep "best_val=" outputs/train_log.txt | tail -1
# Output: best_val_loss = 0.0234 (example)
```

### Step 2: Run Creative Optimizer
```bash
python CR_recon/creative_optimizer.py --log outputs/train_log.txt
```

This will:
- 📊 Analyze your current performance
- 💡 Generate 5 new model ideas
- 💔 Generate 5 new loss function ideas
- 📄 Create implementation stubs
- 📁 Save detailed proposals to `outputs/creative_proposals.json`

### Step 3: Test a Proposal
```bash
# Try the new HuberPearsonLoss (already implemented)
python CR_recon/train.py --config CR_recon/configs/test_huber_pearson.yaml

# Check new loss value
grep "best_val=" outputs/train_log.txt | tail -1
# Example: best_val_loss = 0.0198 ✓ Better!
```

### Step 4: Compare Results
```bash
# Keep track of your experiments
Original:         0.0234
HuberPearson:     0.0198  ← Better!
SmoothnessMSE:    0.0205
```

### Step 5: Iterate
```bash
# Run optimizer again with updated log
python CR_recon/creative_optimizer.py --log outputs/train_log.txt

# Will propose new ideas based on current performance
```

## 📦 What's Already Implemented?

Ready-to-use loss functions:

1. **HuberPearsonLoss** ✅
   - More robust to outliers than MSE
   - File: `CR_recon/losses/huber_pearson.py`
   - Config: `test_huber_pearson.yaml`

2. **SmoothnessMSELoss** ✅
   - Encourages smooth spectral predictions
   - File: `CR_recon/losses/smoothness_mse.py`
   - Config: `test_smoothness_mse.yaml`

## 🛠️ How to Implement New Proposals

### For Loss Functions (Easy - 10 minutes)

Example: Implement QuantileRobustLoss

1. Create `CR_recon/losses/quantile_robust.py`:
```python
def get_quantile_robust_loss(lower_quantile=0.5, upper_quantile=0.9, weight_ratio=2.0):
    def loss_fn(pred, tgt):
        # Reshape if needed
        if pred.dim() == 4:
            pred = pred.view(pred.shape[0], -1, pred.shape[-1])
            tgt = tgt.view(tgt.shape[0], -1, tgt.shape[-1])

        # Quantile loss implementation
        lower_loss = torch.nn.functional.smooth_l1_loss(
            torch.clamp(pred, min=torch.quantile(pred, lower_quantile)),
            tgt
        )
        upper_loss = torch.nn.functional.smooth_l1_loss(
            torch.clamp(pred, min=torch.quantile(pred, upper_quantile)),
            tgt
        )

        return lower_loss + weight_ratio * upper_loss

    return loss_fn
```

2. Update `CR_recon/losses/__init__.py`:
```python
from .quantile_robust import get_quantile_robust_loss

_LOSSES = {
    # ... existing losses ...
    "quantile_robust": get_quantile_robust_loss,
}
```

3. Create config `CR_recon/configs/test_quantile_robust.yaml`:
```yaml
loss:
  name: quantile_robust
  params:
    lower_quantile: 0.5
    upper_quantile: 0.9
    weight_ratio: 2.0
# ... rest of config ...
```

4. Train:
```bash
python CR_recon/train.py --config CR_recon/configs/test_quantile_robust.yaml
```

### For Models (Medium - 30 minutes)

Example: Implement MultiHeadChannelAttention

1. Create `CR_recon/models/multihead_attention_cnn.py`:
```python
import torch
import torch.nn as nn

class MultiHeadChannelAttention(nn.Module):
    def __init__(self, channels, heads=4):
        super().__init__()
        self.heads = heads
        self.head_dim = channels // heads

        self.fc1 = nn.Linear(channels, channels // 16)
        self.fc2 = nn.Linear(channels // 16, channels)

    def forward(self, x):
        B, C, H, W = x.shape

        # Global average pooling
        avg = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)  # (B, C)

        # Multi-head processing
        out = self.fc2(F.relu(self.fc1(avg)))  # (B, C)

        # Reshape for multi-head
        out = out.view(B, self.heads, self.head_dim)
        out = out.softmax(dim=-1)  # Attention weights
        out = out.view(B, C)

        return x * out.unsqueeze(-1).unsqueeze(-1)

class MetaSpec_MultiheadAttentionCNN(nn.Module):
    def __init__(self, out_len=30, d_model=192, heads=4, **kwargs):
        super().__init__()
        self.out_len = out_len
        self.d_model = d_model

        # Build architecture (similar to cnn_xattn but with MultiHeadChannelAttention)
        # ... [implement stages with MultiHeadChannelAttention instead of single-head]

    def forward(self, x):
        # ... [implement forward pass]
```

2. Update `CR_recon/models/__init__.py`:
```python
from .multihead_attention_cnn import MetaSpec_MultiheadAttentionCNN

_MODELS = {
    # ... existing models ...
    "multihead_attention_cnn": MetaSpec_MultiheadAttentionCNN,
}
```

3. Create config and train as usual

## 📊 Understanding the Proposals

### When to Try What?

| Problem | Recommendation |
|---------|-----------------|
| Loss not decreasing enough | Try HuberPearsonLoss or SmoothnessMSELoss |
| Overfitting (high train-val gap) | Try RegularizedCNN |
| Loss plateauing | Try DenseResidualCNN or LocalTransformerCNN |
| Noisy predictions | Try SmoothnessMSELoss or QuantileRobustLoss |
| Slow convergence | Try LocalTransformerCNN |
| Periodic structure in data | Try SpectralAwareCNN |

### Reading the Analysis Output

```
📈 Performance Metrics:
  Improvement Rate: 45.23%        ← Good if > 30%
  Convergence Trend: improving    ← Good sign
  Train-Val Gap: 8.50%            ← Good if < 10%
  Overfitting Status: unlikely    ← Good sign
```

## 🔄 Comparison Workflow

### Test Multiple Proposals Simultaneously

```bash
# Terminal 1
python CR_recon/train.py --config CR_recon/configs/test_huber_pearson.yaml > log1.txt 2>&1

# Terminal 2
python CR_recon/train.py --config CR_recon/configs/test_smoothness_mse.yaml > log2.txt 2>&1

# Terminal 3
python CR_recon/train.py --config CR_recon/configs/test_quantile_robust.yaml > log3.txt 2>&1

# Compare results
echo "HuberPearson:"; grep "best_val=" outputs/train_log.txt | tail -1
echo "SmoothnessMSE:"; tail -1 log2.txt | grep "best_val="
# etc.
```

## 📈 Expected Workflow

```
Week 1:
  Day 1: Baseline training (best_val = 0.0234)
  Day 2: Run Creative Optimizer
  Day 3: Implement 2-3 proposals
  Day 4: Test and pick winner (best_val = 0.0198)
  Day 5: Fine-tune hyperparameters (best_val = 0.0185)

Week 2:
  Day 1: Run Creative Optimizer again
  Day 2-3: Test new proposals
  Day 4-5: Iterate further...
```

## 🎯 Tips for Success

1. **Test one change at a time initially**
   - Change loss → see if it improves
   - If yes, keep it; then change model
   - Then try combinations

2. **Save your experiments**
   ```bash
   # Create experiment log
   echo "Baseline CNN_XAttn + MSE_Pearson: 0.0234" > experiments.txt
   echo "HuberPearson loss: 0.0198" >> experiments.txt
   # Add more as you go...
   ```

3. **Trust the proposals**
   - They're based on your actual performance metrics
   - Not random guesses
   - Grounded in ML theory

4. **Iterate quickly**
   - Each test should take 30min-1hour
   - Run 5-10 iterations
   - Pick the best configuration

5. **Combine improvements**
   ```yaml
   # test_config_combined_best.yaml
   model:
     name: local_transformer_cnn  # Best model found
   loss:
     name: huber_pearson          # Best loss found
   ```

## 🆘 Common Issues

### "NotImplementedError in forward"
- The stub file was generated, you need to fill it in
- Check the `# TODO` comments for guidance
- Or see examples in `cnn_xattn.py` for reference

### "Loss value didn't improve"
- Try a different proposal
- Adjust hyperparameters (learning rate, dropout)
- The loss might be near its minimum for your data

### "Shape mismatch error"
- Ensure your implementation handles (B, 2, 2, L) input shape
- Reshape to (B, 4, L) if needed in loss functions
- Check existing losses for pattern

## 📚 Learn More

See `CREATIVE_OPTIMIZER_GUIDE.md` for:
- Detailed explanation of each proposal
- Architecture sketches
- Implementation examples
- Advanced customization

## 🎓 Example Session

```bash
# 1. Initial training
python CR_recon/train.py --config CR_recon/configs/default.yaml
# Result: best_val = 0.0234

# 2. Analyze
python CR_recon/creative_optimizer.py --log outputs/train_log.txt
# Suggests: HuberPearson, Smoothness, etc.

# 3. Test HuberPearson (pre-implemented)
python CR_recon/train.py --config CR_recon/configs/test_huber_pearson.yaml
# Result: best_val = 0.0198 ✓

# 4. Keep HuberPearson, test Smoothness
python CR_recon/train.py --config CR_recon/configs/test_smoothness_mse.yaml
# Result: best_val = 0.0205 (HuberPearson is better)

# 5. Combine best so far with different model
# (Implement LocalTransformerCNN if interested)
# python CR_recon/train.py --config test_local_transformer_huber.yaml
# Result: best_val = 0.0187 ✓✓

# Keep iterating...
```

## ✅ Checklist

- [ ] Run `creative_optimizer.py` at least once
- [ ] Review the proposals in `outputs/creative_proposals.json`
- [ ] Test HuberPearsonLoss (already implemented)
- [ ] Test SmoothnessMSELoss (already implemented)
- [ ] Pick the best performer
- [ ] Run optimizer again with new baseline
- [ ] Implement and test at least one new proposal
- [ ] Achieve lower loss than baseline

Happy optimizing! 🚀
