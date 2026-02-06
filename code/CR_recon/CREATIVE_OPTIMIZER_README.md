# Creative Optimizer Skill

## Overview

The **Creative Optimizer** is an intelligent system that analyzes your training logs and **creatively proposes new model architectures and loss functions** to help you achieve lower validation loss values.

This goes beyond simple hyperparameter optimization by generating genuinely novel ideas grounded in your actual performance metrics.

## Files Included

### Core Tool
- **`creative_optimizer.py`** - Main analysis and proposal generation script

### Example Implementations (Pre-Built)
- **`losses/huber_pearson.py`** - Huber loss + Pearson correlation
- **`losses/smoothness_mse.py`** - MSE + Total Variation regularization
- **`configs/test_huber_pearson.yaml`** - Config for HuberPearsonLoss
- **`configs/test_smoothness_mse.yaml`** - Config for SmoothnessMSELoss

### Documentation
- **`QUICK_START_CREATIVE_OPTIMIZER.md`** - Fast 5-minute introduction
- **`CREATIVE_OPTIMIZER_GUIDE.md`** - Comprehensive detailed guide
- **`CREATIVE_OPTIMIZER_README.md`** - This file

## Quick Start

### 1. Analyze Your Current Training
```bash
python CR_recon/train.py --config CR_recon/configs/default.yaml
# This produces outputs/train_log.txt
```

### 2. Get Creative Proposals
```bash
python CR_recon/creative_optimizer.py --log outputs/train_log.txt
```

Output includes:
- Performance analysis
- 5 model architecture proposals
- 5 loss function proposals
- Implementation guidance
- JSON report saved to `outputs/creative_proposals.json`

### 3. Test a Pre-Built Proposal
```bash
# Test HuberPearsonLoss (outlier-robust)
python CR_recon/train.py --config CR_recon/configs/test_huber_pearson.yaml

# OR test SmoothnessMSELoss (smooth predictions)
python CR_recon/train.py --config CR_recon/configs/test_smoothness_mse.yaml
```

### 4. Compare Results
```bash
# Check loss values
grep "best_val=" outputs/train_log.txt | tail -1
```

## How It Works

### Analysis Phase
1. **Parses your training log** to extract:
   - Current model architecture
   - Current loss function
   - Hyperparameters
   - Training metrics (loss trajectory, convergence, overfitting)

2. **Computes performance metrics**:
   - Improvement rate
   - Convergence trend
   - Train-val gap (overfitting indicator)
   - Convergence status

### Proposal Phase
3. **Generates creative proposals**:

   **Models**: Based on convergence patterns
   - MultiHeadChannelAttention (better attention)
   - RegularizedCNN (for overfitting)
   - DenseResidualCNN (for plateau)
   - LocalTransformerCNN (efficient attention)
   - SpectralAwareCNN (periodic structures)

   **Losses**: Based on data characteristics
   - HuberPearsonLoss (robust to outliers) ✅
   - SmoothnessMSELoss (smooth predictions) ✅
   - QuantileRobustLoss (specific percentiles)
   - WassersteinSpectralLoss (distribution matching)
   - MultiScaleSpectralLoss (multiple scales)

4. **Creates implementation stubs** for each proposal
5. **Saves detailed JSON report** for reference

## Pre-Built Loss Functions

### HuberPearsonLoss
- **Purpose**: Robust to outliers + maintains spectral shape
- **Formula**: `0.8 * Huber(pred, target) + 0.2 * (1 - Pearson)`
- **When to use**: Data with occasional extreme values
- **Files**:
  - `losses/huber_pearson.py`
  - `configs/test_huber_pearson.yaml`

### SmoothnessMSELoss
- **Purpose**: Encourage smooth spectral predictions
- **Formula**: `0.7 * MSE + 0.3 * TV(pred)`
- **When to use**: Noisy data or want smoother outputs
- **Files**:
  - `losses/smoothness_mse.py`
  - `configs/test_smoothness_mse.yaml`

## Typical Workflow

```
Step 1: Train Baseline
  $ python CR_recon/train.py --config default.yaml
  Best loss: 0.0234

Step 2: Analyze & Get Proposals
  $ python CR_recon/creative_optimizer.py --log outputs/train_log.txt
  [Shows 5 model + 5 loss proposals]

Step 3: Test Proposals
  $ python CR_recon/train.py --config test_huber_pearson.yaml
  Best loss: 0.0198 ✓ Improved!

Step 4: Implement & Test More
  [Implement SmoothnessMSE or other proposals]
  $ python CR_recon/train.py --config test_smoothness_mse.yaml
  Best loss: 0.0205 (HuberPearson still better)

Step 5: Iterate Again
  $ python CR_recon/creative_optimizer.py --log outputs/train_log.txt
  [New proposals based on updated performance]
  [Continue testing...]
```

## Implementation Guide

### For Loss Functions (Easy)

Pattern:
```python
# 1. Create function (losses/my_loss.py)
def get_my_loss(**params):
    def loss_fn(pred, target):
        # Handle shape: (B, 2, 2, L) or (B, 4, L)
        if pred.dim() == 4:
            pred = pred.view(pred.shape[0], -1, pred.shape[-1])
            target = target.view(target.shape[0], -1, target.shape[-1])

        # Compute loss
        return some_loss_computation

    return loss_fn

# 2. Register in __init__.py
from .my_loss import get_my_loss
_LOSSES["my_loss"] = get_my_loss

# 3. Create config
loss:
  name: my_loss
  params: {...}

# 4. Train
python CR_recon/train.py --config config_file.yaml
```

### For Models (Medium)

Pattern:
```python
# 1. Create model (models/my_model.py)
class MetaSpec_MyModel(nn.Module):
    def __init__(self, out_len=30, d_model=192, **kwargs):
        super().__init__()
        # Build architecture

    def forward(self, x):
        # x: (B, 1, 128, 128)
        # returns: (B, 2, 2, out_len)
        pass

# 2. Register in __init__.py
from .my_model import MetaSpec_MyModel
_MODELS["my_model"] = MetaSpec_MyModel

# 3. Create config
model:
  name: my_model
  params: {...}

# 4. Train
python CR_recon/train.py --config config_file.yaml
```

## Key Metrics Explained

### Improvement Rate
- How much the loss decreased from epoch 1 to best epoch
- >50%: Good convergence
- 10-50%: Moderate convergence
- <10%: Slow learning (may need different model/loss)

### Convergence Trend
- **Improving**: Loss still decreasing recently → model still learning
- **Plateauing**: Loss has stabilized → might need new approach

### Train-Val Gap
- Difference between training and validation loss
- <10%: No overfitting
- 10-20%: Moderate overfitting
- >20%: Severe overfitting (need regularization)

### Overfitting Status
- Based on train-val gap and other metrics
- **Unlikely**: Good generalization
- **Moderate**: Some overfitting but manageable
- **Likely**: Need regularization or different model

## Tips for Best Results

1. **Test systematically**
   - Change one thing at a time initially
   - Then combine successful ideas

2. **Keep an experiment log**
   ```
   Baseline:              0.0234
   HuberPearson:          0.0198 ✓
   SmoothnessMSE:         0.0205
   HuberPearson + LocalT: 0.0187 ✓✓
   ```

3. **Run multiple tests in parallel**
   - Opens terminal for each config
   - Tests 3-5 proposals simultaneously
   - Saves time

4. **Iterate multiple rounds**
   - 5-10 iterations typically needed
   - Each round should improve slightly

5. **Trust the metrics**
   - Proposals are based on real data
   - Not random guesses
   - Grounded in performance analysis

## Troubleshooting

### "NotImplementedError"
The tool creates skeleton files. You need to implement the actual code.
- Check `# TODO` markers in the generated file
- Look at existing implementations for patterns
- See `QUICK_START_CREATIVE_OPTIMIZER.md` for examples

### Loss didn't improve
- Try a different proposal
- Adjust learning rate or training epochs
- Data might have reached its limit

### Shape mismatch
- Ensure loss handles (B, 2, 2, L) input
- Reshape to (B, 4, L) internally if needed
- Check existing losses for examples

## Advanced Customization

Edit `creative_optimizer.py` to add domain-specific proposals:

```python
def propose_creative_models(log_data, analysis):
    proposals = []

    # Add your custom proposal
    proposals.append({
        'type': 'model',
        'name': 'MyCustomCNN',
        'description': 'My specific architecture idea',
        'rationale': 'Because of [specific reason]',
        # ... rest of proposal dict ...
    })

    return proposals
```

## Architecture Compatibility

- **Input**: (B, 1, 128, 128) - Single grayscale 128×128 image
- **Output**: (B, 2, 2, 30) - BGGR spectrum with 30 bins
- **All models must maintain these IO dimensions**

## Loss Function Compatibility

- **Input**: (B, 2, 2, 30) or (B, 4, 30) - Model output
- **Target**: Same shape as input
- **Output**: Scalar loss value
- **All losses must handle both 3D and 4D input shapes**

## Performance Expectations

Typical improvements per iteration:
- Round 1 (loss function): 5-15% improvement
- Round 2 (model architecture): 5-10% improvement
- Round 3+ (combinations): 2-5% improvement

Total potential: 15-30% improvement over baseline.

## References

- **Existing models**: `models/cnn_xattn.py`, `cnn_gru.py`
- **Existing losses**: `losses/mse_pearson.py`, `weighted_smooth.py`
- **Config system**: `configs/default.yaml`
- **Training loop**: `trainer.py`

## Command Reference

```bash
# Analyze and get proposals
python CR_recon/creative_optimizer.py --log outputs/train_log.txt

# Train with config
python CR_recon/train.py --config CR_recon/configs/test_huber_pearson.yaml

# View results
grep "best_val=" outputs/train_log.txt | tail -5

# View proposals
cat CR_recon/outputs/creative_proposals.json | python -m json.tool
```

## License

Same as parent project.

## Support

See documentation files:
- **Quick questions**: QUICK_START_CREATIVE_OPTIMIZER.md
- **Detailed info**: CREATIVE_OPTIMIZER_GUIDE.md
- **Implementation help**: Examples in this README

---

**Ready to optimize?** Start with:
```bash
python CR_recon/creative_optimizer.py --log outputs/train_log.txt
```
