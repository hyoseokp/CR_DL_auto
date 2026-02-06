# Creative Optimizer Skill Guide

## Overview

The **Creative Optimizer** is an intelligent analysis tool that reads your training logs and proposes creative new model architectures and loss functions to achieve lower validation loss values.

Unlike the basic `optimize_hyperparams.py` which only suggests switching between existing models/losses and adjusting hyperparameters, the Creative Optimizer generates **novel architectural ideas** based on:

- Current performance metrics
- Convergence patterns
- Overfitting analysis
- Task characteristics (spectral prediction, periodic structures)

## Usage

```bash
python CR_recon/creative_optimizer.py --log outputs/train_log.txt
```

### Output

The tool produces:

1. **Console Output**
   - Current performance analysis
   - 5 creative model architecture proposals
   - 5 creative loss function proposals
   - Next steps guide

2. **Implementation Stubs**
   - Skeleton files created in `CR_recon/models/` and `CR_recon/losses/`
   - Ready for implementation (includes TODO markers)

3. **JSON Report**
   - Saved to `CR_recon/outputs/creative_proposals.json`
   - Contains full analysis and proposal details

## Proposal Categories

### 🎨 Creative Model Proposals

1. **MultiHeadChannelAttention**
   - Multi-head channel attention mechanism
   - Better utilization of Transformer-style parallel processing
   - Ideal for: CNN + XAttn users wanting more powerful attention

2. **RegularizedCNN**
   - Enhanced regularization through mixed normalization
   - Stronger dropout and attention regularization
   - Ideal for: Overfitting issues

3. **DenseResidualCNN**
   - Dense connections inspired by DenseNet
   - Improved gradient flow and feature reuse
   - Ideal for: Plateau in convergence

4. **LocalTransformerCNN**
   - Local window attention (8x8 windows)
   - Efficient alternative to full self-attention
   - Ideal for: Speed + attention benefits

5. **SpectralAwareCNN**
   - Fourier domain preprocessing
   - Periodic structure-aware learning
   - Ideal for: Taking advantage of 128×128 periodicity

### 💔 Creative Loss Function Proposals

1. **HuberPearsonLoss**
   - Huber loss (outlier-robust) + Pearson correlation
   - Better handling of outliers
   - Formula: `0.8 * Huber(pred, target) + 0.2 * (1 - Pearson)`

2. **SmoothnessMSELoss**
   - MSE + Total Variation regularization
   - Forces smooth spectral predictions
   - Formula: `0.7 * MSE + 0.3 * TV(pred)`

3. **QuantileRobustLoss**
   - Different weights for different percentiles
   - Emphasizes high-intensity accuracy
   - Formula: Weighted quantile loss at 50% and 90%

4. **WassersteinSpectralLoss**
   - Distribution matching instead of pointwise comparison
   - More robust to distribution shifts
   - Formula: `W(pred_dist, target_dist) + 0.1 * Pearson`

5. **MultiScaleSpectralLoss**
   - Simultaneous learning at multiple scales
   - Balanced frequency domain performance
   - Formula: `MSE + 0.5*MSE(smooth) + 0.5*Pearson`

## Implementation Workflow

### Step 1: Run Creative Optimizer

```bash
python CR_recon/creative_optimizer.py --log outputs/train_log.txt
```

### Step 2: Review Proposals

Read the console output and `creative_proposals.json` to understand each proposal.

### Step 3: Implement Your Choice

Fill in the generated stub files with actual code:

```python
# Example: Implementing HuberPearsonLoss
class HuberPearsonLoss(nn.Module):
    def __init__(self, huber_delta=0.5, pearson_weight=0.2):
        super().__init__()
        self.huber_delta = huber_delta
        self.pearson_weight = pearson_weight

    def forward(self, pred, target):
        # Reshape if needed
        if pred.dim() == 4:  # (B, 2, 2, out_len)
            pred = pred.view(-1, 4, pred.shape[-1]).view(-1, pred.shape[-1])
            target = target.view(-1, 4, target.shape[-1]).view(-1, target.shape[-1])

        # Huber loss
        huber = F.smooth_l1_loss(pred, target, beta=self.huber_delta)

        # Pearson correlation
        pred_mean = pred.mean(dim=-1, keepdim=True)
        target_mean = target.mean(dim=-1, keepdim=True)

        pred_centered = pred - pred_mean
        target_centered = target - target_mean

        cov = (pred_centered * target_centered).mean(dim=-1)
        std_pred = pred_centered.std(dim=-1)
        std_target = target_centered.std(dim=-1)

        pearson = cov / (std_pred * std_target + 1e-8)
        pearson_loss = 1 - pearson.mean()

        return (1 - self.pearson_weight) * huber + self.pearson_weight * pearson_loss
```

### Step 4: Create Config File

Create a test config file (or edit `default.yaml`):

```yaml
# CR_recon/configs/test_config_huber_pearson.yaml
model:
  name: cnn_xattn  # Use any model
  params:
    out_len: 30
    d_model: 192

loss:
  name: huber_pearson  # Your new loss
  params:
    huber_delta: 0.5
    pearson_weight: 0.2

data:
  batch_size: 64

training:
  lr: 0.001
  epochs: 50
  # ... other hyperparams
```

### Step 5: Train and Compare

```bash
# Train with new loss function
python CR_recon/train.py --config CR_recon/configs/test_config_huber_pearson.yaml

# Check the new best_val_loss in outputs/train_log.txt
```

### Step 6: Analyze Results

Compare new `best_val_loss` with the previous one. If improved:

1. Keep the new model/loss in permanent files
2. Update `default.yaml`
3. Run Creative Optimizer again with updated log for further iterations

## Example: Complete Iterative Loop

```bash
# Initial training
python CR_recon/train.py --config CR_recon/configs/default.yaml
# Log shows: best_val_loss = 0.0234

# Round 1: Creative Analysis
python CR_recon/creative_optimizer.py --log outputs/train_log.txt
# Proposes: HuberPearsonLoss, LocalTransformerCNN, etc.

# Round 1: Implement and test loss
# (implement huber_pearson.py)
python CR_recon/train.py --config test_config_huber_pearson.yaml
# Result: best_val_loss = 0.0198 ✓ Better!

# Round 2: Keep the loss, try new model
# (implement local_transformer_cnn.py, use huber_pearson loss)
python CR_recon/train.py --config test_config_local_transformer.yaml
# Result: best_val_loss = 0.0187 ✓ Even better!

# Round 3: Analyze new config
python CR_recon/creative_optimizer.py --log outputs/train_log.txt
# New proposals for further improvement...
```

## Key Metrics Explained

### Improvement Rate
- Percentage improvement from first epoch to best epoch
- Higher is better (model is learning)
- >50%: Good convergence
- <10%: Slow learning (consider better hyperparams or different model)

### Convergence Trend
- **Improving**: Loss still decreasing in recent epochs
- **Plateauing**: Loss has stabilized
- Action: Improving → Keep training; Plateauing → Try new model/loss/learning rate

### Train-Val Gap
- Difference between training and validation loss as percentage of training loss
- 0-10%: No overfitting
- 10-20%: Moderate overfitting
- >20%: Severe overfitting (need regularization)

### Overfitting Status
- Based on train-val gap and other metrics
- **Unlikely**: Model generalizing well
- **Moderate**: Some overfitting, but manageable
- **Likely**: Need regularization (RegularizedCNN, more dropout, etc.)

## Tips for Success

### 1. Test Multiple Proposals Simultaneously

Don't just test one. Run 2-3 promising proposals in parallel:

```bash
# Terminal 1
python CR_recon/train.py --config test_config_huber_pearson.yaml

# Terminal 2
python CR_recon/train.py --config test_config_local_transformer.yaml

# Terminal 3
python CR_recon/train.py --config test_config_smoothness_mse.yaml
```

### 2. Combine Model + Loss Innovations

The best improvements often come from combining a new model with a new loss:

```yaml
# test_config_combined.yaml
model:
  name: local_transformer_cnn  # New model
loss:
  name: huber_pearson          # New loss
```

### 3. Track Your Experiments

Keep a log of experiments:

```
# experiments.txt
Baseline (CNN_XAttn + MSE_Pearson): 0.0234

Round 1:
- HuberPearson loss: 0.0198 ✓
- LocalTransformer model: 0.0215

Round 2 (Combined):
- LocalTransformer + HuberPearson: 0.0187 ✓

Round 3:
- SpectralAwareCNN + MultiScaleSpectral: 0.0195
```

### 4. Leverage Domain Knowledge

The proposals are generic suggestions. Customize them based on your understanding:

- **Periodic structure?** → SpectralAwareCNN
- **Noisy data?** → HuberPearsonLoss or WassersteinSpectralLoss
- **Slow convergence?** → DenseResidualCNN or LocalTransformerCNN
- **Overfitting?** → RegularizedCNN

### 5. Iterate Gradually

Start small, test one change at a time, then combine successful ideas:

```
Step 1: Find best loss function
Step 2: Keep best loss, find best model
Step 3: Fine-tune hyperparameters with winning combo
Step 4: If still not satisfied, run Creative Optimizer again
```

## Troubleshooting

### "NotImplementedError in model forward"

The stub file was generated. You need to implement the actual architecture:

1. Open the file (e.g., `CR_recon/models/local_transformer_cnn.py`)
2. Replace `# TODO` sections with actual code
3. Test import: `python -c "from CR_recon.models.local_transformer_cnn import LocalTransformerCNN"`

### "Loss function has shape mismatch"

Ensure your loss function handles both (B, 4, out_len) and (B, 2, 2, out_len) shapes:

```python
# Normalize to (B, 4, out_len)
if pred.dim() == 4:
    pred = pred.view(pred.shape[0], -1, pred.shape[-1])
    target = target.view(target.shape[0], -1, target.shape[-1])
```

### "Config file not found"

Config files should be in `CR_recon/configs/`:

```bash
ls -la CR_recon/configs/  # Check existing configs
cp CR_recon/configs/default.yaml CR_recon/configs/test_config_xxx.yaml
# Edit test_config_xxx.yaml
```

## Advanced: Custom Proposals

You can modify `creative_optimizer.py` to add domain-specific proposals:

```python
def propose_creative_models(log_data, analysis):
    proposals = []

    # Add custom proposal
    proposals.append({
        'type': 'model',
        'name': 'MyCustomModel',
        'description': 'My specific idea',
        'rationale': 'Because...',
        'architecture_sketch': {'component': 'description'},
        'expected_benefit': 'Better performance',
        'implementation_file': 'models/my_custom_model.py',
        'config_changes': {'model': {'name': 'my_custom_model', 'params': {}}}
    })

    return proposals
```

## References

- **Current models**: `CR_recon/models/cnn_xattn.py`, `cnn_gru.py`
- **Current losses**: `CR_recon/losses/mse_pearson.py`, `weighted_smooth.py`
- **Config system**: `CR_recon/configs/default.yaml`
- **Trainer**: `CR_recon/trainer.py`

## Quick Command Reference

```bash
# Analyze log and get proposals
python CR_recon/creative_optimizer.py --log outputs/train_log.txt

# Train with specific config
python CR_recon/train.py --config CR_recon/configs/test_config_xxx.yaml

# View latest training results
cat outputs/train_log.txt | tail -20

# Check proposal details
cat CR_recon/outputs/creative_proposals.json | python -m json.tool

# Compare loss values
grep "best_val=" outputs/train_log.txt | tail -1
```

Happy experimenting! 🚀
