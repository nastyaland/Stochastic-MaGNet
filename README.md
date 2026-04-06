# Stochastic MaGNet

Optimization study of the MaGNet algorithm from https://github.com/PeilinTime/MaGNet with stochastic extensions based on MC Dropout.

## Model Variants

| Version | MC Dropout placement |
|---------|----------------------|
| Magnetv1 | After MAGE + before output |
| Magnetv2 | After F2DAttn + before output |
| Magnetv3 | After MAGE, after F2DAttn, and before output |

## Setup

Run all commands from the `Stochastic-MaGNet` folder:

```powershell
cd .\Stochastic-MaGNet
```

Install the minimum packages needed for training and backtesting:

```powershell
pip install torch transformers torcheval tqdm einops
```

Optional plotting support:

```powershell
pip install matplotlib
```

## Data

Generate the NASDAQ-100 tensor with:

- `data_collection/Nasdaq100_data_collection_and_cleaning.ipynb`

Expected output file:

- `data_collection/my_nas100_2025_data.pt`

## Training

1. Open `train.py` and choose the model version:

```python
MODEL_VERSION = 'Magnetv1'  # or 'Magnetv2' / 'Magnetv3'
```

2. Run training:

```powershell
python train.py
```

3. Outputs are saved with the chosen version suffix:

- `best_model_Magnetv1.pth`
- `final_model_Magnetv1.pth`
- `training_history_Magnetv1.pt`
- `training_history_Magnetv1.png`

## Optional MC Inference

If you want standalone MC Dropout inference outside backtesting:

```powershell
python inference_MC.py
```

This produces:

- `mc_results_Magnetv1.pt`

with:

- `mean_pred`
- `var_pred`

## Backtesting

Two scripts are provided:

- `backtest_baseline.py` for deterministic backtesting
- `backtest_bayesian.py` for MC Dropout uncertainty-aware backtesting

Both scripts expect:

- `data_collection\my_nas100_2025_data.pt`
- a trained checkpoint such as `trainHistory\trainMagnetv1\best_model_Magnetv1.pth`

### Baseline Backtest

Deterministic baseline with fixed parameters:

```powershell
python backtest_baseline.py --data-name NASDAQ100 --data-path data_collection\my_nas100_2025_data.pt --weight-path trainHistory\trainMagnetv1\best_model_Magnetv1.pth --model-version Magnetv1 --fixed-p-ratio 1 --fixed-q-stop-loss 0.2 --fixed-r-rising-ratio 1
```

Deterministic baseline with validation grid search:

```powershell
python backtest_baseline.py --data-name NASDAQ100 --data-path data_collection\my_nas100_2025_data.pt --weight-path trainHistory\trainMagnetv1\best_model_Magnetv1.pth --model-version Magnetv1 --p-grid 0.2 0.4 0.6 0.8 1.0 --q-grid 0.05 0.2 0.4 --r-grid 0 0.5 1
```

### Bayesian Backtest

Use deterministic validation for `p / q / r` tuning and MC Dropout only on test:

```powershell
python backtest_bayesian.py --data-name NASDAQ100 --data-path data_collection\my_nas100_2025_data.pt --weight-path trainHistory\trainMagnetv1\best_model_Magnetv1.pth --model-version Magnetv1 --deterministic-val --num-mc-runs 20 --p-grid 0.2 0.4 0.6 0.8 1.0 --q-grid 0.05 0.2 0.4 --r-grid 0 0.5 1
```

Use fixed baseline parameters plus Bayesian risk control:

```powershell
python backtest_bayesian.py --data-name NASDAQ100 --data-path data_collection\my_nas100_2025_data.pt --weight-path trainHistory\trainMagnetv1\best_model_Magnetv1.pth --model-version Magnetv1 --deterministic-val --num-mc-runs 20 --fixed-p-ratio 1 --fixed-q-stop-loss 0.2 --fixed-r-rising-ratio 1 --prob-threshold 0.5 --variance-quantile 0.8 --variance-weight 10 --market-variance-threshold 0.003
```

Bayesian risk-control arguments:

- `--prob-threshold`: minimum mean rise probability for a stock to enter the candidate pool
- `--variance-quantile`: filters out the highest-variance candidate stocks each day
- `--variance-weight`: penalty strength in the risk-adjusted ranking score
- `--market-variance-threshold`: if average daily market uncertainty is above this threshold, skip trading for that day
- `--num-mc-runs`: number of stochastic forward passes for MC Dropout
- `--deterministic-val`: tune `p / q / r` on deterministic validation predictions, then apply Bayesian inference only on test

## Output Files

Backtest outputs are written to `backtest_outputs/`:

- `backtest_Magnetv1_baseline.png`
- `backtest_Magnetv1_baseline.json`
- `backtest_Magnetv1_bayesian.png`
- `backtest_Magnetv1_bayesian.json`
- `variance_summary_Magnetv1_bayesian.pt`

## Notes

- `backtest_baseline.py` uses a single deterministic forward pass.
- `backtest_bayesian.py` is slower because it runs multiple stochastic forward passes.
- On CPU, Bayesian backtesting can take a long time. For faster experiments, reduce `--num-mc-runs` from `100` to `20`.
- Device selection is automatic: CUDA first, then MPS, then CPU.
