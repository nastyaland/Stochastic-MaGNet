# Stochastic MaGNet

Optimization study of the MaGNet algorithm from https://github.com/PeilinTime/MaGNet with stochastic extensions. This study showcases how the MaGNet can be
optimized using Monte Carlo Dropout to improve stock predictions by taking into account the prediction mean and variance. Those
uncertainty metrics are then used during backtesting to enhance the stock trading strategy.


<img src="https://github.com/nastyaland/Stochastic-MaGNet/blob/f2e46da19bf937536f39a5ae1aeaa125cbb47f20/images/Difference_graph.png" width="70%">

## Model Variants
Three model variations were analyzed based on the location of the MC Dropout Layers.

<table>
<tr><th>Model Variations </th><th>Architecture Representation</th></tr>
<tr><td>

| Version | MC Dropout placement |
|---------|----------------------|
| Magnetv1 | After MAGE + before output |
| Magnetv2 | After F2DAttn + before output |
| Magnetv3 | After MAGE, after F2DAttn, and before output |

</td><td>

<img src="https://github.com/nastyaland/Stochastic-MaGNet/blob/f2e46da19bf937536f39a5ae1aeaa125cbb47f20/images/MaGNet%20Models.png" width="100%">

</td></tr> </table>

## Setup

Run all commands from the `Stochastic-MaGNet` folder:

```powershell
cd .\Stochastic-MaGNet
```

Install the minimum packages needed for training and backtesting:

```powershell
pip install torch transformers torcheval tqdm einops optuna
```

Optional plotting support:

```powershell
pip install matplotlib
```

## Data

Generate the NASDAQ-100 tensor with:

- `data_collection/Nasdaq100_data_collection_and_cleaning.ipynb`

Expected output file:

- `new_my_nas100_2025_data.pt`

## Hyperparameter Search 

1.  Open `train.py` and choose the model version:

```python
MODEL_VERSION = 'Magnetv1'  # or 'Magnetv2' / 'Magnetv3'
```
2. Go to main and set
```python
SEARCH = True
```
4. In SEARCH, adjust the hyperparameters to search, the num_trials, and search_epochs
5. Run:
```powershell
python train.py
```
5. Optuna runs a tree search and terminal outputs the best hyperparameters found
## Training

1. Open `train.py` and choose the model version:

```python
MODEL_VERSION = 'Magnetv1'  # or 'Magnetv2' / 'Magnetv3'
```

2. Set `SEARCH = False` and plug in the best hyperparameters from the search.

3. Run training:

```powershell
python train.py
```

3. Outputs are saved with the chosen version suffix:

- `best_model_Magnetv1.pth`
- `final_model_Magnetv1.pth`
- `training_history_Magnetv1.pt`
- `training_history_Magnetv1.png`

## Model Performance (NASDAQ-100)

Best hyperparameters and test-set results at peak validation accuracy:

| Model | T | num_TCH | TopK | M1 | M2 | lr | Val Acc | Test Acc | Test AUROC | Test F1 |
|-------|---|---------|------|----|----|-----|---------|----------|------------|--------|
| Magnetv1 | 20 | 2 | 128 | 128 | 64 | 3.98e-4 | 0.5452 | 0.5287 | 0.5006 | 0.6728 |
| Magnetv2 | 20 | 1 | 64 | 128 | 32 | 2.04e-4 | 0.5401 | 0.5183 | 0.5041 | 0.6703 |
| Magnetv3 | 20 | 2 | 64 | 32 | 32 | 2.97e-4 | 0.5666 | 0.5029 | 0.5074 | 0.4849 |

**Magnetv1** achieves the best test accuracy and F1, generalising most consistently from validation to test.

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

Useful split options:

- `--eval-split test`: evaluate validation and test separately
- `--eval-split val_test`: evaluate a combined out-of-sample period using validation + test

### Operation Guide

The three commands below are the current recommended NASDAQ100 runs with the best-performing fixed parameter set:

- `p_ratio = 0.5`
- `q_stop_loss = 0.4`
- `r_rising_ratio = 1`
- `rebalance_frequency = 5`

#### 1. Combined Deterministic Baseline

This is the strongest deterministic reference run on the combined out-of-sample period:

```powershell
python backtest_baseline.py --data-name NASDAQ100 --data-path data_collection\my_nas100_2025_data.pt --weight-path trainHistory\trainMagnetv1\best_model_Magnetv1.pth --model-version Magnetv1 --fixed-p-ratio 0.5 --fixed-q-stop-loss 0.4 --fixed-r-rising-ratio 1 --rebalance-frequency 5 --eval-split val_test
```

#### 2. Combined Bayesian Mean-Only

This run uses MC Dropout mean probabilities only, with no variance filtering:

```powershell
python backtest_bayesian.py --data-name NASDAQ100 --data-path data_collection\my_nas100_2025_data.pt --weight-path trainHistory\trainMagnetv1\best_model_Magnetv1.pth --model-version Magnetv1 --deterministic-val --num-mc-runs 20 --fixed-p-ratio 0.5 --fixed-q-stop-loss 0.4 --fixed-r-rising-ratio 1 --rebalance-frequency 5 --prob-threshold 0.5 --variance-quantile 1.0 --variance-weight 0 --eval-split val_test
```

#### 3. Combined Bayesian with Light Variance Selection

This run adds a light uncertainty filter on top of the MC mean signal:

```powershell
python backtest_bayesian.py --data-name NASDAQ100 --data-path data_collection\my_nas100_2025_data.pt --weight-path trainHistory\trainMagnetv1\best_model_Magnetv1.pth --model-version Magnetv1 --deterministic-val --num-mc-runs 20 --fixed-p-ratio 0.5 --fixed-q-stop-loss 0.4 --fixed-r-rising-ratio 1 --rebalance-frequency 5 --prob-threshold 0.5 --variance-quantile 0.95 --variance-weight 0.5 --eval-split val_test
```

### Baseline Backtest

Recommended deterministic baseline with lower turnover:

```powershell
python backtest_baseline.py --data-name NASDAQ100 --data-path data_collection\my_nas100_2025_data.pt --weight-path trainHistory\trainMagnetv1\best_model_Magnetv1.pth --model-version Magnetv1 --fixed-p-ratio 0.5 --fixed-q-stop-loss 0.4 --fixed-r-rising-ratio 1 --rebalance-frequency 5
```

Deterministic baseline with validation grid search:

```powershell
python backtest_baseline.py --data-name NASDAQ100 --data-path data_collection\my_nas100_2025_data.pt --weight-path trainHistory\trainMagnetv1\best_model_Magnetv1.pth --model-version Magnetv1 --rebalance-frequency 5 --p-grid 0.1 0.2 0.3 0.5 0.7 1.0 --q-grid 0.05 0.1 0.2 0.4 --r-grid 0.5 0.75 1
```

### Bayesian Backtest

Use deterministic validation for `p / q / r` tuning and MC Dropout only on test.

Lightweight Bayesian run (`20` MC passes, faster on CPU):

```powershell
python backtest_bayesian.py --data-name NASDAQ100 --data-path data_collection\my_nas100_2025_data.pt --weight-path trainHistory\trainMagnetv1\best_model_Magnetv1.pth --model-version Magnetv1 --deterministic-val --num-mc-runs 20 --rebalance-frequency 5 --p-grid 0.1 0.2 0.3 0.5 0.7 1.0 --q-grid 0.05 0.1 0.2 0.4 --r-grid 0.5 0.75 1
```

Standard Bayesian run (`100` MC passes):

```powershell
python backtest_bayesian.py --data-name NASDAQ100 --data-path data_collection\my_nas100_2025_data.pt --weight-path trainHistory\trainMagnetv1\best_model_Magnetv1.pth --model-version Magnetv1 --deterministic-val --num-mc-runs 100 --rebalance-frequency 5 --p-grid 0.1 0.2 0.3 0.5 0.7 1.0 --q-grid 0.05 0.1 0.2 0.4 --r-grid 0.5 0.75 1
```

Use fixed baseline parameters plus Bayesian risk control.

Lightweight Bayesian risk-control run (`20` MC passes):

```powershell
python backtest_bayesian.py --data-name NASDAQ100 --data-path data_collection\my_nas100_2025_data.pt --weight-path trainHistory\trainMagnetv1\best_model_Magnetv1.pth --model-version Magnetv1 --deterministic-val --num-mc-runs 20 --fixed-p-ratio 0.5 --fixed-q-stop-loss 0.4 --fixed-r-rising-ratio 1 --rebalance-frequency 5 --prob-threshold 0.5 --variance-quantile 0.8 --variance-weight 10 --market-variance-threshold 0.003
```

Standard Bayesian risk-control run (`100` MC passes):

```powershell
python backtest_bayesian.py --data-name NASDAQ100 --data-path data_collection\my_nas100_2025_data.pt --weight-path trainHistory\trainMagnetv1\best_model_Magnetv1.pth --model-version Magnetv1 --deterministic-val --num-mc-runs 100 --fixed-p-ratio 0.5 --fixed-q-stop-loss 0.4 --fixed-r-rising-ratio 1 --rebalance-frequency 5 --prob-threshold 0.5 --variance-quantile 0.8 --variance-weight 10 --market-variance-threshold 0.003
```

Bayesian risk-control arguments:

- `--prob-threshold`: minimum mean rise probability for a stock to enter the candidate pool
- `--variance-quantile`: filters out the highest-variance candidate stocks each day
- `--variance-weight`: penalty strength in the risk-adjusted ranking score
- `--market-variance-threshold`: if average daily market uncertainty is above this threshold, skip trading for that day
- `--num-mc-runs`: number of stochastic forward passes for MC Dropout
- `--deterministic-val`: tune `p / q / r` on deterministic validation predictions, then apply Bayesian inference only on test
- `--rebalance-frequency`: rebalance every `k` trading days instead of every day; this can materially reduce turnover and transaction costs
- `--eval-split`: choose `test` for separate validation/test reporting or `val_test` for a combined out-of-sample evaluation period

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
- On CPU, Bayesian backtesting can take a long time. Use `--num-mc-runs 20` for faster exploratory runs and `--num-mc-runs 100` for the standard MC setting.
- In our NASDAQ100 tests, lowering turnover with `--rebalance-frequency 5` produced much better baseline performance than daily rebalancing.
- Device selection is automatic: CUDA first, then MPS, then CPU.

## Citations 
- P. Tan, C. Shi, D. Tu, and L. Xie, “Magnet: A mamba
dual-hypergraph network for stock prediction via
temporal-causal and global relational learning,” 2025.
[Online]. Available: https://arxiv.org/abs/2511.00085
- J. A. Debo and G. Ciresi, “Predicting implicit patterns
and optimizing market entry and exit decisions in stock
prices using integrated bayesian cnn-lstm with deep
q-learning as a meta-labeller,” SSRN, March 2024.
[Online]. Available: https://ssrn.com/abstract=4794069
