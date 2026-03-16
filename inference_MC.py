import torch
import importlib
from torch.utils.data import DataLoader
from Dataset import StockDataset

# CHANGE THIS to 'Magnetv1', 'Magnetv2', or 'Magnetv3' 
MODEL_VERSION = 'Magnetv1'
MaGNet = importlib.import_module(MODEL_VERSION).MaGNet
print(f"Using model: {MODEL_VERSION}")

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# same settings as train.py
data_path = "my_nas100_2025_data.pt"
weight_path = f"best_model_{MODEL_VERSION}.pth"

dim = 32
num_experts = 4
num_heads_mha = 2
num_channels = 4
num_heads_CausalMHA = 2
T = 10
batch_size = 24
num_MAGE = 1
num_F2DAttn = 1
num_TCH = 2
TopK = 64
M1 = 64
num_S2DAttn = 1
num_GPH = 2
M2 = 32
num_mc_runs = 100

# load data
data = torch.load(data_path, weights_only=True).to(device)

total_date = data.shape[1]
train_cutoff = int(total_date * 0.7)
valid_cutoff = train_cutoff + int(total_date * 0.1)

test_data = data[:, valid_cutoff:]

epsilon = 1e-6
test_data_mean = test_data.mean(dim=1, keepdim=True)
test_data_std = test_data.std(dim=1, keepdim=True)
test_data = (test_data - test_data_mean) / (test_data_std + epsilon)

test_dataset = StockDataset(test_data, T, device)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# rebuild model
N = data.shape[0]
F = data.shape[2]

model = MaGNet(
    N, T, F, dim, num_MAGE, num_experts,
    num_heads_mha, num_F2DAttn, num_channels,
    num_heads_CausalMHA, num_TCH, TopK, M1,
    num_S2DAttn, num_GPH, M2,
    device=device,
    dropout=0.1
).to(device)

model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))

# keep dropout ON
model.train()

all_mc_preds = []

print("Running MC Dropout inference...")

with torch.no_grad():
    for run in range(num_mc_runs):
        one_run_preds = []

        for X_batch, _ in test_loader:
            batch_size_actual = X_batch.size(0)
            for b in range(batch_size_actual):
                X = X_batch[b].to(device)  # [N, T, F]
                output, *_ = model(X)
                prob = torch.softmax(output, dim=-1)
                one_run_preds.append(prob)

        one_run_preds = torch.cat(one_run_preds, dim=0)
        all_mc_preds.append(one_run_preds)

        if (run + 1) % 10 == 0:
            print(f"  MC run {run + 1}/{num_mc_runs}")

all_mc_preds = torch.stack(all_mc_preds, dim=0)

mean_pred = all_mc_preds.mean(dim=0)
var_pred = all_mc_preds.var(dim=0)

print("mean_pred shape:", mean_pred.shape)
print("var_pred shape:", var_pred.shape)

torch.save(
    {
        "mean_pred": mean_pred.cpu(),
        "var_pred": var_pred.cpu(),
    },
    f"mc_results_{MODEL_VERSION}.pt"
)

print(f"Saved to mc_results_{MODEL_VERSION}.pt")
