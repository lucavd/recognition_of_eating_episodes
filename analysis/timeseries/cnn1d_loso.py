"""
CNN-1D per Eating Detection - LOSO Validation
Fase 8.2: Convolutional Neural Network su time series raw
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import recall_score, roc_auc_score, confusion_matrix
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set seed
torch.manual_seed(2026)
np.random.seed(2026)

print("=== FASE 8.2: CNN-1D TIME SERIES ===\n")

# =============================================================================
# 1. LOAD DATA
# =============================================================================

print("1. Caricamento dati...")
data_path = Path(__file__).parent / "ts_data.csv"
df = pd.read_csv(data_path)

print(f"   Righe: {len(df)}")
print(f"   Colonne: {df.shape[1]}")
print(f"   Soggetti: {df['subject'].nunique()}")
print(f"   Eating: {df['EU'].sum()} ({df['EU'].mean()*100:.1f}%)")

# Prepare data: 6 channels x 25 timesteps
channels = ['acc_x', 'acc_y', 'acc_z', 'pitch', 'roll', 'power']
n_timesteps = 25

def prepare_X(df):
    X = np.zeros((len(df), len(channels), n_timesteps))
    for i, ch in enumerate(channels):
        cols = [f"{ch}_{t}" for t in range(1, n_timesteps + 1)]
        X[:, i, :] = df[cols].values
    return X

X = prepare_X(df)
y = df['EU'].values
subjects = df['subject'].values

print(f"   X shape: {X.shape}")

# Normalize per channel
for i in range(X.shape[1]):
    mean = np.nanmean(X[:, i, :])
    std = np.nanstd(X[:, i, :])
    X[:, i, :] = (X[:, i, :] - mean) / (std + 1e-8)

# Replace NaN with 0
X = np.nan_to_num(X, nan=0.0)

# =============================================================================
# 2. CNN-1D MODEL
# =============================================================================

class CNN1D(nn.Module):
    def __init__(self, n_channels=6, n_timesteps=25):
        super(CNN1D, self).__init__()
        
        self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        
        # After 3 poolings: 25 -> 12 -> 6 -> 3
        self.fc1 = nn.Linear(128 * 3, 64)
        self.fc2 = nn.Linear(64, 1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.sigmoid(self.fc2(x))
        return x

# =============================================================================
# 3. LOSO VALIDATION
# =============================================================================

print("\n2. LOSO Validation...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"   Device: {device}")

unique_subjects = np.unique(subjects)
results = []

for subj in unique_subjects:
    # Split
    test_mask = subjects == subj
    train_mask = ~test_mask
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    if len(np.unique(y_test)) < 2:
        print(f"   Subject {subj}: skipped (single class)")
        continue
    
    # Class weight
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    
    # Tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Model
    model = CNN1D().to(device)
    criterion = nn.BCELoss(weight=torch.FloatTensor([pos_weight]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train
    model.train()
    for epoch in range(30):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        y_prob = model(X_test_t).cpu().numpy().flatten()
        y_pred = (y_prob > 0.5).astype(int)
    
    sens = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    spec = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except:
        auc = 0.5
    
    results.append({
        'subject': subj,
        'sens': sens,
        'spec': spec,
        'auc': auc
    })
    print(f"   Subject {subj}: sens={sens:.3f}, spec={spec:.3f}, auc={auc:.3f}")

# =============================================================================
# 4. RESULTS
# =============================================================================

print("\n" + "="*50)
print("=== RISULTATI CNN-1D + LOSO ===")
print("="*50 + "\n")

results_df = pd.DataFrame(results)
print("Per soggetto:")
print(results_df.to_string(index=False))

mean_sens = results_df['sens'].mean()
mean_spec = results_df['spec'].mean()
mean_auc = results_df['auc'].mean()
std_sens = results_df['sens'].std()

print(f"\n\nMETRICHE AGGREGATE:")
print(f"Sensitivity: {mean_sens:.3f} (± {std_sens:.3f})")
print(f"Specificity: {mean_spec:.3f}")
print(f"AUC: {mean_auc:.3f}")

print(f"\n\nCONFRONTO CON BASELINE:")
print(f"Baseline LOSO:  sens = 0.417")
print(f"MiniRocket:     sens = 0.388")
print(f"CNN-1D:         sens = {mean_sens:.3f}")
print(f"Delta vs base:  {(mean_sens - 0.417) * 100:+.1f}%")

# Save results
results_df.to_csv(Path(__file__).parent / "cnn1d_results.csv", index=False)
print(f"\nSaved to 02_cnn1d/cnn1d_results.csv")
