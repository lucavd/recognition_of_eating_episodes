"""
CNN+LSTM End-to-End per Eating Detection - Stile Kyritsis et al. (2021)
Fase 8.5: Approccio end-to-end con CNN per feature extraction + LSTM per temporal modeling

Reference: Kyritsis, K., Diou, C., & Delopoulos, A. (2021). 
"A Data Driven End-to-End Approach for In-the-Wild Monitoring of Eating Behavior"
IEEE Journal of Biomedical and Health Informatics, 25(1), 22-34.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import recall_score, roc_auc_score, f1_score
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set seed for reproducibility
torch.manual_seed(2026)
np.random.seed(2026)

print("=" * 60)
print("=== FASE 8.5: CNN+LSTM END-TO-END (Kyritsis-style) ===")
print("=" * 60 + "\n")

# =============================================================================
# 1. LOAD DATA
# =============================================================================

print("1. Caricamento dati...")
data_path = Path(__file__).parent.parent / "02_cnn1d" / "ts_data.csv"
df = pd.read_csv(data_path)

print(f"   Righe: {len(df)}")
print(f"   Soggetti: {df['subject'].nunique()}")
print(f"   Eating: {df['EU'].sum()} ({df['EU'].mean()*100:.1f}%)")

# Prepare data: 6 channels x 25 timesteps
channels = ['acc_x', 'acc_y', 'acc_z', 'pitch', 'roll', 'power']
n_timesteps = 25
n_channels = len(channels)

def prepare_X(df):
    X = np.zeros((len(df), len(channels), n_timesteps))
    for i, ch in enumerate(channels):
        cols = [f"{ch}_{t}" for t in range(1, n_timesteps + 1)]
        X[:, i, :] = df[cols].values
    return X

X = prepare_X(df)
y = df['EU'].values
subjects = df['subject'].values

print(f"   X shape: {X.shape} (samples, channels, timesteps)")

# Normalize per channel (z-score)
for i in range(X.shape[1]):
    mean = np.nanmean(X[:, i, :])
    std = np.nanstd(X[:, i, :])
    X[:, i, :] = (X[:, i, :] - mean) / (std + 1e-8)

# Replace NaN with 0
X = np.nan_to_num(X, nan=0.0)

# =============================================================================
# 2. CNN+LSTM MODEL (Kyritsis-inspired architecture)
# =============================================================================

class CNN_LSTM(nn.Module):
    """
    End-to-end CNN+LSTM architecture inspired by Kyritsis et al. (2021)
    
    Architecture:
    1. CNN layers: Extract local temporal patterns from each channel
    2. LSTM layers: Capture temporal dependencies across the sequence
    3. FC layers: Classification head
    """
    
    def __init__(self, n_channels=6, n_timesteps=25, 
                 cnn_filters=[32, 64], kernel_size=3,
                 lstm_hidden=64, lstm_layers=2,
                 dropout=0.3, bidirectional=True):
        super(CNN_LSTM, self).__init__()
        
        self.n_channels = n_channels
        self.bidirectional = bidirectional
        
        # CNN Feature Extractor
        self.conv1 = nn.Conv1d(n_channels, cnn_filters[0], kernel_size=kernel_size, padding=1)
        self.bn1 = nn.BatchNorm1d(cnn_filters[0])
        self.conv2 = nn.Conv1d(cnn_filters[0], cnn_filters[1], kernel_size=kernel_size, padding=1)
        self.bn2 = nn.BatchNorm1d(cnn_filters[1])
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.dropout_cnn = nn.Dropout(dropout)
        
        # After 2 poolings: 25 -> 12 -> 6
        cnn_out_len = n_timesteps // 4  # ~6
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=cnn_filters[1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Classification head
        lstm_out_size = lstm_hidden * (2 if bidirectional else 1)
        self.fc1 = nn.Linear(lstm_out_size, 32)
        self.dropout_fc = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: (batch, channels, timesteps)
        
        # CNN feature extraction
        x = self.pool(self.relu(self.bn1(self.conv1(x))))  # -> (batch, 32, 12)
        x = self.pool(self.relu(self.bn2(self.conv2(x))))  # -> (batch, 64, 6)
        x = self.dropout_cnn(x)
        
        # Prepare for LSTM: (batch, timesteps, features)
        x = x.permute(0, 2, 1)  # -> (batch, 6, 64)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state (or concat of both directions)
        if self.bidirectional:
            # Concatenate last hidden states from both directions
            h_forward = h_n[-2, :, :]  # Last layer, forward
            h_backward = h_n[-1, :, :]  # Last layer, backward
            h_final = torch.cat([h_forward, h_backward], dim=1)
        else:
            h_final = h_n[-1, :, :]
        
        # Classification
        x = self.relu(self.fc1(h_final))
        x = self.dropout_fc(x)
        x = self.sigmoid(self.fc2(x))
        
        return x


class CNN_LSTM_Attention(nn.Module):
    """
    CNN+LSTM with Attention mechanism (più simile a Kyritsis)
    Usa attention sui timestep LSTM per pesare i momenti rilevanti
    """
    
    def __init__(self, n_channels=6, n_timesteps=25,
                 cnn_filters=[32, 64], kernel_size=3,
                 lstm_hidden=64, lstm_layers=2,
                 dropout=0.3, bidirectional=True):
        super(CNN_LSTM_Attention, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv1d(n_channels, cnn_filters[0], kernel_size=kernel_size, padding=1)
        self.bn1 = nn.BatchNorm1d(cnn_filters[0])
        self.conv2 = nn.Conv1d(cnn_filters[0], cnn_filters[1], kernel_size=kernel_size, padding=1)
        self.bn2 = nn.BatchNorm1d(cnn_filters[1])
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.dropout_cnn = nn.Dropout(dropout)
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_filters[1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_out_size = lstm_hidden * (2 if bidirectional else 1)
        
        # Attention layer
        self.attention = nn.Sequential(
            nn.Linear(lstm_out_size, lstm_out_size // 2),
            nn.Tanh(),
            nn.Linear(lstm_out_size // 2, 1)
        )
        
        # Classification head
        self.fc1 = nn.Linear(lstm_out_size, 32)
        self.dropout_fc = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # CNN
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.dropout_cnn(x)
        
        # LSTM
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, lstm_out_size)
        
        # Attention
        attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        # Weighted sum
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, lstm_out_size)
        
        # Classification
        x = self.relu(self.fc1(context))
        x = self.dropout_fc(x)
        x = self.sigmoid(self.fc2(x))
        
        return x


# =============================================================================
# 3. TRAINING FUNCTION
# =============================================================================

def train_model(model, train_loader, criterion, optimizer, epochs=50, device='cpu'):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
    return model


def evaluate_model(model, X_test, y_test, device='cpu'):
    model.eval()
    X_test_t = torch.FloatTensor(X_test).to(device)
    
    with torch.no_grad():
        y_prob = model(X_test_t).cpu().numpy().flatten()
    
    y_pred = (y_prob > 0.5).astype(int)
    
    sens = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    spec = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    try:
        auc = roc_auc_score(y_test, y_prob)
    except:
        auc = 0.5
    
    return sens, spec, auc, f1, y_prob


# =============================================================================
# 4. LOSO VALIDATION
# =============================================================================

print("\n2. LOSO Validation con CNN+LSTM...")

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"   Device: {device}")

unique_subjects = np.unique(subjects)
print(f"   Soggetti: {len(unique_subjects)}")

# Test both architectures
architectures = {
    'CNN_LSTM': CNN_LSTM,
    'CNN_LSTM_Attention': CNN_LSTM_Attention
}

all_results = []

for arch_name, arch_class in architectures.items():
    print(f"\n{'='*50}")
    print(f"Testing: {arch_name}")
    print('='*50)
    
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
        
        # Class weight for imbalance
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        pos_weight = torch.FloatTensor([n_neg / max(n_pos, 1)]).to(device)
        
        # Tensors
        X_train_t = torch.FloatTensor(X_train).to(device)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
        
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        
        # Model
        model = arch_class(
            n_channels=n_channels,
            n_timesteps=n_timesteps,
            cnn_filters=[32, 64],
            lstm_hidden=64,
            lstm_layers=2,
            dropout=0.4,
            bidirectional=True
        ).to(device)
        
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        # Change last layer to not use sigmoid (use BCEWithLogitsLoss)
        model.sigmoid = nn.Identity()  # Remove sigmoid for BCEWithLogitsLoss
        
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Train
        model = train_model(model, train_loader, criterion, optimizer, epochs=50, device=device)
        
        # Re-add sigmoid for evaluation
        model.sigmoid = nn.Sigmoid()
        
        # Evaluate
        sens, spec, auc, f1, _ = evaluate_model(model, X_test, y_test, device)
        
        results.append({
            'architecture': arch_name,
            'subject': subj,
            'sens': sens,
            'spec': spec,
            'auc': auc,
            'f1': f1
        })
        print(f"   Subject {subj}: sens={sens:.3f}, spec={spec:.3f}, auc={auc:.3f}, f1={f1:.3f}")
    
    # Aggregate results
    results_df = pd.DataFrame(results)
    mean_sens = results_df['sens'].mean()
    mean_spec = results_df['spec'].mean()
    mean_auc = results_df['auc'].mean()
    mean_f1 = results_df['f1'].mean()
    std_sens = results_df['sens'].std()
    
    print(f"\n   {arch_name} AGGREGATE:")
    print(f"   Sensitivity: {mean_sens:.3f} (± {std_sens:.3f})")
    print(f"   Specificity: {mean_spec:.3f}")
    print(f"   AUC: {mean_auc:.3f}")
    print(f"   F1: {mean_f1:.3f}")
    
    all_results.extend(results)

# =============================================================================
# 5. FINAL RESULTS
# =============================================================================

print("\n" + "="*60)
print("=== RISULTATI FINALI CNN+LSTM ===")
print("="*60 + "\n")

all_results_df = pd.DataFrame(all_results)

# Summary per architecture
summary = all_results_df.groupby('architecture').agg({
    'sens': ['mean', 'std'],
    'spec': ['mean', 'std'],
    'auc': ['mean', 'std'],
    'f1': ['mean', 'std']
}).round(3)

print("SUMMARY PER ARCHITETTURA:")
print(summary)
print()

# Compare with baselines
print("\nCONFRONTO CON BASELINE:")
print("-" * 50)
print(f"{'Method':<25} {'Sens':>8} {'Spec':>8} {'AUC':>8}")
print("-" * 50)
print(f"{'XGBoost baseline':<25} {'0.417':>8} {'0.814':>8} {'0.693':>8}")
print(f"{'XGB tuned (best ML)':<25} {'0.593':>8} {'0.692':>8} {'0.712':>8}")
print(f"{'CNN-1D':<25} {'0.330':>8} {'0.827':>8} {'0.615':>8}")
print(f"{'Attention':<25} {'0.498':>8} {'0.709':>8} {'0.640':>8}")
print(f"{'Transformer':<25} {'0.703':>8} {'0.476':>8} {'0.638':>8}")
print("-" * 50)

for arch in all_results_df['architecture'].unique():
    arch_data = all_results_df[all_results_df['architecture'] == arch]
    sens = arch_data['sens'].mean()
    spec = arch_data['spec'].mean()
    auc = arch_data['auc'].mean()
    print(f"{arch:<25} {sens:>8.3f} {spec:>8.3f} {auc:>8.3f}")

print("-" * 50)

# Save results
output_path = Path(__file__).parent / "cnn_lstm_results.csv"
all_results_df.to_csv(output_path, index=False)
print(f"\nResults saved to: {output_path}")

# Best result
best_sens = all_results_df.groupby('architecture')['sens'].mean().idxmax()
best_val = all_results_df.groupby('architecture')['sens'].mean().max()
print(f"\nBest architecture for sensitivity: {best_sens} ({best_val:.3f})")

delta_vs_baseline = (best_val - 0.417) * 100
delta_vs_xgb_tuned = (best_val - 0.593) * 100
print(f"Delta vs XGBoost baseline: {delta_vs_baseline:+.1f}%")
print(f"Delta vs XGB tuned: {delta_vs_xgb_tuned:+.1f}%")
