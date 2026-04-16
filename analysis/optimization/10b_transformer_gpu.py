"""
10b: Transformer End-to-End con GPU
Fase 10b: Transformer allenato end-to-end su time series per eating detection.

Architettura:
- Input: 6 canali × 25 timesteps
- Positional encoding (learnable)
- Multi-head self-attention (2-4 layers)
- Classification head

Training:
- LOSO validation (19 soggetti)
- Class weights per sbilanciamento
- Dropout aggressivo (0.3-0.5)
- Early stopping

Requisiti: PyTorch con CUDA
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import recall_score, roc_auc_score, balanced_accuracy_score
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Seed for reproducibility
np.random.seed(2026)
torch.manual_seed(2026)
if torch.cuda.is_available():
    torch.cuda.manual_seed(2026)

print("=" * 60)
print("=== FASE 10b: TRANSFORMER END-TO-END (GPU) ===")
print("=" * 60 + "\n", flush=True)

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print()

# =============================================================================
# 1. LOAD DATA
# =============================================================================

print("1. Caricamento dati...", flush=True)
data_path = Path(__file__).parent.parent / "08_timeseries" / "02_cnn1d" / "ts_data.csv"
df = pd.read_csv(data_path)

print(f"   Righe: {len(df)}", flush=True)

# Reshape to (samples, timesteps, channels)
channels = ['acc_x', 'acc_y', 'acc_z', 'pitch', 'roll', 'power']
n_timesteps = 25
n_channels = len(channels)

X_3d = np.zeros((len(df), n_timesteps, n_channels))
for i, ch in enumerate(channels):
    cols = [f"{ch}_{t}" for t in range(1, n_timesteps + 1)]
    X_3d[:, :, i] = df[cols].values

y = df['EU'].values
subjects = df['subject'].values

# Normalize per channel
for i in range(n_channels):
    mean = np.nanmean(X_3d[:, :, i])
    std = np.nanstd(X_3d[:, :, i])
    X_3d[:, :, i] = (X_3d[:, :, i] - mean) / (std + 1e-8)

X_3d = np.nan_to_num(X_3d, nan=0.0)
print(f"   X shape: {X_3d.shape}", flush=True)

unique_subjects = np.unique(subjects)
print(f"   Soggetti: {len(unique_subjects)}\n", flush=True)

# =============================================================================
# 2. TRANSFORMER MODEL
# =============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Learnable positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=6, seq_len=25, d_model=64, nhead=4, 
                 num_layers=2, dim_feedforward=128, dropout=0.3):
        super().__init__()
        
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        
        # Project input
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        
        # Global average pooling over sequence
        x = x.mean(dim=1)  # (batch, d_model)
        
        # Classification
        x = self.classifier(x)  # (batch, 1)
        
        return x.squeeze(-1)


# =============================================================================
# 3. TRAINING FUNCTION
# =============================================================================

def train_and_evaluate(X_train, y_train, X_test, y_test, config, device):
    """
    Train transformer and evaluate on test set.
    """
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.FloatTensor(y_test).to(device)
    
    # Class weights for loss
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos]).to(device)
    
    # Weighted sampler for balanced batches
    class_weights = np.where(y_train == 1, n_neg / n_pos, 1.0)
    sampler = WeightedRandomSampler(
        weights=class_weights,
        num_samples=len(y_train),
        replacement=True
    )
    
    # DataLoader
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        sampler=sampler
    )
    
    # Model
    model = TransformerClassifier(
        input_dim=config['input_dim'],
        seq_len=config['seq_len'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout']
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        epoch_loss /= len(train_loader)
        scheduler.step(epoch_loss)
        
        # Early stopping
        if epoch_loss < best_loss - 0.001:
            best_loss = epoch_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        y_logits = model(X_test_t)
        y_prob = torch.sigmoid(y_logits).cpu().numpy()
        y_pred = (y_prob > 0.5).astype(int)
        y_test_np = y_test_t.cpu().numpy()
    
    sens = recall_score(y_test_np, y_pred, pos_label=1, zero_division=0)
    spec = recall_score(y_test_np, y_pred, pos_label=0, zero_division=0)
    bal_acc = balanced_accuracy_score(y_test_np, y_pred)
    
    try:
        auc = roc_auc_score(y_test_np, y_prob)
    except:
        auc = 0.5
    
    return {
        'sens': sens,
        'spec': spec,
        'bal_acc': bal_acc,
        'auc': auc,
        'y_prob': y_prob,
        'y_true': y_test_np
    }


# =============================================================================
# 4. CONFIGURATIONS TO TEST
# =============================================================================

print("2. Definizione configurazioni...", flush=True)

configs = [
    # Baseline
    {
        'name': 'baseline',
        'd_model': 64, 'nhead': 4, 'num_layers': 2, 'dim_feedforward': 128,
        'dropout': 0.3, 'learning_rate': 1e-3, 'weight_decay': 1e-4,
        'batch_size': 64, 'epochs': 100, 'patience': 15
    },
    # Deeper
    {
        'name': 'deep',
        'd_model': 64, 'nhead': 4, 'num_layers': 4, 'dim_feedforward': 128,
        'dropout': 0.4, 'learning_rate': 5e-4, 'weight_decay': 1e-4,
        'batch_size': 64, 'epochs': 100, 'patience': 15
    },
    # Wider
    {
        'name': 'wide',
        'd_model': 128, 'nhead': 8, 'num_layers': 2, 'dim_feedforward': 256,
        'dropout': 0.3, 'learning_rate': 5e-4, 'weight_decay': 1e-4,
        'batch_size': 64, 'epochs': 100, 'patience': 15
    },
    # More regularization
    {
        'name': 'regularized',
        'd_model': 64, 'nhead': 4, 'num_layers': 2, 'dim_feedforward': 128,
        'dropout': 0.5, 'learning_rate': 1e-3, 'weight_decay': 1e-3,
        'batch_size': 32, 'epochs': 100, 'patience': 15
    },
    # Small but deep
    {
        'name': 'small_deep',
        'd_model': 32, 'nhead': 4, 'num_layers': 4, 'dim_feedforward': 64,
        'dropout': 0.4, 'learning_rate': 1e-3, 'weight_decay': 1e-4,
        'batch_size': 64, 'epochs': 100, 'patience': 15
    },
    # Large
    {
        'name': 'large',
        'd_model': 128, 'nhead': 8, 'num_layers': 4, 'dim_feedforward': 256,
        'dropout': 0.4, 'learning_rate': 3e-4, 'weight_decay': 1e-4,
        'batch_size': 32, 'epochs': 150, 'patience': 20
    },
]

# Add common params
for cfg in configs:
    cfg['input_dim'] = n_channels
    cfg['seq_len'] = n_timesteps

print(f"   Configurazioni da testare: {len(configs)}\n", flush=True)

# =============================================================================
# 5. LOSO VALIDATION FOR EACH CONFIG
# =============================================================================

print("3. LOSO Validation per ogni configurazione...\n", flush=True)

all_results = []

for cfg_idx, config in enumerate(configs):
    print(f"\n{'='*60}")
    print(f"CONFIG {cfg_idx+1}/{len(configs)}: {config['name']}")
    print(f"  d_model={config['d_model']}, nhead={config['nhead']}, "
          f"layers={config['num_layers']}, dropout={config['dropout']}")
    print(f"{'='*60}\n", flush=True)
    
    fold_results = []
    
    for subj_idx, subj in enumerate(unique_subjects):
        test_mask = subjects == subj
        train_mask = ~test_mask
        
        X_train, y_train = X_3d[train_mask], y[train_mask]
        X_test, y_test = X_3d[test_mask], y[test_mask]
        
        if len(np.unique(y_test)) < 2:
            continue
        
        result = train_and_evaluate(X_train, y_train, X_test, y_test, config, device)
        fold_results.append(result)
        
        print(f"  Subject {subj}: sens={result['sens']:.3f}, "
              f"spec={result['spec']:.3f}, bal_acc={result['bal_acc']:.3f}", flush=True)
    
    # Aggregate
    mean_sens = np.mean([r['sens'] for r in fold_results])
    mean_spec = np.mean([r['spec'] for r in fold_results])
    mean_bal_acc = np.mean([r['bal_acc'] for r in fold_results])
    mean_auc = np.mean([r['auc'] for r in fold_results])
    
    std_sens = np.std([r['sens'] for r in fold_results])
    
    all_results.append({
        'config_name': config['name'],
        'd_model': config['d_model'],
        'nhead': config['nhead'],
        'num_layers': config['num_layers'],
        'dim_feedforward': config['dim_feedforward'],
        'dropout': config['dropout'],
        'learning_rate': config['learning_rate'],
        'sens': mean_sens,
        'spec': mean_spec,
        'bal_acc': mean_bal_acc,
        'auc': mean_auc,
        'sens_std': std_sens
    })
    
    print(f"\n  MEAN: sens={mean_sens:.3f}±{std_sens:.3f}, "
          f"spec={mean_spec:.3f}, bal_acc={mean_bal_acc:.3f}, auc={mean_auc:.3f}", flush=True)

# =============================================================================
# 6. RESULTS
# =============================================================================

print("\n\n" + "=" * 60)
print("=== RISULTATI TRANSFORMER ===")
print("=" * 60 + "\n", flush=True)

results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values('bal_acc', ascending=False)

print("TUTTE LE CONFIGURAZIONI (ordinate per bal_acc):\n")
print(results_df.to_string(index=False), flush=True)

best = results_df.iloc[0]
print(f"\n\nMIGLIOR CONFIGURAZIONE: {best['config_name']}")
print(f"  d_model: {best['d_model']}")
print(f"  nhead: {best['nhead']}")
print(f"  num_layers: {best['num_layers']}")
print(f"  dropout: {best['dropout']}")
print(f"  Balanced Accuracy: {best['bal_acc']:.3f}")
print(f"  Sensitivity: {best['sens']:.3f} ± {best['sens_std']:.3f}")
print(f"  Specificity: {best['spec']:.3f}")
print(f"  AUC: {best['auc']:.3f}")

print(f"\n\nCONFRONTO:")
print(f"  Attention default:  sens=0.498, spec=0.709, bal_acc=0.604")
print(f"  XGB tuned:          sens=0.593, spec=0.692, bal_acc=0.643")
print(f"  Transformer best:   sens={best['sens']:.3f}, "
      f"spec={best['spec']:.3f}, bal_acc={best['bal_acc']:.3f}")

# =============================================================================
# 7. SAVE RESULTS
# =============================================================================

output_dir = Path(__file__).parent
output_dir.mkdir(parents=True, exist_ok=True)

results_df.to_csv(output_dir / "transformer_results.csv", index=False)

best_config = configs[results_df.index[0]]
with open(output_dir / "best_transformer_config.json", "w") as f:
    json.dump({k: v for k, v in best_config.items() if k != 'name'}, f, indent=2)

print(f"\nRisultati salvati in:")
print(f"  - {output_dir / 'transformer_results.csv'}")
print(f"  - {output_dir / 'best_transformer_config.json'}")

print("\nDONE!", flush=True)
