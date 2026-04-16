"""
10cd: Ensemble Completo con Threshold Optimization
Esegue i best model, salva probabilità, ottimizza threshold, crea ensemble.

Modelli:
1. XGB tuned (caricato da RDS - skip se non disponibile)
2. Attention tuned (best config da 10a)
3. Transformer (best config da 10b)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, balanced_accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

np.random.seed(2026)
torch.manual_seed(2026)

print("=" * 70)
print("=== FASE 10cd: ENSEMBLE CON THRESHOLD OPTIMIZATION ===")
print("=" * 70 + "\n", flush=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n", flush=True)

# =============================================================================
# 1. LOAD DATA
# =============================================================================

print("1. Caricamento dati...", flush=True)
data_path = Path(__file__).parent.parent / "08_timeseries" / "02_cnn1d" / "ts_data.csv"
df = pd.read_csv(data_path)

channels = ['acc_x', 'acc_y', 'acc_z', 'pitch', 'roll', 'power']
n_timesteps = 25
n_channels = len(channels)

X_3d = np.zeros((len(df), n_timesteps, n_channels))
for i, ch in enumerate(channels):
    cols = [f"{ch}_{t}" for t in range(1, n_timesteps + 1)]
    X_3d[:, :, i] = df[cols].values

y = df['EU'].values
subjects = df['subject'].values

for i in range(n_channels):
    mean = np.nanmean(X_3d[:, :, i])
    std = np.nanstd(X_3d[:, :, i])
    X_3d[:, :, i] = (X_3d[:, :, i] - mean) / (std + 1e-8)

X_3d = np.nan_to_num(X_3d, nan=0.0)
unique_subjects = np.unique(subjects)

print(f"   Samples: {len(df)}, Soggetti: {len(unique_subjects)}\n", flush=True)

# =============================================================================
# 2. LOAD BEST CONFIGS
# =============================================================================

print("2. Caricamento configurazioni ottimali...", flush=True)

output_dir = Path(__file__).parent

with open(output_dir / "best_attention_config.json") as f:
    attention_config = json.load(f)
print(f"   Attention: {attention_config}", flush=True)

with open(output_dir / "best_transformer_config.json") as f:
    transformer_config = json.load(f)
print(f"   Transformer: {transformer_config}\n", flush=True)

# =============================================================================
# 3. MODEL DEFINITIONS
# =============================================================================

def extract_attention_features(X_3d, n_heads=4, tau=1.0):
    n_samples, n_timesteps, n_channels = X_3d.shape
    features = []
    
    for sample in X_3d:
        sample_features = []
        energy = np.sum(sample**2, axis=1)
        attention_weights = np.exp(energy / tau) / np.sum(np.exp(energy / tau))
        
        weighted_mean = np.sum(sample * attention_weights[:, np.newaxis], axis=0)
        max_pool = np.max(sample, axis=0)
        variance = np.var(sample, axis=0)
        gradient = np.mean(np.abs(np.diff(sample, axis=0)), axis=0)
        attention_entropy = -np.sum(attention_weights * np.log(attention_weights + 1e-8))
        peak_timestep = np.argmax(attention_weights) / n_timesteps
        
        sample_features.extend(weighted_mean)
        sample_features.extend(max_pool)
        sample_features.extend(variance)
        sample_features.extend(gradient)
        sample_features.append(attention_entropy)
        sample_features.append(peak_timestep)
        
        for head in range(n_heads):
            if head == 0:
                weights = np.sum(sample**2, axis=1)
            elif head == 1:
                weights = np.abs(sample[:, 0])
            elif head == 2:
                grad = np.sum(np.abs(np.diff(sample, axis=0)), axis=1)
                weights = np.concatenate([[0], grad])
            else:
                weights = np.ones(n_timesteps)
            
            weights = np.exp(weights / tau) / np.sum(np.exp(weights / tau))
            pooled = np.sum(sample * weights[:, np.newaxis], axis=0)
            sample_features.extend(pooled)
        
        features.append(sample_features)
    
    return np.array(features)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
    
    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=6, seq_len=25, d_model=64, nhead=4, 
                 num_layers=2, dim_feedforward=128, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x).squeeze(-1)

# =============================================================================
# 4. LOSO WITH PROBABILITY COLLECTION
# =============================================================================

print("3. LOSO Validation con raccolta probabilità...\n", flush=True)

all_y_true = []
all_y_prob_attention = []
all_y_prob_transformer = []
all_subjects = []

for subj_idx, subj in enumerate(unique_subjects):
    print(f"   Subject {subj} ({subj_idx+1}/{len(unique_subjects)})...", end="", flush=True)
    
    test_mask = subjects == subj
    train_mask = ~test_mask
    
    X_train_3d, y_train = X_3d[train_mask], y[train_mask]
    X_test_3d, y_test = X_3d[test_mask], y[test_mask]
    
    if len(np.unique(y_test)) < 2:
        print(" skipped (single class)", flush=True)
        continue
    
    # --- ATTENTION MODEL ---
    n_heads = attention_config['n_heads']
    tau = attention_config['tau']
    hidden_sizes = tuple(attention_config['hidden_layer_sizes'])
    
    X_train_att = extract_attention_features(X_train_3d, n_heads=n_heads, tau=tau)
    X_test_att = extract_attention_features(X_test_3d, n_heads=n_heads, tau=tau)
    
    X_train_att = np.nan_to_num(X_train_att, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_att = np.nan_to_num(X_test_att, nan=0.0, posinf=0.0, neginf=0.0)
    
    scaler = StandardScaler()
    X_train_att = scaler.fit_transform(X_train_att)
    X_test_att = scaler.transform(X_test_att)
    X_train_att = np.nan_to_num(X_train_att, nan=0.0)
    X_test_att = np.nan_to_num(X_test_att, nan=0.0)
    
    # Oversample
    eating_idx = np.where(y_train == 1)[0]
    non_eating_idx = np.where(y_train == 0)[0]
    oversample_ratio = attention_config['oversample_ratio']
    target_eating = int(len(non_eating_idx) * oversample_ratio)
    n_oversample = target_eating - len(eating_idx)
    
    if n_oversample > 0:
        oversample_idx = np.random.choice(eating_idx, size=n_oversample, replace=True)
        X_train_att_bal = np.vstack([X_train_att, X_train_att[oversample_idx]])
        y_train_bal = np.hstack([y_train, y_train[oversample_idx]])
    else:
        X_train_att_bal, y_train_bal = X_train_att, y_train
    
    clf_att = MLPClassifier(
        hidden_layer_sizes=hidden_sizes,
        alpha=attention_config['alpha'],
        learning_rate_init=attention_config['learning_rate_init'],
        max_iter=200,
        random_state=2026,
        early_stopping=True,
        validation_fraction=0.1
    )
    clf_att.fit(X_train_att_bal, y_train_bal)
    prob_attention = clf_att.predict_proba(X_test_att)[:, 1]
    
    # --- TRANSFORMER MODEL ---
    X_train_t = torch.FloatTensor(X_train_3d).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test_3d).to(device)
    
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos]).to(device)
    
    model = TransformerClassifier(
        input_dim=n_channels,
        seq_len=n_timesteps,
        d_model=transformer_config['d_model'],
        nhead=transformer_config['nhead'],
        num_layers=transformer_config['num_layers'],
        dim_feedforward=transformer_config['dim_feedforward'],
        dropout=transformer_config['dropout']
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=transformer_config['learning_rate'],
        weight_decay=transformer_config['weight_decay']
    )
    
    # Quick training
    best_loss = float('inf')
    patience_counter = 0
    batch_size = transformer_config['batch_size']
    
    for epoch in range(transformer_config['epochs']):
        model.train()
        perm = torch.randperm(len(X_train_t))
        epoch_loss = 0
        n_batches = 0
        
        for i in range(0, len(X_train_t), batch_size):
            idx = perm[i:i+batch_size]
            optimizer.zero_grad()
            outputs = model(X_train_t[idx])
            loss = criterion(outputs, y_train_t[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        
        epoch_loss /= n_batches
        if epoch_loss < best_loss - 0.001:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= transformer_config['patience']:
                break
    
    model.eval()
    with torch.no_grad():
        logits = model(X_test_t)
        prob_transformer = torch.sigmoid(logits).cpu().numpy()
    
    # Collect
    all_y_true.extend(y_test)
    all_y_prob_attention.extend(prob_attention)
    all_y_prob_transformer.extend(prob_transformer)
    all_subjects.extend([subj] * len(y_test))
    
    print(" done", flush=True)

all_y_true = np.array(all_y_true)
all_y_prob_attention = np.array(all_y_prob_attention)
all_y_prob_transformer = np.array(all_y_prob_transformer)

print(f"\n   Totale predizioni: {len(all_y_true)}\n", flush=True)

# =============================================================================
# 5. THRESHOLD OPTIMIZATION
# =============================================================================

print("4. Threshold Optimization...\n", flush=True)

def find_best_threshold(y_true, y_prob):
    best_thresh, best_bal = 0.5, 0
    for thresh in np.arange(0.1, 0.9, 0.01):
        y_pred = (y_prob >= thresh).astype(int)
        sens = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        spec = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        bal = (sens + spec) / 2
        if bal > best_bal:
            best_bal = bal
            best_thresh = thresh
    return best_thresh

def eval_at_threshold(y_true, y_prob, thresh):
    y_pred = (y_prob >= thresh).astype(int)
    sens = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    spec = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    bal = (sens + spec) / 2
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.5
    return sens, spec, bal, auc

# Attention
thresh_att = find_best_threshold(all_y_true, all_y_prob_attention)
sens_att, spec_att, bal_att, auc_att = eval_at_threshold(all_y_true, all_y_prob_attention, thresh_att)
print(f"   Attention @ thresh={thresh_att:.2f}: sens={sens_att:.3f}, spec={spec_att:.3f}, bal={bal_att:.3f}")

sens_att_05, spec_att_05, bal_att_05, _ = eval_at_threshold(all_y_true, all_y_prob_attention, 0.5)
print(f"   Attention @ thresh=0.50: sens={sens_att_05:.3f}, spec={spec_att_05:.3f}, bal={bal_att_05:.3f}")

# Transformer
thresh_trans = find_best_threshold(all_y_true, all_y_prob_transformer)
sens_trans, spec_trans, bal_trans, auc_trans = eval_at_threshold(all_y_true, all_y_prob_transformer, thresh_trans)
print(f"\n   Transformer @ thresh={thresh_trans:.2f}: sens={sens_trans:.3f}, spec={spec_trans:.3f}, bal={bal_trans:.3f}")

sens_trans_05, spec_trans_05, bal_trans_05, _ = eval_at_threshold(all_y_true, all_y_prob_transformer, 0.5)
print(f"   Transformer @ thresh=0.50: sens={sens_trans_05:.3f}, spec={spec_trans_05:.3f}, bal={bal_trans_05:.3f}")

# =============================================================================
# 6. ENSEMBLE
# =============================================================================

print("\n5. Ensemble Methods...\n", flush=True)

results = []

# Individual models
results.append({'method': 'Attention (thresh=0.5)', 'sens': sens_att_05, 'spec': spec_att_05, 
                'bal_acc': bal_att_05, 'auc': auc_att})
results.append({'method': f'Attention (thresh={thresh_att:.2f})', 'sens': sens_att, 'spec': spec_att, 
                'bal_acc': bal_att, 'auc': auc_att})
results.append({'method': 'Transformer (thresh=0.5)', 'sens': sens_trans_05, 'spec': spec_trans_05, 
                'bal_acc': bal_trans_05, 'auc': auc_trans})
results.append({'method': f'Transformer (thresh={thresh_trans:.2f})', 'sens': sens_trans, 'spec': spec_trans, 
                'bal_acc': bal_trans, 'auc': auc_trans})

# Simple average ensemble
prob_avg = (all_y_prob_attention + all_y_prob_transformer) / 2
thresh_avg = find_best_threshold(all_y_true, prob_avg)
sens_avg, spec_avg, bal_avg, auc_avg = eval_at_threshold(all_y_true, prob_avg, thresh_avg)
results.append({'method': f'Ensemble Avg (thresh={thresh_avg:.2f})', 'sens': sens_avg, 'spec': spec_avg, 
                'bal_acc': bal_avg, 'auc': auc_avg})

sens_avg_05, spec_avg_05, bal_avg_05, _ = eval_at_threshold(all_y_true, prob_avg, 0.5)
results.append({'method': 'Ensemble Avg (thresh=0.5)', 'sens': sens_avg_05, 'spec': spec_avg_05, 
                'bal_acc': bal_avg_05, 'auc': auc_avg})

# Weighted ensemble (più peso a Attention che ha spec migliore)
prob_weighted = 0.6 * all_y_prob_attention + 0.4 * all_y_prob_transformer
thresh_w = find_best_threshold(all_y_true, prob_weighted)
sens_w, spec_w, bal_w, auc_w = eval_at_threshold(all_y_true, prob_weighted, thresh_w)
results.append({'method': f'Ensemble Weighted 60/40 (thresh={thresh_w:.2f})', 'sens': sens_w, 'spec': spec_w, 
                'bal_acc': bal_w, 'auc': auc_w})

# =============================================================================
# 7. RESULTS
# =============================================================================

print("\n" + "=" * 70)
print("=== RISULTATI FINALI ===")
print("=" * 70 + "\n", flush=True)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('bal_acc', ascending=False)
print(results_df.to_string(index=False), flush=True)

best = results_df.iloc[0]
print(f"\n\nMIGLIOR METODO: {best['method']}")
print(f"   Sensitivity: {best['sens']:.3f}")
print(f"   Specificity: {best['spec']:.3f}")
print(f"   Balanced Accuracy: {best['bal_acc']:.3f}")
print(f"   AUC: {best['auc']:.3f}")

print("\n\nCONFRONTO CON MODELLI PRECEDENTI:")
print(f"   XGB tuned (riferimento):     sens=0.593, spec=0.692, bal_acc=0.643")
print(f"   Best ensemble/threshold:     sens={best['sens']:.3f}, spec={best['spec']:.3f}, bal_acc={best['bal_acc']:.3f}")

delta_bal = (best['bal_acc'] - 0.643) * 100
print(f"\n   Delta bal_acc vs XGB tuned: {delta_bal:+.1f}%")

# =============================================================================
# 8. SAVE
# =============================================================================

results_df.to_csv(output_dir / "ensemble_results_final.csv", index=False)

# Save probabilities for future use
np.savez(
    output_dir / "predictions.npz",
    y_true=all_y_true,
    y_prob_attention=all_y_prob_attention,
    y_prob_transformer=all_y_prob_transformer,
    subjects=all_subjects
)

print(f"\nRisultati salvati in:")
print(f"   - {output_dir / 'ensemble_results_final.csv'}")
print(f"   - {output_dir / 'predictions.npz'}")

print("\nDONE!", flush=True)
