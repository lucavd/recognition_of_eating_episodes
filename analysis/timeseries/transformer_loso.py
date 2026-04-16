"""
Transformer/Attention per Eating Detection - LOSO
Fase 8.4: Self-attention su time series
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

np.random.seed(2026)

print("=== FASE 8.4: TRANSFORMER/ATTENTION ===\n", flush=True)

# =============================================================================
# 1. LOAD DATA
# =============================================================================

print("1. Caricamento dati...", flush=True)
data_path = Path(__file__).parent.parent / "02_cnn1d" / "ts_data.csv"
df = pd.read_csv(data_path)

print(f"   Righe: {len(df)}", flush=True)

# Reshape to (samples, timesteps, channels)
channels = ['acc_x', 'acc_y', 'acc_z', 'pitch', 'roll', 'power']
n_timesteps = 25

X_3d = np.zeros((len(df), n_timesteps, len(channels)))
for i, ch in enumerate(channels):
    cols = [f"{ch}_{t}" for t in range(1, n_timesteps + 1)]
    X_3d[:, :, i] = df[cols].values

y = df['EU'].values
subjects = df['subject'].values

# Normalize
for i in range(len(channels)):
    mean = np.nanmean(X_3d[:, :, i])
    std = np.nanstd(X_3d[:, :, i])
    X_3d[:, :, i] = (X_3d[:, :, i] - mean) / (std + 1e-8)

X_3d = np.nan_to_num(X_3d, nan=0.0)
print(f"   X shape: {X_3d.shape}", flush=True)

# =============================================================================
# 2. SIMPLE ATTENTION MECHANISM
# =============================================================================

def simple_attention(X_3d):
    """
    Compute attention-weighted features.
    X_3d: (n_samples, n_timesteps, n_channels)
    Returns: (n_samples, n_features) with attention pooling
    """
    n_samples, n_timesteps, n_channels = X_3d.shape
    
    features = []
    
    for sample in X_3d:
        # sample: (n_timesteps, n_channels)
        
        # Energy-based attention weights
        energy = np.sum(sample**2, axis=1)  # (n_timesteps,)
        attention_weights = np.exp(energy) / np.sum(np.exp(energy))
        
        # Attention-weighted mean
        weighted_mean = np.sum(sample * attention_weights[:, np.newaxis], axis=0)
        
        # Max pooling
        max_pool = np.max(sample, axis=0)
        
        # Variance
        variance = np.var(sample, axis=0)
        
        # Temporal gradient
        gradient = np.mean(np.abs(np.diff(sample, axis=0)), axis=0)
        
        # Attention entropy (how focused is attention)
        attention_entropy = -np.sum(attention_weights * np.log(attention_weights + 1e-8))
        
        # Peak attention timestep
        peak_timestep = np.argmax(attention_weights) / n_timesteps
        
        # Combine all features
        feat = np.concatenate([
            weighted_mean,      # 6 features
            max_pool,           # 6 features
            variance,           # 6 features
            gradient,           # 6 features
            [attention_entropy, peak_timestep]  # 2 features
        ])
        features.append(feat)
    
    return np.array(features)

print("\n2. Extracting attention features...", flush=True)
X_attention = simple_attention(X_3d)
print(f"   Attention features shape: {X_attention.shape}", flush=True)

# =============================================================================
# 3. MULTI-HEAD ATTENTION FEATURES
# =============================================================================

def multi_head_attention(X_3d, n_heads=4):
    """
    Multi-head attention pooling.
    Different heads focus on different aspects of the time series.
    """
    n_samples, n_timesteps, n_channels = X_3d.shape
    
    features = []
    
    for sample in X_3d:
        head_features = []
        
        for head in range(n_heads):
            # Different attention mechanisms per head
            if head == 0:
                # Energy-based attention
                weights = np.sum(sample**2, axis=1)
            elif head == 1:
                # First channel (acc_x) attention
                weights = np.abs(sample[:, 0])
            elif head == 2:
                # Gradient-based attention
                grad = np.sum(np.abs(np.diff(sample, axis=0)), axis=1)
                weights = np.concatenate([[0], grad])
            else:
                # Uniform attention
                weights = np.ones(n_timesteps)
            
            # Softmax
            weights = np.exp(weights) / np.sum(np.exp(weights))
            
            # Weighted pooling
            pooled = np.sum(sample * weights[:, np.newaxis], axis=0)
            head_features.append(pooled)
        
        features.append(np.concatenate(head_features))
    
    return np.array(features)

print("\n3. Extracting multi-head attention features...", flush=True)
X_multihead = multi_head_attention(X_3d, n_heads=4)
print(f"   Multi-head features shape: {X_multihead.shape}", flush=True)

# Combine all features
X_combined = np.hstack([X_attention, X_multihead])
print(f"   Combined features shape: {X_combined.shape}", flush=True)

# Normalize and handle NaN
X_combined = np.nan_to_num(X_combined, nan=0.0, posinf=0.0, neginf=0.0)
scaler = StandardScaler()
X_combined = scaler.fit_transform(X_combined)
X_combined = np.nan_to_num(X_combined, nan=0.0)

# =============================================================================
# 4. LOSO VALIDATION
# =============================================================================

print("\n4. LOSO Validation...", flush=True)

unique_subjects = np.unique(subjects)
results = []

for subj in unique_subjects:
    test_mask = subjects == subj
    train_mask = ~test_mask
    
    X_train, y_train = X_combined[train_mask], y[train_mask]
    X_test, y_test = X_combined[test_mask], y[test_mask]
    
    if len(np.unique(y_test)) < 2:
        continue
    
    # Oversample eating class
    eating_idx = np.where(y_train == 1)[0]
    non_eating_idx = np.where(y_train == 0)[0]
    n_oversample = len(non_eating_idx) - len(eating_idx)
    if n_oversample > 0:
        oversample_idx = np.random.choice(eating_idx, size=n_oversample, replace=True)
        X_train_bal = np.vstack([X_train, X_train[oversample_idx]])
        y_train_bal = np.hstack([y_train, y_train[oversample_idx]])
    else:
        X_train_bal, y_train_bal = X_train, y_train
    
    # MLP classifier
    clf = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=100,
        random_state=2026,
        early_stopping=True
    )
    clf.fit(X_train_bal, y_train_bal)
    
    y_prob = clf.predict_proba(X_test)[:, 1]
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
    print(f"   Subject {subj}: sens={sens:.3f}, spec={spec:.3f}, auc={auc:.3f}", flush=True)

# =============================================================================
# 5. RESULTS
# =============================================================================

print("\n" + "="*50, flush=True)
print("=== RISULTATI ATTENTION + LOSO ===", flush=True)
print("="*50 + "\n", flush=True)

results_df = pd.DataFrame(results)
print("Per soggetto:", flush=True)
print(results_df.to_string(index=False), flush=True)

mean_sens = results_df['sens'].mean()
mean_spec = results_df['spec'].mean()
mean_auc = results_df['auc'].mean()
std_sens = results_df['sens'].std()

print(f"\n\nMETRICHE AGGREGATE:", flush=True)
print(f"Sensitivity: {mean_sens:.3f} (± {std_sens:.3f})", flush=True)
print(f"Specificity: {mean_spec:.3f}", flush=True)
print(f"AUC: {mean_auc:.3f}", flush=True)

print(f"\n\nCONFRONTO COMPLETO:", flush=True)
print(f"Baseline LOSO:  sens = 0.417", flush=True)
print(f"MiniRocket:     sens = 0.388 (-2.9%)", flush=True)
print(f"CNN-1D:         sens = 0.330 (-8.7%)", flush=True)
print(f"MLP Embedding:  sens = 0.434 (+1.7%)", flush=True)
print(f"Attention:      sens = {mean_sens:.3f} ({(mean_sens - 0.417) * 100:+.1f}%)", flush=True)

# Save
results_df.to_csv(Path(__file__).parent / "transformer_results.csv", index=False)
print(f"\nSaved to 04_transformer/transformer_results.csv", flush=True)
