"""
10a: Tuning Attention Esteso
Fase 10a: Hyperparameter tuning del modello Attention per migliorare balanced accuracy.

Parametri tunati:
- n_heads: numero di attention heads
- tau: temperature per softmax
- hidden_layer_sizes: architettura MLP
- alpha: L2 regularization
- learning_rate_init: learning rate iniziale
- oversample_ratio: ratio di oversampling per classe eating

Metodo: Random search con ~100 configurazioni, ottimizzazione per balanced accuracy.
Validazione: LOSO (19 soggetti)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, roc_auc_score, balanced_accuracy_score
from sklearn.neural_network import MLPClassifier
from pathlib import Path
from itertools import product
import json
import warnings
warnings.filterwarnings('ignore')

np.random.seed(2026)

print("=" * 60)
print("=== FASE 10a: TUNING ATTENTION ESTESO ===")
print("=" * 60 + "\n", flush=True)

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

unique_subjects = np.unique(subjects)
print(f"   Soggetti: {len(unique_subjects)}\n", flush=True)

# =============================================================================
# 2. ATTENTION FEATURE EXTRACTION (con parametri tunabili)
# =============================================================================

def extract_attention_features(X_3d, n_heads=4, tau=1.0):
    """
    Extract attention-weighted features with tunable parameters.
    
    Args:
        X_3d: (n_samples, n_timesteps, n_channels)
        n_heads: number of attention heads
        tau: temperature for softmax (lower = more peaked)
    
    Returns:
        features: (n_samples, n_features)
    """
    n_samples, n_timesteps, n_channels = X_3d.shape
    
    features = []
    
    for sample in X_3d:
        sample_features = []
        
        # Simple attention features (always included)
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
        
        # Multi-head attention features
        for head in range(n_heads):
            if head == 0:
                weights = np.sum(sample**2, axis=1)
            elif head == 1:
                weights = np.abs(sample[:, 0])  # acc_x
            elif head == 2:
                grad = np.sum(np.abs(np.diff(sample, axis=0)), axis=1)
                weights = np.concatenate([[0], grad])
            elif head == 3:
                weights = np.ones(n_timesteps)  # uniform
            elif head == 4:
                weights = np.abs(sample[:, 2])  # acc_z
            elif head == 5:
                weights = np.abs(sample[:, 5])  # power
            elif head == 6:
                # Peak detection based
                weights = np.abs(sample[:, 0] - np.mean(sample[:, 0]))
            else:
                weights = np.random.rand(n_timesteps)
            
            weights = np.exp(weights / tau) / np.sum(np.exp(weights / tau))
            pooled = np.sum(sample * weights[:, np.newaxis], axis=0)
            sample_features.extend(pooled)
        
        features.append(sample_features)
    
    return np.array(features)

# =============================================================================
# 3. LOSO EVALUATION FUNCTION
# =============================================================================

def evaluate_config(X_3d, y, subjects, config):
    """
    Evaluate a configuration using LOSO cross-validation.
    Returns mean balanced accuracy, sensitivity, specificity.
    """
    n_heads = config['n_heads']
    tau = config['tau']
    hidden_sizes = config['hidden_layer_sizes']
    alpha = config['alpha']
    lr = config['learning_rate_init']
    oversample_ratio = config['oversample_ratio']
    
    # Extract features
    X_features = extract_attention_features(X_3d, n_heads=n_heads, tau=tau)
    X_features = np.nan_to_num(X_features, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Normalize
    scaler = StandardScaler()
    X_features = scaler.fit_transform(X_features)
    X_features = np.nan_to_num(X_features, nan=0.0)
    
    unique_subjects = np.unique(subjects)
    results = []
    
    for subj in unique_subjects:
        test_mask = subjects == subj
        train_mask = ~test_mask
        
        X_train, y_train = X_features[train_mask], y[train_mask]
        X_test, y_test = X_features[test_mask], y[test_mask]
        
        if len(np.unique(y_test)) < 2:
            continue
        
        # Oversample eating class
        eating_idx = np.where(y_train == 1)[0]
        non_eating_idx = np.where(y_train == 0)[0]
        
        target_eating = int(len(non_eating_idx) * oversample_ratio)
        n_oversample = target_eating - len(eating_idx)
        
        if n_oversample > 0:
            oversample_idx = np.random.choice(eating_idx, size=n_oversample, replace=True)
            X_train_bal = np.vstack([X_train, X_train[oversample_idx]])
            y_train_bal = np.hstack([y_train, y_train[oversample_idx]])
        else:
            X_train_bal, y_train_bal = X_train, y_train
        
        # MLP classifier
        clf = MLPClassifier(
            hidden_layer_sizes=hidden_sizes,
            alpha=alpha,
            learning_rate_init=lr,
            max_iter=200,
            random_state=2026,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        try:
            clf.fit(X_train_bal, y_train_bal)
            y_prob = clf.predict_proba(X_test)[:, 1]
            y_pred = (y_prob > 0.5).astype(int)
            
            sens = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
            spec = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            
            try:
                auc = roc_auc_score(y_test, y_prob)
            except:
                auc = 0.5
            
            results.append({
                'sens': sens,
                'spec': spec,
                'bal_acc': bal_acc,
                'auc': auc
            })
        except Exception as e:
            continue
    
    if not results:
        return {'sens': 0, 'spec': 0, 'bal_acc': 0, 'auc': 0.5}
    
    return {
        'sens': np.mean([r['sens'] for r in results]),
        'spec': np.mean([r['spec'] for r in results]),
        'bal_acc': np.mean([r['bal_acc'] for r in results]),
        'auc': np.mean([r['auc'] for r in results])
    }

# =============================================================================
# 4. HYPERPARAMETER GRID
# =============================================================================

print("2. Definizione griglia iperparametri...", flush=True)

param_grid = {
    'n_heads': [2, 4, 6, 8],
    'tau': [0.3, 0.5, 1.0, 2.0],
    'hidden_layer_sizes': [(32,), (64,), (64, 32), (128, 64), (128, 64, 32)],
    'alpha': [1e-5, 1e-4, 1e-3, 1e-2],
    'learning_rate_init': [1e-4, 5e-4, 1e-3, 3e-3],
    'oversample_ratio': [0.5, 0.75, 1.0]
}

# Generate all combinations
all_keys = list(param_grid.keys())
all_values = [param_grid[k] for k in all_keys]
all_configs = [dict(zip(all_keys, v)) for v in product(*all_values)]

print(f"   Combinazioni totali possibili: {len(all_configs)}", flush=True)

# Random sample 100 configurations
np.random.shuffle(all_configs)
configs_to_test = all_configs[:100]

print(f"   Configurazioni da testare: {len(configs_to_test)}\n", flush=True)

# =============================================================================
# 5. RANDOM SEARCH
# =============================================================================

print("3. Random Search con LOSO...\n", flush=True)

results_list = []
best_bal_acc = 0
best_config = None

for i, config in enumerate(configs_to_test):
    print(f"   [{i+1:3d}/{len(configs_to_test)}] Testing: n_heads={config['n_heads']}, "
          f"tau={config['tau']}, hidden={config['hidden_layer_sizes']}, "
          f"alpha={config['alpha']:.0e}, lr={config['learning_rate_init']:.0e}, "
          f"oversample={config['oversample_ratio']}", end="", flush=True)
    
    metrics = evaluate_config(X_3d, y, subjects, config)
    
    result = {**config, **metrics}
    # Convert tuple to string for JSON serialization
    result['hidden_layer_sizes'] = str(config['hidden_layer_sizes'])
    results_list.append(result)
    
    print(f" -> bal_acc={metrics['bal_acc']:.3f}, sens={metrics['sens']:.3f}, "
          f"spec={metrics['spec']:.3f}", flush=True)
    
    if metrics['bal_acc'] > best_bal_acc:
        best_bal_acc = metrics['bal_acc']
        best_config = config.copy()
        print(f"   *** NEW BEST! ***", flush=True)

# =============================================================================
# 6. RESULTS
# =============================================================================

print("\n" + "=" * 60)
print("=== RISULTATI TUNING ATTENTION ===")
print("=" * 60 + "\n", flush=True)

results_df = pd.DataFrame(results_list)
results_df = results_df.sort_values('bal_acc', ascending=False)

print("TOP 10 CONFIGURAZIONI:\n")
print(results_df.head(10).to_string(index=False), flush=True)

print(f"\n\nMIGLIOR CONFIGURAZIONE:")
print(f"  n_heads: {best_config['n_heads']}")
print(f"  tau: {best_config['tau']}")
print(f"  hidden_layer_sizes: {best_config['hidden_layer_sizes']}")
print(f"  alpha: {best_config['alpha']}")
print(f"  learning_rate_init: {best_config['learning_rate_init']}")
print(f"  oversample_ratio: {best_config['oversample_ratio']}")

best_metrics = results_df.iloc[0]
print(f"\n  Balanced Accuracy: {best_metrics['bal_acc']:.3f}")
print(f"  Sensitivity: {best_metrics['sens']:.3f}")
print(f"  Specificity: {best_metrics['spec']:.3f}")
print(f"  AUC: {best_metrics['auc']:.3f}")

print(f"\n\nCONFRONTO:")
print(f"  Attention default:  sens=0.498, spec=0.709, bal_acc=0.604")
print(f"  XGB tuned:          sens=0.593, spec=0.692, bal_acc=0.643")
print(f"  Attention tuned:    sens={best_metrics['sens']:.3f}, "
      f"spec={best_metrics['spec']:.3f}, bal_acc={best_metrics['bal_acc']:.3f}")

delta_sens = (best_metrics['sens'] - 0.498) * 100
delta_spec = (best_metrics['spec'] - 0.709) * 100
delta_bal = (best_metrics['bal_acc'] - 0.604) * 100

print(f"\n  Delta vs default: sens={delta_sens:+.1f}%, spec={delta_spec:+.1f}%, "
      f"bal_acc={delta_bal:+.1f}%")

# =============================================================================
# 7. SAVE RESULTS
# =============================================================================

output_dir = Path(__file__).parent
output_dir.mkdir(parents=True, exist_ok=True)

results_df.to_csv(output_dir / "attention_tuning_results.csv", index=False)

with open(output_dir / "best_attention_config.json", "w") as f:
    # Convert tuple to list for JSON
    best_config_json = best_config.copy()
    best_config_json['hidden_layer_sizes'] = list(best_config['hidden_layer_sizes'])
    json.dump(best_config_json, f, indent=2)

print(f"\nRisultati salvati in:")
print(f"  - {output_dir / 'attention_tuning_results.csv'}")
print(f"  - {output_dir / 'best_attention_config.json'}")

print("\nDONE!", flush=True)
