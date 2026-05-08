import os
import pickle
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Import dal tuo ecosistema
from evaluation.metrics import evaluate_model, find_optimal_threshold, results_to_dataframe
from models.deep.encoding import BiologicalMismatchEncoder
from models.deep.neural_scm import NeuralSCM

def load_tree_model(path: Path):
    """Carica un modello XGBoost/CatBoost salvato con pickle."""
    with open(path, "rb") as f:
        return pickle.load(f)

def predict_scm_in_batches(model, guides: list[str], offtargets: list[str], context_features: np.ndarray | None = None, batch_size: int = 128, device: torch.device = torch.device("cpu")) -> np.ndarray:
    """Fa inferenza con l'SCM processando a blocchi per evitare Out-Of-Memory (OOM)."""
    model.eval()
    all_logits = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(guides), batch_size), desc="Inferenza SCM"):
            batch_g = guides[i : i + batch_size]
            batch_off = offtargets[i : i + batch_size]
            
            # Forward pass
            if context_features is not None:
                # Estrai il batch di context features
                context_batch = torch.tensor(context_features[i : i + batch_size], dtype=torch.float32).to(device)
                out = model(batch_g, batch_off, context_features=context_batch)
            else:
                out = model(batch_g, batch_off)
            
            # Assicuriamoci di estendere la lista con valori scalari (1D).
            # Preferiamo usare l'`activity_probability` (già compresa di pam_gate),
            # se presente; altrimenti ricostruiamo da `logit`.
            if 'activity_probability' in out:
                all_logits.extend(out['activity_probability'].cpu().numpy().ravel().tolist())
            else:
                all_logits.extend(out['logit'].cpu().numpy().ravel().tolist())
            
    # Se abbiamo già raccolto probabilità (activity_probability), non applicare sigmoid.
    all_vals = np.asarray(all_logits, dtype=float)
    # Heuristica: se i valori sono già in [0,1] li consideriamo probabilità
    if np.all((all_vals >= -1e-6) & (all_vals <= 1.0 + 1e-6)):
        probs = np.clip(all_vals, 0.0, 1.0)
    else:
        probs = 1 / (1 + np.exp(-all_vals))
    return probs

def plot_curves(results: list, output_dir: Path):
    """Genera e salva i grafici ROC e PRC da inserire nella tesi."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax_roc, ax_prc) = plt.subplots(1, 2, figsize=(16, 7))

    colors = {"XGBoost": "#E63946", "CatBoost": "#F4A261", "Neural SCM (Run 10)": "#1D3557", "Neural SCM (Exp03 + Context)": "#457B9D"}
    linestyles = {"XGBoost": "--", "CatBoost": "-.", "Neural SCM (Run 10)": "-", "Neural SCM (Exp03 + Context)": ":"}

    for res in results:
        name = res.model_name
        color = colors.get(name, "#000000")
        ls = linestyles.get(name, "-")

        # ROC Curve
        ax_roc.plot(res.roc_fpr, res.roc_tpr, color=color, linestyle=ls, linewidth=2.5, 
                    label=f'{name} (AUC = {res.auroc:.3f})')
        
        # PRC Curve
        ax_prc.plot(res.pr_recall, res.pr_precision, color=color, linestyle=ls, linewidth=2.5, 
                    label=f'{name} (AUPRC = {res.auprc:.3f})')

    # Formatting ROC
    ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax_roc.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax_roc.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax_roc.set_title('Receiver Operating Characteristic (ROC)', fontsize=14, fontweight='bold')
    ax_roc.legend(loc="lower right", fontsize=11)
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])

    # Formatting PRC
    baseline_prc = results[0].n_pos_true / results[0].n_total if results else 0.5
    ax_prc.axhline(y=baseline_prc, color='k', linestyle='--', alpha=0.5, label=f'Baseline ({baseline_prc:.2f})')
    ax_prc.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax_prc.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax_prc.set_title('Precision-Recall Curve (PRC)', fontsize=14, fontweight='bold')
    ax_prc.legend(loc="upper right", fontsize=11)
    ax_prc.set_xlim([0.0, 1.0])
    ax_prc.set_ylim([0.0, 1.05])

    plt.tight_layout()
    plot_path = output_dir / "benchmarking_curves.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n[+] Grafici HD salvati in: {plot_path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking Arena inizializzata su: {device}")

    # ==========================================
    # 1. SETUP DEI PERCORSI (Adattali al tuo workspace)
    # ==========================================
    data_dir = Path("data/processed/splits")
    val_path = data_dir / "val.parquet"
    test_path = data_dir / "test.parquet"
    guideseq_path = Path("data/processed/features/guideseq_features.parquet")
    
    results_dir = Path("experiments/results/benchmark_final")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Path ai modelli addestrati
    xgb_path = Path("experiments/results/exp_01_baseline/xgboost_model.pkl")
    cat_path = Path("experiments/results/exp_01_baseline/catboost_model.pkl")
    scm_path = Path("experiments/results/Exp06_TypedMLP_HardPrior/neural_scm.pt")
    scm_ctx_path = Path("experiments/results/Exp11_HybridCausal_TypedMLP/neural_scm.pt")

    # Feature usate da XGBoost (prendile dal tuo config.yaml)
    feature_cols = [
        "mean_energy_penalty", "total_energy_penalty", "node_B_proximal", 
        "node_C_seed_extension", "node_D_non_seed", "pam_score", # ... aggiungi le tue esatte
    ]

    # ==========================================
    # 2. CARICAMENTO DATI (GUIDE-seq per cross-assay evaluation)
    # ==========================================
    if guideseq_path.exists():
        test_df = pd.read_parquet(guideseq_path)
        print(f"Dati caricati (GUIDE-seq cross-assay): {len(test_df)} righe")
    else:
        print(f"[WARN] GUIDE-seq non trovato in {guideseq_path}; uso test.parquet come fallback")
        test_df = pd.read_parquet(test_path)
        print(f"Dati caricati (fallback CHANGE-seq test): {len(test_df)} righe")
    
    # Carica val per threshold optimization (usa sempre CHANGE-seq val)
    val_df = pd.read_parquet(val_path)
    
    y_val = np.asarray(val_df["label"].values, dtype=int)
    y_test = np.asarray(test_df["label"].values, dtype=int)

    print(f"Valutazione su GUIDE-seq (cross-assay) -> Test: {len(test_df)} righe")
    print(f"Threshold optimization su CHANGE-seq val -> Val: {len(val_df)} righe\n")

    all_results = []

    # ==========================================
    # 3. VALUTAZIONE XGBOOST
    # ==========================================
    if xgb_path.exists():
        print("\n--- Valutazione XGBoost ---")
        xgb_model = load_tree_model(xgb_path)
        # Se il wrapper salvato contiene i nomi delle feature, usiamoli per allineare le colonne
        model_feature_names = None
        if hasattr(xgb_model, "feature_names") and xgb_model.feature_names:
            model_feature_names = xgb_model.feature_names
        else:
            # Tentativo con attributo interno del modello xgboost
            try:
                booster = getattr(xgb_model, "model", None)
                if booster is not None:
                    b = getattr(booster, "get_booster", None)
                    if callable(b):
                        model_feature_names = getattr(b(), 'feature_names', None)
            except Exception:
                model_feature_names = None

        if model_feature_names is None:
            # Fallback: usare le feature hardcoded e avvisare
            print("[WARN] Non ho trovato i nomi delle feature nel modello XGBoost salvato; usando 'feature_cols' dallo script.")
            used_features = feature_cols
        else:
            print(f"[INFO] Allineo le colonne usando le feature salvate nel modello (n={len(model_feature_names)})")
            used_features = model_feature_names

        # Estrazione feature tabulari
        X_val_xgb = val_df[used_features].to_numpy()
        X_test_xgb = test_df[used_features].to_numpy()
        
        # Probabilità
        y_val_prob_xgb = xgb_model.predict_proba(X_val_xgb)[:, 1]
        y_test_prob_xgb = xgb_model.predict_proba(X_test_xgb)[:, 1]
        
        # Ottimizzazione Threshold e Valutazione
        thr_xgb = find_optimal_threshold(y_val, y_val_prob_xgb, metric="f1")
        res_xgb = evaluate_model("XGBoost", y_test, y_test_prob_xgb, split="test", threshold=thr_xgb, store_curves=True)
        all_results.append(res_xgb)

    # ==========================================
    # 3b. VALUTAZIONE CATBOOST
    # ==========================================
    if cat_path.exists():
        print("\n--- Valutazione CatBoost ---")
        cat_model = load_tree_model(cat_path)

        # Allineamento feature come per XGBoost
        model_feature_names = None
        if hasattr(cat_model, "feature_names") and cat_model.feature_names:
            model_feature_names = cat_model.feature_names
        else:
            try:
                booster = getattr(cat_model, "model", None)
                if booster is not None:
                    b = getattr(booster, "get_booster", None)
                    if callable(b):
                        model_feature_names = getattr(b(), 'feature_names', None)
            except Exception:
                model_feature_names = None

        if model_feature_names is None:
            print("[WARN] Non ho trovato i nomi delle feature nel modello CatBoost salvato; usando 'feature_cols' dallo script.")
            used_features = feature_cols
        else:
            print(f"[INFO] Allineo le colonne usando le feature salvate nel modello CatBoost (n={len(model_feature_names)})")
            used_features = model_feature_names

        X_val_cat = val_df[used_features].to_numpy()
        X_test_cat = test_df[used_features].to_numpy()

        # Probabilità
        y_val_prob_cat = cat_model.predict_proba(X_val_cat)[:, 1]
        y_test_prob_cat = cat_model.predict_proba(X_test_cat)[:, 1]

        # Ottimizzazione Threshold e Valutazione
        thr_cat = find_optimal_threshold(y_val, y_val_prob_cat, metric="f1")
        res_cat = evaluate_model("CatBoost", y_test, y_test_prob_cat, split="test", threshold=thr_cat, store_curves=True)
        all_results.append(res_cat)

    # ==========================================
    # 4. VALUTAZIONE NEURAL SCM (Run 10)
    # ==========================================
    if scm_path.exists():
        print("\n--- Valutazione Neural SCM ---")
        encoder = BiologicalMismatchEncoder()
        scm_model = NeuralSCM(encoder=encoder, architecture="typed_mlp", hidden_dim=8)
        scm_model.load_state_dict(torch.load(scm_path, map_location=device))
        scm_model.to(device)
        
        # Estrazione stringhe
        guides_val = val_df["sgRNA_seq"].tolist()
        offs_val = val_df["off_seq"].tolist()
        guides_test = test_df["sgRNA_seq"].tolist()
        offs_test = test_df["off_seq"].tolist()
        
        # Probabilità a blocchi (senza context features)
        y_val_prob_scm = predict_scm_in_batches(scm_model, guides_val, offs_val, context_features=None, device=device)
        y_test_prob_scm = predict_scm_in_batches(scm_model, guides_test, offs_test, context_features=None, device=device)
        
        # Ottimizzazione Threshold e Valutazione
        thr_scm = find_optimal_threshold(y_val, y_val_prob_scm, metric="f1")
        res_scm = evaluate_model("Neural SCM (Run 10)", y_test, y_test_prob_scm, split="test", threshold=thr_scm, store_curves=True)
        all_results.append(res_scm)

    # ==========================================
    # 4b. VALUTAZIONE NEURAL SCM (Exp03 + Context Cols)
    # ==========================================
    # Versione con context_cols (gc_sgRNA, gc_offtarget, concept_gc_delta)
    if scm_ctx_path.exists():
        print("\n--- Valutazione Neural SCM (Exp03 + Context) ---")
        encoder_ctx = BiologicalMismatchEncoder()
        scm_model_ctx = NeuralSCM(encoder=encoder_ctx, architecture="typed_mlp", hidden_dim=8)
        scm_model_ctx.load_state_dict(torch.load(scm_ctx_path, map_location=device))
        scm_model_ctx.to(device)
        
        # Estrazione stringhe
        guides_val = val_df["sgRNA_seq"].tolist()
        offs_val = val_df["off_seq"].tolist()
        guides_test = test_df["sgRNA_seq"].tolist()
        offs_test = test_df["off_seq"].tolist()
        
        # Context features (variabili esogene)
        context_cols = ["gc_sgRNA", "gc_offtarget", "concept_gc_delta"]
        context_val = val_df[context_cols].to_numpy(dtype=np.float32) if all(c in val_df.columns for c in context_cols) else None
        context_test = test_df[context_cols].to_numpy(dtype=np.float32) if all(c in test_df.columns for c in context_cols) else None
        
        # Probabilità a blocchi (con context se disponibile)
        if context_val is not None:
            y_val_prob_scm_ctx = predict_scm_in_batches(scm_model_ctx, guides_val, offs_val, context_features=context_val, device=device)
        else:
            print("[WARN] Context features non disponibili nel val DataFrame; usando solo sequenze.")
            y_val_prob_scm_ctx = predict_scm_in_batches(scm_model_ctx, guides_val, offs_val, context_features=None, device=device)
        
        if context_test is not None:
            y_test_prob_scm_ctx = predict_scm_in_batches(scm_model_ctx, guides_test, offs_test, context_features=context_test, device=device)
        else:
            print("[WARN] Context features non disponibili nel test DataFrame; usando solo sequenze.")
            y_test_prob_scm_ctx = predict_scm_in_batches(scm_model_ctx, guides_test, offs_test, context_features=None, device=device)
        
        # Ottimizzazione Threshold e Valutazione
        thr_scm_ctx = find_optimal_threshold(y_val, y_val_prob_scm_ctx, metric="f1")
        res_scm_ctx = evaluate_model("Neural SCM (Exp03 + Context)", y_test, y_test_prob_scm_ctx, split="test", threshold=thr_scm_ctx, store_curves=True)
        all_results.append(res_scm_ctx)

    # ==========================================
    # 5. ESPORTAZIONE RISULTATI
    # ==========================================
    if all_results:
        # Tabella riassuntiva
        metrics_df = results_to_dataframe(all_results)
        metrics_df.to_csv(results_dir / "benchmarking_metrics.csv", index=False)
        print("\nTabella Riassuntiva:")
        print(metrics_df.to_string(index=False))
        
        # Grafici ROC e PRC
        plot_curves(all_results, results_dir)

if __name__ == "__main__":
    main()