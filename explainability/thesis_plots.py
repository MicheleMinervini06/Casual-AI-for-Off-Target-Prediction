"""Three thesis-ready figures:
  1) Model comparison: 5 SCM milestones + 2 baselines on AUPRC / AUROC / F1.
  2) Figure A: per-position effective weights |w_i| of the final model.
  3) Figure B: U_off stratified by mismatch distance, CHANGEseq vs GUIDEseq.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F

from models.deep.encoding import BiologicalMismatchEncoder
from models.deep.neural_scm import NeuralSCM


# ---------- paths ----------

RESULTS = Path("experiments/results")
OUT_DIR = Path("explainability/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCM_RUNS = [
    ("exp_03_neural_scm",                 "Exp03\nfirst Neural SCM"),
    ("Exp04_LinearBypass_HardPrior",      "Exp04\nLinear Bypass"),
    ("Exp06_TypedMLP_HardPrior",          "Exp06\nTyped MLP"),
    ("Exp12_Positional_Only",             "Exp12\nPositional"),
    ("Exp15_Positional_ExtendedOneCycle", "Exp15\nPositional + GC\n(final)"),
]
BASELINES_CSV = RESULTS / "exp_01_baseline" / "metrics.csv"
FINAL_MODEL   = RESULTS / "Exp15_Positional_ExtendedOneCycle" / "neural_scm.pt"

GUIDESEQ_BATCH  = Path("explainability/batch_results/guideseq_batch_results.csv")
CHANGESEQ_BATCH = Path("explainability/batch_results/changeseq_batch_results.csv")


# ============================================================================
# FIGURE 1 — Model comparison
# ============================================================================

def fig_comparison() -> Path:
    # Load SCM metrics
    rows = []
    for slug, label in SCM_RUNS:
        gs = json.load(open(RESULTS / slug / "metrics_guideseq.json"))
        rows.append({"label": label, "AUPRC": gs["auprc"], "AUROC": gs["auroc"], "F1": gs["f1"]})
    scm_df = pd.DataFrame(rows)

    # Baselines: cross_assay rows from CSV
    base = pd.read_csv(BASELINES_CSV)
    base = base[base["split"] == "cross_assay"][["model", "AUPRC", "AUROC", "F1"]].reset_index(drop=True)

    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    metrics  = ["AUPRC", "AUROC", "F1"]
    palette  = sns.color_palette("rocket_r", len(scm_df))   # darker = later exp
    palette[-1] = (0.85, 0.10, 0.10)                          # highlight final in red
    baseline_styles = {"XGBoost": "--", "CatBoost": ":"}
    baseline_colors = {"XGBoost": "#1f77b4", "CatBoost": "#2ca02c"}

    for ax, metric in zip(axes, metrics):
        x = np.arange(len(scm_df))
        bars = ax.bar(x, scm_df[metric].values, color=palette, edgecolor="black", linewidth=0.6)
        for bar, v in zip(bars, scm_df[metric].values):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.012, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=10)

        # baselines as horizontal reference lines
        for _, row in base.iterrows():
            ax.axhline(row[metric],
                       color=baseline_colors[row["model"]],
                       linestyle=baseline_styles[row["model"]],
                       linewidth=2,
                       label=f'{row["model"]} ({row[metric]:.3f})')

        ax.set_title(f"{metric} — GUIDEseq cross-assay")
        ax.set_xticks(x)
        ax.set_xticklabels(scm_df["label"].values, fontsize=9)
        ax.set_ylim(0, max(scm_df[metric].max(), base[metric].max()) * 1.18)
        ax.legend(loc="upper left", fontsize=9, frameon=True)

    fig.suptitle("Neural SCM evolution vs feature-engineered baselines",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = OUT_DIR / "thesis_fig1_comparison.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


# ============================================================================
# FIGURE 2 (A) — per-position effective weights of the final model
# ============================================================================

def fig_w_pos() -> Path:
    device = torch.device("cpu")
    state = torch.load(FINAL_MODEL, map_location=device)

    ctx_dim = state["context_net.0.weight"].shape[1] if "context_net.0.weight" in state else 0
    encoder = BiologicalMismatchEncoder()
    model = NeuralSCM(encoder=encoder, architecture="positional_mlp",
                      hidden_dim=8, context_dim=ctx_dim)
    model.load_state_dict(state)
    model.eval()

    # w_pos_eff = -softplus(w_pos)  →  effective penalty per position (non-positive)
    # plot |w_pos_eff| as "importance" magnitude
    w_pos_eff = -F.softplus(model.w_pos).detach().cpu().numpy()
    w_mag = np.abs(w_pos_eff)
    positions = np.arange(1, 21)

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(13, 5.5))

    # region shading: non-seed 1-8, seed 9-16, PAM-proximal 17-20
    ax.axvspan(0.5,  8.5,  color="gray",   alpha=0.10, zorder=0, label="Non-seed (1–8)")
    ax.axvspan(8.5,  16.5, color="gold",   alpha=0.18, zorder=0, label="Seed (9–16)")
    ax.axvspan(16.5, 20.5, color="tomato", alpha=0.18, zorder=0, label="PAM-proximal (17–20)")

    bars = ax.bar(positions, w_mag, color="#1f3b6e", edgecolor="black", linewidth=0.6, zorder=2)
    for bar, v in zip(bars, w_mag):
        ax.text(bar.get_x() + bar.get_width()/2, v + max(w_mag)*0.01, f"{v:.2f}",
                ha="center", va="bottom", fontsize=8.5)

    ax.set_xlabel("Spacer position (5' → 3', PAM at 3' end)")
    ax.set_ylabel("|w_i|  (effective per-position penalty magnitude)")
    ax.set_title("Figure A — Learned per-position weights recover the canonical seed pattern\n"
                 "(no positional supervision; only sign + seed-dominance priors)",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(positions)
    ax.set_xlim(0.5, 20.5)
    ax.legend(loc="upper left", fontsize=10, frameon=True)

    plt.tight_layout()
    path = OUT_DIR / "thesis_fig2_w_pos.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


# ============================================================================
# FIGURE 3 (B) — U_off stratified by mismatch distance
# ============================================================================

def fig_u_off_by_distance() -> Path:
    gs = pd.read_csv(GUIDESEQ_BATCH)
    cs = pd.read_csv(CHANGESEQ_BATCH)

    def agg(df):
        g = df.groupby("distance")["U_off"].agg(
            median="median",
            q25=lambda s: s.quantile(0.25),
            q75=lambda s: s.quantile(0.75),
            n="size",
        ).reset_index()
        return g

    gs_g = agg(gs)
    cs_g = agg(cs)

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(11, 6))

    ax.axhline(0, color="black", linewidth=0.8, linestyle=":", zorder=1)

    # CHANGEseq (in vitro) — blue
    ax.fill_between(cs_g["distance"], cs_g["q25"], cs_g["q75"],
                    color="#1f77b4", alpha=0.20, zorder=2)
    ax.plot(cs_g["distance"], cs_g["median"], marker="o", markersize=10,
            linewidth=2.5, color="#1f77b4",
            label=f"CHANGEseq (in vitro, n={len(cs)})", zorder=3)

    # GUIDEseq (in vivo) — orange
    ax.fill_between(gs_g["distance"], gs_g["q25"], gs_g["q75"],
                    color="#ff7f0e", alpha=0.20, zorder=2)
    ax.plot(gs_g["distance"], gs_g["median"], marker="s", markersize=10,
            linewidth=2.5, color="#ff7f0e",
            label=f"GUIDEseq (in vivo, n={len(gs)})", zorder=3)

    ax.set_xlabel("Number of mismatches (distance)")
    ax.set_ylabel("U_off  (median ± IQR, logit scale)")
    ax.set_title("Figure B — Exogenous noise U_off by assay regime\n"
                 "Monotonic growth in vitro vs flat profile in vivo  →  model internalised thermodynamics",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(sorted(set(cs_g["distance"]).union(gs_g["distance"])))
    ax.legend(loc="upper left", fontsize=11, frameon=True)

    plt.tight_layout()
    path = OUT_DIR / "thesis_fig3_u_off_by_distance.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


# ---------- main ----------

def main():
    print("Generating thesis-ready figures...\n")
    for name, fn in [
        ("Figure 1 — comparison", fig_comparison),
        ("Figure 2 (A) — per-position weights", fig_w_pos),
        ("Figure 3 (B) — U_off by distance", fig_u_off_by_distance),
    ]:
        try:
            out = fn()
            print(f"  [OK]  {name:40s} -> {out}")
        except Exception as e:
            print(f"  [ERR] {name:40s} -> {e}")
            raise


if __name__ == "__main__":
    main()
