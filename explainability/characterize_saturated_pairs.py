"""Caratterizzazione delle feature delle coppie saturate (CHANGE-seq cell-free).

Test F22.1: identifica gli elementi comuni alle coppie che presentano il fenomeno
di saturazione (off_reads >= on_reads), per capire se sono:
  - errori di misurazione casuali
  - sistematicamente diverse dal resto (es. specifiche regioni del genoma,
    pattern di mismatch, GC bias, particolari guide)

Per ogni feature candidata, calcola:
  - Distribuzione stratificata saturated vs not-saturated
  - Mann-Whitney U test (più robusto del t-test) + Cohen's d
  - Effect size visualizzato in barplot

Output: plot a multi-panel + JSON con metriche.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def gc_fraction(seq: str) -> float:
    if not isinstance(seq, str) or len(seq) == 0:
        return np.nan
    return sum(1 for c in seq.upper() if c in "GC") / len(seq)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d (pooled std). Effect size standardizzato."""
    a = np.asarray(a)
    b = np.asarray(b)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    s_pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    if s_pooled < 1e-12:
        return 0.0
    return float((b.mean() - a.mean()) / s_pooled)


def mismatch_positions_mask(sgrna: str, off_target: str) -> np.ndarray:
    """Restituisce array binario [20] con 1 dove c'e' mismatch nello spacer (pos 0-19)."""
    sg = sgrna[:20].upper().ljust(20, "N")
    ot = off_target[:20].upper().ljust(20, "N")
    return np.array([1 if sg[i] != ot[i] else 0 for i in range(20)], dtype=np.int8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv", type=Path,
        default=Path("explainability/batch_results/changeseq_batch_results_shift+2.73.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("explainability/batch_results"),
    )
    args = parser.parse_args()

    print(f"Loading {args.csv}")
    df = pd.read_csv(args.csv)
    print(f"  rows: {len(df)}")

    df["is_saturated"] = (df["off_reads"] >= df["on_reads"]).astype(bool)

    # === Compute derived features ===
    print("Computing derived features...")
    df["gc_sgrna"] = df["sgRNA"].apply(gc_fraction)
    df["gc_offtarget_spacer"] = df["off_target"].apply(lambda x: gc_fraction(str(x)[:20]))
    df["gc_pam"] = df["off_target"].apply(lambda x: gc_fraction(str(x)[20:23]))
    df["gc_delta"] = df["gc_sgrna"] - df["gc_offtarget_spacer"]
    df["pam_seq"] = df["off_target"].apply(lambda x: str(x)[20:23].upper() if isinstance(x, str) and len(x) >= 23 else "NNN")
    df["pam_is_ngg"] = df["pam_seq"].apply(lambda p: p.endswith("GG"))

    # Per-position mismatch flags
    print("  computing per-position mismatch flags...")
    mm_arrays = df.apply(lambda r: mismatch_positions_mask(str(r["sgRNA"]), str(r["off_target"])), axis=1)
    mm_matrix = np.stack(mm_arrays.values)  # shape [N, 20]
    for i in range(20):
        df[f"mm_pos_{i:02d}"] = mm_matrix[:, i]

    # Region-specific mismatch counts
    df["mm_nonseed_count"] = mm_matrix[:, 0:8].sum(axis=1)   # PAM-distal
    df["mm_seed_count"]    = mm_matrix[:, 8:16].sum(axis=1)
    df["mm_prox_count"]    = mm_matrix[:, 16:20].sum(axis=1) # PAM-proximal

    # === Feature comparison ===
    sat = df[df["is_saturated"]]
    nsat = df[~df["is_saturated"]]
    n_sat = len(sat)
    n_nsat = len(nsat)

    print(f"\n=== POPULATIONS ===")
    print(f"  saturated:     n={n_sat:>6d}  ({100*n_sat/len(df):.1f}%)")
    print(f"  not saturated: n={n_nsat:>6d}  ({100*n_nsat/len(df):.1f}%)")

    continuous_features = [
        ("distance",              "Mismatch count (total)"),
        ("mm_nonseed_count",      "Mismatches in non-seed (pos 0-7)"),
        ("mm_seed_count",         "Mismatches in seed (pos 8-15)"),
        ("mm_prox_count",         "Mismatches in PAM-proximal (pos 16-19)"),
        ("gc_sgrna",              "sgRNA GC content"),
        ("gc_offtarget_spacer",   "off-target spacer GC content"),
        ("gc_pam",                "off-target PAM GC content"),
        ("gc_delta",              "GC(sgRNA) - GC(off-target)"),
        ("pam_off_f",             "Model pam_gate (sigmoid output)"),
    ]

    print(f"\n=== CONTINUOUS FEATURE COMPARISON (saturated vs not) ===")
    print(f"{'Feature':<35s}  {'mean_nsat':>10s}  {'mean_sat':>10s}  {'cohen_d':>8s}  {'U_stat':>12s}  {'p_value':>10s}")
    print("-" * 95)

    continuous_results = []
    for key, label in continuous_features:
        a = nsat[key].dropna().values
        b = sat[key].dropna().values
        if len(a) < 2 or len(b) < 2:
            continue
        d = cohens_d(a, b)
        u_stat, p_val = stats.mannwhitneyu(a, b, alternative="two-sided")
        print(f"{label:<35s}  {a.mean():>10.3f}  {b.mean():>10.3f}  {d:>+8.3f}  {u_stat:>12.0f}  {p_val:>10.2e}")
        continuous_results.append({
            "feature": key,
            "label": label,
            "mean_not_saturated": float(a.mean()),
            "mean_saturated": float(b.mean()),
            "std_not_saturated": float(a.std()),
            "std_saturated": float(b.std()),
            "median_not_saturated": float(np.median(a)),
            "median_saturated": float(np.median(b)),
            "cohens_d": float(d),
            "mannwhitneyu_stat": float(u_stat),
            "p_value": float(p_val),
        })

    # === PAM categorical analysis ===
    print(f"\n=== PAM SEQUENCE BREAKDOWN ===")
    pam_table = pd.crosstab(df["pam_seq"], df["is_saturated"], margins=True, margins_name="Total")
    pam_table.columns = ["NotSat", "Saturated", "Total"]
    pam_table["P(Sat|PAM)"] = pam_table["Saturated"] / pam_table["Total"]
    pam_table = pam_table.sort_values("Total", ascending=False).head(15)
    print(pam_table.to_string())

    # === Per-guide saturation rate ===
    print(f"\n=== PER-GUIDE SATURATION RATE (top 15 most-saturated guides) ===")
    per_guide = df.groupby("name").agg(
        n_pairs=("is_saturated", "size"),
        n_saturated=("is_saturated", "sum"),
    )
    per_guide["sat_rate"] = per_guide["n_saturated"] / per_guide["n_pairs"]
    print(per_guide.sort_values("sat_rate", ascending=False).head(15).to_string())

    print(f"\nMean per-guide saturation rate: {per_guide['sat_rate'].mean():.3f}")
    print(f"Median per-guide saturation rate: {per_guide['sat_rate'].median():.3f}")
    print(f"Number of guides with >50% saturation: {(per_guide['sat_rate'] > 0.5).sum()}/{len(per_guide)}")
    print(f"Number of guides with <10% saturation: {(per_guide['sat_rate'] < 0.1).sum()}/{len(per_guide)}")

    # === Per-position mismatch frequency ===
    print(f"\n=== PER-POSITION MISMATCH FREQUENCY ===")
    pos_freq_nsat = nsat[[f"mm_pos_{i:02d}" for i in range(20)]].mean().values
    pos_freq_sat = sat[[f"mm_pos_{i:02d}" for i in range(20)]].mean().values
    print(f"{'pos':<4s}  {'P(mm|nsat)':>10s}  {'P(mm|sat)':>10s}  {'delta':>8s}")
    for i in range(20):
        delta = pos_freq_sat[i] - pos_freq_nsat[i]
        print(f"{i:<4d}  {pos_freq_nsat[i]:>10.3f}  {pos_freq_sat[i]:>10.3f}  {delta:>+8.3f}")

    # === Plot ===
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(3, 3, figsize=(17, 13))

    # Panel: distance
    ax = axes[0, 0]
    bins = np.arange(0, max(df["distance"].max(), 7) + 1.5) - 0.5
    ax.hist(nsat["distance"].values, bins=bins, color="steelblue", alpha=0.65, label="Not saturated", density=True)
    ax.hist(sat["distance"].values,  bins=bins, color="crimson",   alpha=0.65, label="Saturated", density=True)
    ax.set_xlabel("Number of mismatches (distance)")
    ax.set_ylabel("Density")
    ax.set_title("Total mismatch count")
    ax.legend()

    # Panel: seed mismatches
    ax = axes[0, 1]
    bins = np.arange(0, 9) - 0.5
    ax.hist(nsat["mm_seed_count"].values, bins=bins, color="steelblue", alpha=0.65, label="Not saturated", density=True)
    ax.hist(sat["mm_seed_count"].values,  bins=bins, color="crimson",   alpha=0.65, label="Saturated", density=True)
    ax.set_xlabel("Mismatches in seed (pos 8-15)")
    ax.set_ylabel("Density")
    ax.set_title("Seed mismatches")
    ax.legend()

    # Panel: PAM-proximal mismatches
    ax = axes[0, 2]
    bins = np.arange(0, 5) - 0.5
    ax.hist(nsat["mm_prox_count"].values, bins=bins, color="steelblue", alpha=0.65, label="Not saturated", density=True)
    ax.hist(sat["mm_prox_count"].values,  bins=bins, color="crimson",   alpha=0.65, label="Saturated", density=True)
    ax.set_xlabel("Mismatches in PAM-proximal (pos 16-19)")
    ax.set_ylabel("Density")
    ax.set_title("PAM-proximal mismatches")
    ax.legend()

    # Panel: GC sgRNA
    ax = axes[1, 0]
    bins = np.linspace(0, 1, 40)
    ax.hist(nsat["gc_sgrna"].values, bins=bins, color="steelblue", alpha=0.65, label="Not saturated", density=True)
    ax.hist(sat["gc_sgrna"].values,  bins=bins, color="crimson",   alpha=0.65, label="Saturated", density=True)
    ax.set_xlabel("sgRNA GC content")
    ax.set_ylabel("Density")
    ax.set_title("sgRNA GC")
    ax.legend()

    # Panel: GC delta
    ax = axes[1, 1]
    bins = np.linspace(-0.5, 0.5, 40)
    ax.hist(nsat["gc_delta"].values, bins=bins, color="steelblue", alpha=0.65, label="Not saturated", density=True)
    ax.hist(sat["gc_delta"].values,  bins=bins, color="crimson",   alpha=0.65, label="Saturated", density=True)
    ax.set_xlabel("GC(sgRNA) - GC(off-target spacer)")
    ax.set_ylabel("Density")
    ax.set_title("GC delta")
    ax.legend()

    # Panel: pam_off_f
    ax = axes[1, 2]
    bins = np.linspace(df["pam_off_f"].min(), df["pam_off_f"].max(), 40)
    ax.hist(nsat["pam_off_f"].values, bins=bins, color="steelblue", alpha=0.65, label="Not saturated", density=True)
    ax.hist(sat["pam_off_f"].values,  bins=bins, color="crimson",   alpha=0.65, label="Saturated", density=True)
    ax.set_xlabel("Model pam_gate (sigmoid)")
    ax.set_ylabel("Density")
    ax.set_title("Model PAM gate output")
    ax.legend()

    # Panel: per-position mismatch frequency
    ax = axes[2, 0]
    x = np.arange(20)
    width = 0.4
    ax.bar(x - width/2, pos_freq_nsat, width, color="steelblue", alpha=0.75, label="Not saturated")
    ax.bar(x + width/2, pos_freq_sat,  width, color="crimson",   alpha=0.75, label="Saturated")
    ax.set_xlabel("Position (0=PAM-distal, 19=PAM-proximal)")
    ax.set_ylabel("P(mismatch at this position)")
    ax.set_title("Per-position mismatch frequency")
    ax.legend()
    ax.axvspan(-0.5, 7.5, alpha=0.05, color="green")
    ax.axvspan(7.5, 15.5, alpha=0.05, color="orange")
    ax.axvspan(15.5, 19.5, alpha=0.05, color="red")

    # Panel: per-guide saturation rate distribution
    ax = axes[2, 1]
    ax.hist(per_guide["sat_rate"].values, bins=30, color="purple", alpha=0.7, edgecolor="white")
    ax.axvline(per_guide["sat_rate"].mean(), color="red", linestyle="--",
               label=f"mean={per_guide['sat_rate'].mean():.2f}")
    ax.axvline(per_guide["sat_rate"].median(), color="black", linestyle=":",
               label=f"median={per_guide['sat_rate'].median():.2f}")
    ax.set_xlabel("Per-guide saturation rate")
    ax.set_ylabel("Number of guides")
    ax.set_title(f"Saturation rate per guide (n={len(per_guide)})")
    ax.legend()

    # Panel: Cohen's d barplot
    ax = axes[2, 2]
    sorted_results = sorted(continuous_results, key=lambda r: abs(r["cohens_d"]), reverse=True)
    labels = [r["label"] for r in sorted_results]
    ds = [r["cohens_d"] for r in sorted_results]
    colors = ["crimson" if d > 0 else "steelblue" for d in ds]
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, ds, color=colors, alpha=0.75)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(0.2, color="gray", linestyle=":", linewidth=0.6, label="small (0.2)")
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.6, label="medium (0.5)")
    ax.axvline(0.8, color="gray", linestyle="-", linewidth=0.6, label="large (0.8)")
    ax.axvline(-0.2, color="gray", linestyle=":", linewidth=0.6)
    ax.axvline(-0.5, color="gray", linestyle="--", linewidth=0.6)
    ax.axvline(-0.8, color="gray", linestyle="-", linewidth=0.6)
    ax.set_xlabel("Cohen's d (saturated - not_saturated, positive = higher in saturated)")
    ax.set_title("Effect size by feature")
    ax.legend(loc="lower right", fontsize=7)

    plt.tight_layout()
    plot_path = args.output_dir / "changeseq_saturated_pairs_characterization.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"\nSalvato {plot_path}")

    # === JSON output ===
    payload = {
        "csv_input": str(args.csv),
        "n_total": len(df),
        "n_saturated": n_sat,
        "n_not_saturated": n_nsat,
        "continuous_features": continuous_results,
        "per_position_mismatch_freq": {
            "not_saturated": pos_freq_nsat.tolist(),
            "saturated": pos_freq_sat.tolist(),
        },
        "pam_breakdown_top": pam_table.head(10).to_dict(),
        "per_guide_saturation_stats": {
            "mean_rate": float(per_guide["sat_rate"].mean()),
            "median_rate": float(per_guide["sat_rate"].median()),
            "std_rate": float(per_guide["sat_rate"].std()),
            "n_guides_above_50pct": int((per_guide["sat_rate"] > 0.5).sum()),
            "n_guides_below_10pct": int((per_guide["sat_rate"] < 0.1).sum()),
            "n_guides_total": int(len(per_guide)),
        },
    }
    json_path = args.output_dir / "changeseq_saturated_pairs_characterization.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"Salvato {json_path}")

    # === Verdict ===
    print(f"\n=== VERDICT ===")
    top_features = sorted(continuous_results, key=lambda r: abs(r["cohens_d"]), reverse=True)[:3]
    print(f"Top 3 discriminating features (by |Cohen's d|):")
    for r in top_features:
        print(f"  {r['label']:<40s} d = {r['cohens_d']:+.3f}")


if __name__ == "__main__":
    main()
