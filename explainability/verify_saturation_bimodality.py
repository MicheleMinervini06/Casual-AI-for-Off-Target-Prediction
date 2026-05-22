"""Verifica empirica: Mode 2 della distribuzione U_off post-shift su CHANGE-seq
corrisponde alle coppie con saturazione del cap reads_to_prob (off_reads >= on_reads)?

Test F22 — conferma o falsifica l'ipotesi che la bimodalità di U_off post-calibrazione
sia generata dalla censuratura misurativa del cell-free assay.

Input: il CSV prodotto da `simulate_intervention_batch.py --dataset changeseq --assay-shift 2.73`
       (default: explainability/batch_results/changeseq_batch_results_shift+2.73.csv).

Output:
  - Statistiche stratificate U_off per saturated vs normal
  - Plot istogramma sovrapposto colorato per saturation status
  - JSON con metriche di separazione (es. % di Mode 2 spiegato da saturazione)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("explainability/batch_results/changeseq_batch_results_shift+2.73.csv"),
        help="CSV prodotto da simulate_intervention_batch.py con assay shift applicato",
    )
    parser.add_argument(
        "--mode2-threshold",
        type=float,
        default=0.5,
        help="Soglia U_off oltre la quale considerare una coppia parte di Mode 2 (default: +0.5)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("explainability/batch_results"),
    )
    args = parser.parse_args()

    print(f"Loading {args.csv}")
    df = pd.read_csv(args.csv)
    print(f"  rows: {len(df)}")

    # Definizione operativa di "saturazione misurativa"
    # Una coppia è saturata se y_obs è all'upper cap di reads_to_prob (= 99%).
    # Questo accade quando off_reads >= on_reads.
    df["is_saturated"] = (df["off_reads"] >= df["on_reads"]).astype(bool)
    df["in_mode2"] = (df["U_off"] >= args.mode2_threshold).astype(bool)

    n_total = len(df)
    n_saturated = int(df["is_saturated"].sum())
    n_mode2 = int(df["in_mode2"].sum())
    n_overlap = int((df["is_saturated"] & df["in_mode2"]).sum())

    print(f"\n=== POPULATIONS ===")
    print(f"  total pairs:          {n_total}")
    print(f"  saturated (off>=on):   {n_saturated}  ({100*n_saturated/n_total:.1f}%)")
    print(f"  in Mode 2 (U>={args.mode2_threshold:+.2f}):  {n_mode2}  ({100*n_mode2/n_total:.1f}%)")
    print(f"  overlap saturated & Mode2: {n_overlap}")

    # ---- Confusion-matrix-style breakdown ----
    n_sat_in_m2 = n_overlap
    n_sat_not_m2 = n_saturated - n_overlap
    n_not_sat_in_m2 = n_mode2 - n_overlap
    n_not_sat_not_m2 = n_total - n_saturated - n_not_sat_in_m2

    print(f"\n=== STRATIFIED 2x2 ===")
    print(f"                       Mode 2     not Mode 2     Total")
    print(f"  saturated      :  {n_sat_in_m2:>7d}   {n_sat_not_m2:>10d}   {n_saturated:>7d}")
    print(f"  not saturated  :  {n_not_sat_in_m2:>7d}   {n_not_sat_not_m2:>10d}   {n_total - n_saturated:>7d}")
    print(f"  ----------------------------------------------------------")
    print(f"  total          :  {n_mode2:>7d}   {n_total - n_mode2:>10d}   {n_total:>7d}")

    # Metriche di associazione
    # Sensitivity (recall): se una coppia è saturata, qual è la prob di stare in Mode 2?
    sens = n_sat_in_m2 / max(n_saturated, 1)
    # Precision: se una coppia è in Mode 2, qual è la prob di essere saturata?
    prec = n_sat_in_m2 / max(n_mode2, 1)
    # Specificity: se non-saturata, prob di non stare in Mode 2
    spec = n_not_sat_not_m2 / max(n_total - n_saturated, 1)
    # Fraction of Mode 2 explained by saturation
    explained = n_sat_in_m2 / max(n_mode2, 1)

    print(f"\n=== ASSOCIATION METRICS ===")
    print(f"  P(Mode 2 | saturated)         = {sens:.3f}   (sensitivity / recall)")
    print(f"  P(saturated | Mode 2)         = {prec:.3f}   (precision)")
    print(f"  P(not Mode 2 | not saturated) = {spec:.3f}   (specificity)")
    print(f"  Mode 2 explained by saturation: {100*explained:.1f}%")

    # Statistiche U_off per gruppo
    print(f"\n=== U_off STATS BY SATURATION ===")
    for sat_value in (False, True):
        subset = df[df["is_saturated"] == sat_value]
        label = "saturated   " if sat_value else "not saturated"
        if len(subset) == 0:
            print(f"  {label}: (empty)")
            continue
        u = subset["U_off"].values
        print(f"  {label}: n={len(subset):>6d}  mean={np.mean(u):+.3f}  median={np.median(u):+.3f}  std={np.std(u):.3f}  "
              f"q25={np.percentile(u, 25):+.3f}  q75={np.percentile(u, 75):+.3f}")

    # ---- Plot ----
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    bins = np.linspace(df["U_off"].min() - 0.1, df["U_off"].max() + 0.1, 80)

    # Pannello 1: distribuzioni separate
    ax = axes[0]
    u_normal = df[~df["is_saturated"]]["U_off"].values
    u_sat = df[df["is_saturated"]]["U_off"].values
    ax.hist(u_normal, bins=bins, color="steelblue", alpha=0.7, edgecolor="white",
            label=f"Not saturated (off_reads < on_reads), n={len(u_normal)}")
    ax.hist(u_sat, bins=bins, color="crimson", alpha=0.7, edgecolor="white",
            label=f"Saturated (off_reads >= on_reads), n={len(u_sat)}")
    ax.axvline(args.mode2_threshold, color="black", linestyle="--", linewidth=1,
               label=f"Mode 2 threshold (U >= {args.mode2_threshold:+.2f})")
    ax.set_ylabel("Frequency")
    ax.set_title(f"U_off distribution stratified by saturation status — CHANGE-seq, post-shift +2.73\n"
                 f"P(Mode 2 | saturated) = {sens:.3f}   P(saturated | Mode 2) = {prec:.3f}")
    ax.legend(loc="upper right")

    # Pannello 2: contributo cumulato a Mode 2
    ax = axes[1]
    ax.hist(u_normal, bins=bins, color="steelblue", alpha=0.5, edgecolor="white",
            label="Not saturated", stacked=False)
    ax.hist(u_sat, bins=bins, color="crimson", alpha=0.5, edgecolor="white",
            bottom=np.histogram(u_normal, bins=bins)[0],
            label="Saturated (stacked on top)", stacked=False)
    ax.axvline(args.mode2_threshold, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("U_off (logit units, post-shift)")
    ax.set_ylabel("Frequency (stacked)")
    ax.set_title("Stacked view: which population dominates each U_off range?")
    ax.legend(loc="upper right")

    plt.tight_layout()
    plot_path = args.output_dir / "changeseq_U_distribution_saturation_stratified.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"\nSalvato {plot_path}")

    # ---- JSON output ----
    payload = {
        "csv_input": str(args.csv),
        "mode2_threshold": args.mode2_threshold,
        "n_total": n_total,
        "n_saturated": n_saturated,
        "n_mode2": n_mode2,
        "n_overlap": n_overlap,
        "confusion": {
            "saturated_in_mode2": n_sat_in_m2,
            "saturated_not_mode2": n_sat_not_m2,
            "not_saturated_in_mode2": n_not_sat_in_m2,
            "not_saturated_not_mode2": n_not_sat_not_m2,
        },
        "metrics": {
            "P_mode2_given_saturated": sens,
            "P_saturated_given_mode2": prec,
            "P_notmode2_given_notsaturated": spec,
            "fraction_mode2_explained_by_saturation": explained,
        },
        "u_off_by_group": {
            "saturated": {
                "n": int(len(u_sat)),
                "mean": float(np.mean(u_sat)),
                "median": float(np.median(u_sat)),
                "std": float(np.std(u_sat)),
            },
            "not_saturated": {
                "n": int(len(u_normal)),
                "mean": float(np.mean(u_normal)),
                "median": float(np.median(u_normal)),
                "std": float(np.std(u_normal)),
            },
        },
    }
    json_path = args.output_dir / "changeseq_saturation_verification.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Salvato {json_path}")

    # ---- Verdetto ----
    print(f"\n=== VERDICT ===")
    if prec >= 0.85 and explained >= 0.85:
        print(f"[OK] HYPOTHESIS CONFIRMED: Mode 2 is dominantly composed of saturated pairs.")
        print(f"  {100*prec:.1f}% of Mode 2 pairs are saturated.")
        print(f"  The bimodality is a direct signature of the reads_to_prob cap.")
    elif prec >= 0.50:
        print(f"[~] HYPOTHESIS PARTIALLY CONFIRMED: saturation explains {100*prec:.1f}% of Mode 2.")
        print(f"  There may be additional sources of the +1.5 cluster worth investigating.")
    else:
        print(f"[X] HYPOTHESIS REJECTED: only {100*prec:.1f}% of Mode 2 pairs are saturated.")
        print(f"  The bimodality has a different origin — investigate further.")


if __name__ == "__main__":
    main()
