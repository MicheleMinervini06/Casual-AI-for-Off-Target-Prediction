"""Batch counterfactual analysis: applica interventi a tutte le coppie
(guide, off-target) di un dataset e produce CSV + plot Pareto + distribuzioni di U.

Interventi implementati:
  1) truncate_5p          (sequence-level):  guide → "NN" + guide[2:]
  2) do(pos_14=0)         (DAG node-level):  forza penalità a 0 sul nodo P_14
  3) diversity ACGT       (sequence-level, Treatment-Control):
       Treatment: guide[16:20] = "ACGT"  (massima diversità A/C/G/T)
       Control:   guide[16:20] = "AAAA"  (nessuna diversità)
  4) repeat seed          (sequence-level, Treatment-Control):
       Treatment: guide[8:16] = "ATATATAT"  (perfect period-2 repeat)
       Control:   guide[8:16] = "AAAATTTT"  (stessa composizione, no period-2)

NOTA EPISTEMICA su 3 e 4:
  Il modello positional_mlp processa ogni posizione in modo indipendente — non
  può rappresentare diversità (joint property) o ripetizione (cross-position
  property). Le predizioni risponderanno solo via la somma dei singoli effetti
  posizionali. Implementiamo questi interventi comunque come baseline metodologico
  e per dimostrare empiricamente la limitazione architetturale.

Abduzione pam_gate-aware (Pearl-compliant):
  Il modello è  y = pam_gate * σ(struct_logit + U)
  quindi      U = logit(y_obs / pam_gate) - struct_logit
  con clipping a (eps, 1-eps) per stabilità quando y_obs ≥ pam_gate.

Controfattuale individuale:
  y_cf = pam_gate_cf * σ(struct_logit_cf + U)
  con pam_gate_cf e struct_logit_cf dal forward post-intervento.

Contrasto Treatment-Control (per interventi 3 e 4):
  delta_TC = y_cf_T - y_cf_C
  Effetto "puro" dell'intervento isolato dal contesto della coppia.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from models.deep.encoding import BiologicalMismatchEncoder
from models.deep.neural_scm import NeuralSCM


# ---------- costanti ----------
EPS = 1e-7


# ---------- utility numeriche vettorizzate ----------

def reads_to_prob(reads: np.ndarray, max_reads: np.ndarray, method: str = "log") -> np.ndarray:
    reads = np.maximum(0, reads).astype(np.float64)
    max_reads = np.maximum(reads, max_reads).astype(np.float64)
    if method == "log":
        p = np.log1p(reads) / np.log1p(max_reads)
    elif method == "linear":
        p = reads / max_reads
    else:
        raise ValueError(f"Metodo {method} non supportato.")
    return np.minimum(p * 100.0, 99.0)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def gc_fraction(seq: str) -> float:
    return sum(1 for c in seq if c in "GC") / max(len(seq), 1)


def compute_gc_context_batch(guides: list[str], targets: list[str], device: torch.device) -> torch.Tensor:
    gc_sg = np.array([gc_fraction(g) for g in guides], dtype=np.float32)
    gc_tg = np.array([gc_fraction(t) for t in targets], dtype=np.float32)
    delta = gc_sg - gc_tg
    arr = np.stack([gc_sg, gc_tg, delta], axis=1)
    return torch.tensor(arr, dtype=torch.float32, device=device)


# ---------- abduzione e controfattuale (pam_gate-aware) ----------

def abduct_U(
    y_obs_prob_pct: np.ndarray,
    struct_logit: np.ndarray,
    pam_gate: np.ndarray,
) -> np.ndarray:
    """
    Abduzione Pearl-corretta: U = logit(y_obs / pam_gate) - struct_logit.

    Inversione algebrica del modello  y_obs = pam_gate * σ(struct_logit + U).
    Clipping a (EPS, 1-EPS) gestisce i casi y_obs ≥ pam_gate (incoerenza
    tra osservazione e previsione PAM del modello — raro per NGG canonico).
    """
    p_unit = np.clip(y_obs_prob_pct / 100.0 / pam_gate, EPS, 1.0 - EPS)
    return np.log(p_unit / (1.0 - p_unit)) - struct_logit


def counterfactual_prob_pct(
    struct_logit_cf: np.ndarray,
    pam_gate_cf: np.ndarray,
    U: np.ndarray,
) -> np.ndarray:
    """y_cf (in %) = pam_gate_cf * σ(struct_logit_cf + U) * 100."""
    return pam_gate_cf * sigmoid(struct_logit_cf + U) * 100.0


# ---------- forward batched ----------

def model_forward_batched(
    model: NeuralSCM,
    guides: list[str],
    targets: list[str],
    ctx: torch.Tensor,
    batch_size: int,
    intervention: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Forward batched. Restituisce (struct_logit, pam_gate_post_sigmoid).

    Se `intervention` è fornito, applica `model.do(intervention)` — Pearl
    do-calculus sui nodi DAG. Per positional_mlp sono supportati:
      - intervention["pam_gate"] = <valore pre-sigmoid>
      - intervention["pos_<i>"]  = <penalty value>, i in 0..19
    """
    n = len(guides)
    logits = np.empty(n, dtype=np.float32)
    pam_gates = np.empty(n, dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            j = min(i + batch_size, n)
            if intervention is None:
                out = model(guides[i:j], targets[i:j], context_features=ctx[i:j])
            else:
                out = model.do(
                    guides[i:j], targets[i:j], intervention, context_features=ctx[i:j]
                )
            logits[i:j] = out["logit"].squeeze(-1).cpu().numpy()
            pam_gates[i:j] = out["pam_gate"].squeeze(-1).cpu().numpy()
    return logits, pam_gates


# ---------- interventi a livello sequenza ----------

def truncate_5p(guide: str) -> str:
    """Maschera le prime 2 basi (troncamento 5'). Sequence-level."""
    return "NN" + guide[2:]


def force_pamprox_acgt(guide: str) -> str:
    """Diversity Treatment: forza guide[16:20] = "ACGT" (max diversità A/C/G/T)."""
    return guide[:16] + "ACGT"


def force_pamprox_aaaa(guide: str) -> str:
    """Diversity Control: forza guide[16:20] = "AAAA" (nessuna diversità)."""
    return guide[:16] + "AAAA"


def force_seed_repeat(guide: str) -> str:
    """Repeat Treatment: guide[8:16] = "ATATATAT" (perfect period-2 repeat)."""
    return guide[:8] + "ATATATAT" + guide[16:]


def force_seed_block(guide: str) -> str:
    """Repeat Control: guide[8:16] = "AAAATTTT" (stessa composizione A/T, no period-2)."""
    return guide[:8] + "AAAATTTT" + guide[16:]


# ---------- pipeline ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["changeseq", "guideseq"], default="guideseq")
    parser.add_argument(
        "--model_path",
        default="experiments/results/Exp15_Positional_ExtendedOneCycle/neural_scm.pt",
    )
    parser.add_argument("--output_dir", default="explainability/batch_results")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument(
        "--on-target-mode",
        choices=["drop", "per_run", "global_max"],
        default=None,
        help="Come gestire l'abduzione on-target. Default: per_run per guideseq, drop per changeseq",
    )
    args = parser.parse_args()

    if args.on_target_mode is None:
        args.on_target_mode = "per_run" if args.dataset == "guideseq" else "drop"
    print(f"On-target mode: {args.on_target_mode}")

    if args.dataset == "changeseq":
        csv_path = "data/raw/changeseq/CHANGEseq_positive.csv"
        reads_col = "CHANGEseq_reads"
    else:
        csv_path = "data/raw/guideseq/GUIDEseq_positive.csv"
        reads_col = "GUIDEseq_reads"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Caricamento modello (context_dim ricavato dal checkpoint)
    state_dict = torch.load(args.model_path, map_location=device)
    context_dim = 0
    if "context_net.0.weight" in state_dict:
        context_dim = state_dict["context_net.0.weight"].shape[1]

    encoder = BiologicalMismatchEncoder()
    model = NeuralSCM(
        encoder=encoder,
        architecture="positional_mlp",
        hidden_dim=8,
        context_dim=context_dim,
    )
    model.load_state_dict(state_dict)
    model.to(device).eval()
    print(f"Modello caricato (context_dim={context_dim})")

    # 2. Dataset
    df = pd.read_csv(csv_path)
    print(f"Caricate {len(df)} righe da {csv_path}")

    # 3. Lookup on-target per name (riga distance=0 con max reads)
    on_rows = df[df["distance"] == 0].copy()
    on_lookup = (
        on_rows.sort_values(reads_col, ascending=False)
        .drop_duplicates("name")
        .set_index("name")[["offtarget_sequence", reads_col]]
        .rename(columns={"offtarget_sequence": "on_target_seq", reads_col: "on_reads"})
    )
    print(f"Guide con on-target di riferimento: {len(on_lookup)}")

    # 4. Off-target rows
    off_df = df[df["distance"] > 0].copy().join(on_lookup, on="name", how="inner")
    off_df["sgRNA"] = off_df["target"].str[:20]
    off_df["off_target"] = off_df["offtarget_sequence"]
    off_df["off_reads"] = off_df[reads_col]

    valid = (off_df["sgRNA"].str.len() == 20) & (off_df["off_target"].str.len() == 23) & (
        off_df["on_target_seq"].str.len() == 23
    )
    dropped = (~valid).sum()
    if dropped:
        print(f"[WARN] Scartate {dropped} righe per lunghezze incompatibili")
    off_df = off_df[valid].reset_index(drop=True)
    print(f"Coppie analizzabili: {len(off_df)}")

    # 5a. Probabilità osservate off-target (denominatore = on-target reads della stessa guida)
    off_df["y_obs_off_prob"] = reads_to_prob(off_df["off_reads"].values, off_df["on_reads"].values)

    # 5b. Probabilità osservate on-target — dipende dalla modalità
    if args.on_target_mode == "drop":
        off_df["y_obs_on_prob"] = np.nan
    elif args.on_target_mode == "per_run":
        if "run" not in off_df.columns:
            raise ValueError(
                f"Dataset {args.dataset} non ha colonna 'run', usa --on-target-mode drop o global_max"
            )
        run_max = df.groupby("run")[reads_col].max().to_dict()
        off_df["run_max_reads"] = off_df["run"].map(run_max).astype(np.float64)
        off_df["y_obs_on_prob"] = reads_to_prob(off_df["on_reads"].values, off_df["run_max_reads"].values)
        print(f"Run-level max reads: {run_max}")
    elif args.on_target_mode == "global_max":
        global_max = float(df[reads_col].max())
        off_df["y_obs_on_prob"] = reads_to_prob(
            off_df["on_reads"].values, np.full(len(off_df), global_max)
        )
        print(f"Global max reads: {global_max:.0f}")

    # 6. Costruzione sequenze post-intervento (sequence-level)
    off_df["sgRNA_truncated"] = off_df["sgRNA"].apply(truncate_5p)
    off_df["sgRNA_divT"] = off_df["sgRNA"].apply(force_pamprox_acgt)
    off_df["sgRNA_divC"] = off_df["sgRNA"].apply(force_pamprox_aaaa)
    off_df["sgRNA_repT"] = off_df["sgRNA"].apply(force_seed_repeat)
    off_df["sgRNA_repC"] = off_df["sgRNA"].apply(force_seed_block)

    guides_wt = off_df["sgRNA"].tolist()
    guides_tru = off_df["sgRNA_truncated"].tolist()
    guides_divT = off_df["sgRNA_divT"].tolist()
    guides_divC = off_df["sgRNA_divC"].tolist()
    guides_repT = off_df["sgRNA_repT"].tolist()
    guides_repC = off_df["sgRNA_repC"].tolist()
    off_targets = off_df["off_target"].tolist()
    on_targets = off_df["on_target_seq"].tolist()

    # Helper locale per ridurre boilerplate
    def fwd_pair(guides: list[str], intervention: dict | None = None):
        """Esegue forward su (guides, off_targets) e (guides, on_targets). Restituisce
        ((logit_off, pam_off), (logit_on, pam_on))."""
        ctx_off = compute_gc_context_batch(guides, off_targets, device)
        ctx_on = compute_gc_context_batch(guides, on_targets, device)
        l_off, p_off = model_forward_batched(model, guides, off_targets, ctx_off, args.batch_size, intervention=intervention)
        l_on, p_on = model_forward_batched(model, guides, on_targets, ctx_on, args.batch_size, intervention=intervention)
        return (l_off, p_off), (l_on, p_on)

    # 7a. Forward factual
    print("Forward factual...")
    (logit_off_f, pam_off_f), (logit_on_f, pam_on_f) = fwd_pair(guides_wt)

    # 7b. Sequence intervention: truncation 5'
    print("Forward truncation 5' (sequence intervention)...")
    (logit_off_t, pam_off_t), (logit_on_t, pam_on_t) = fwd_pair(guides_tru)

    # 7c. DAG node intervention: do(pos_14 = 0.0)
    print("Forward do(pos_14 = 0.0) (DAG node intervention)...")
    (logit_off_p14, pam_off_p14), (logit_on_p14, pam_on_p14) = fwd_pair(guides_wt, intervention={"pos_14": 0.0})

    # 7d. Diversity intervention (Treatment ACGT / Control AAAA in pos 16-19)
    print("Forward diversity ACGT (T) e AAAA (C) in pos 16-19 (sequence intervention)...")
    (logit_off_divT, pam_off_divT), (logit_on_divT, pam_on_divT) = fwd_pair(guides_divT)
    (logit_off_divC, pam_off_divC), (logit_on_divC, pam_on_divC) = fwd_pair(guides_divC)

    # 7e. Repeat intervention (Treatment ATATATAT / Control AAAATTTT in pos 8-15)
    print("Forward repeat ATATATAT (T) e AAAATTTT (C) in pos 8-15 (sequence intervention)...")
    (logit_off_repT, pam_off_repT), (logit_on_repT, pam_on_repT) = fwd_pair(guides_repT)
    (logit_off_repC, pam_off_repC), (logit_on_repC, pam_on_repC) = fwd_pair(guides_repC)

    # 8. Predizioni factual (con pam_gate: y_pred = pam_gate * σ(logit), coerente col modello)
    off_df["pam_off_f"] = pam_off_f
    off_df["pam_on_f"] = pam_on_f
    off_df["y_pred_off_prob"] = pam_off_f * sigmoid(logit_off_f) * 100.0
    off_df["y_pred_on_prob"] = pam_on_f * sigmoid(logit_on_f) * 100.0

    # 9. Abduzione off-target (pam_gate-aware)
    off_df["U_off"] = abduct_U(np.asarray(off_df["y_obs_off_prob"].values), logit_off_f, pam_off_f)
    U_off_arr = np.asarray(off_df["U_off"].values)

    # 10. Controfattuali off-target (pam_gate_cf-aware)
    off_df["y_cf_off_tru_prob"] = counterfactual_prob_pct(logit_off_t, pam_off_t, U_off_arr)
    off_df["y_cf_off_p14_prob"] = counterfactual_prob_pct(logit_off_p14, pam_off_p14, U_off_arr)
    off_df["y_cf_off_divT_prob"] = counterfactual_prob_pct(logit_off_divT, pam_off_divT, U_off_arr)
    off_df["y_cf_off_divC_prob"] = counterfactual_prob_pct(logit_off_divC, pam_off_divC, U_off_arr)
    off_df["y_cf_off_repT_prob"] = counterfactual_prob_pct(logit_off_repT, pam_off_repT, U_off_arr)
    off_df["y_cf_off_repC_prob"] = counterfactual_prob_pct(logit_off_repC, pam_off_repC, U_off_arr)

    # Delta vs y_obs (baseline) per interventi single-condition
    off_df["delta_off_tru"] = off_df["y_cf_off_tru_prob"] - off_df["y_obs_off_prob"]
    off_df["delta_off_p14"] = off_df["y_cf_off_p14_prob"] - off_df["y_obs_off_prob"]
    # Delta vs y_obs per T e C separati (utile per stratificare)
    off_df["delta_off_divT"] = off_df["y_cf_off_divT_prob"] - off_df["y_obs_off_prob"]
    off_df["delta_off_divC"] = off_df["y_cf_off_divC_prob"] - off_df["y_obs_off_prob"]
    off_df["delta_off_repT"] = off_df["y_cf_off_repT_prob"] - off_df["y_obs_off_prob"]
    off_df["delta_off_repC"] = off_df["y_cf_off_repC_prob"] - off_df["y_obs_off_prob"]
    # Contrasto Treatment-Control: l'effetto "puro" dell'intervento
    off_df["delta_off_divTC"] = off_df["y_cf_off_divT_prob"] - off_df["y_cf_off_divC_prob"]
    off_df["delta_off_repTC"] = off_df["y_cf_off_repT_prob"] - off_df["y_cf_off_repC_prob"]

    # 11. On-target: due regimi distinti
    if args.on_target_mode == "drop":
        # Nessuna abduzione on-target. Baseline = y_pred_on. CF = pure model output.
        off_df["U_on"] = np.nan
        off_df["y_cf_on_tru_prob"] = pam_on_t * sigmoid(logit_on_t) * 100.0
        off_df["y_cf_on_p14_prob"] = pam_on_p14 * sigmoid(logit_on_p14) * 100.0
        off_df["y_cf_on_divT_prob"] = pam_on_divT * sigmoid(logit_on_divT) * 100.0
        off_df["y_cf_on_divC_prob"] = pam_on_divC * sigmoid(logit_on_divC) * 100.0
        off_df["y_cf_on_repT_prob"] = pam_on_repT * sigmoid(logit_on_repT) * 100.0
        off_df["y_cf_on_repC_prob"] = pam_on_repC * sigmoid(logit_on_repC) * 100.0
        baseline_on = off_df["y_pred_on_prob"]
    else:
        off_df["U_on"] = abduct_U(np.asarray(off_df["y_obs_on_prob"].values), logit_on_f, pam_on_f)
        U_on_arr = np.asarray(off_df["U_on"].values)
        off_df["y_cf_on_tru_prob"] = counterfactual_prob_pct(logit_on_t, pam_on_t, U_on_arr)
        off_df["y_cf_on_p14_prob"] = counterfactual_prob_pct(logit_on_p14, pam_on_p14, U_on_arr)
        off_df["y_cf_on_divT_prob"] = counterfactual_prob_pct(logit_on_divT, pam_on_divT, U_on_arr)
        off_df["y_cf_on_divC_prob"] = counterfactual_prob_pct(logit_on_divC, pam_on_divC, U_on_arr)
        off_df["y_cf_on_repT_prob"] = counterfactual_prob_pct(logit_on_repT, pam_on_repT, U_on_arr)
        off_df["y_cf_on_repC_prob"] = counterfactual_prob_pct(logit_on_repC, pam_on_repC, U_on_arr)
        baseline_on = off_df["y_obs_on_prob"]

    off_df["delta_on_tru"] = off_df["y_cf_on_tru_prob"] - baseline_on
    off_df["delta_on_p14"] = off_df["y_cf_on_p14_prob"] - baseline_on
    off_df["delta_on_divT"] = off_df["y_cf_on_divT_prob"] - baseline_on
    off_df["delta_on_divC"] = off_df["y_cf_on_divC_prob"] - baseline_on
    off_df["delta_on_repT"] = off_df["y_cf_on_repT_prob"] - baseline_on
    off_df["delta_on_repC"] = off_df["y_cf_on_repC_prob"] - baseline_on
    off_df["delta_on_divTC"] = off_df["y_cf_on_divT_prob"] - off_df["y_cf_on_divC_prob"]
    off_df["delta_on_repTC"] = off_df["y_cf_on_repT_prob"] - off_df["y_cf_on_repC_prob"]

    # 12. Salvataggio CSV (sottoinsieme leggibile delle colonne)
    keep_cols = [
        "name", "sgRNA", "off_target", "on_target_seq", "distance",
        "off_reads", "on_reads",
        "pam_off_f", "pam_on_f",
        "y_obs_off_prob", "y_pred_off_prob", "U_off",
        "y_obs_on_prob", "y_pred_on_prob", "U_on",
        # Single-condition (vs baseline)
        "y_cf_off_tru_prob", "y_cf_on_tru_prob", "delta_off_tru", "delta_on_tru",
        "y_cf_off_p14_prob", "y_cf_on_p14_prob", "delta_off_p14", "delta_on_p14",
        # Diversity (T-C contrast)
        "y_cf_off_divT_prob", "y_cf_off_divC_prob",
        "y_cf_on_divT_prob", "y_cf_on_divC_prob",
        "delta_off_divT", "delta_off_divC", "delta_off_divTC",
        "delta_on_divT", "delta_on_divC", "delta_on_divTC",
        # Repeat (T-C contrast)
        "y_cf_off_repT_prob", "y_cf_off_repC_prob",
        "y_cf_on_repT_prob", "y_cf_on_repC_prob",
        "delta_off_repT", "delta_off_repC", "delta_off_repTC",
        "delta_on_repT", "delta_on_repC", "delta_on_repTC",
    ]
    out_csv = output_dir / f"{args.dataset}_batch_results.csv"
    off_df[keep_cols].to_csv(out_csv, index=False)
    print(f"\nSalvato {out_csv}")

    # 13. Plot Pareto trade-off — 4 interventi
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.scatter(off_df["delta_on_tru"], off_df["delta_off_tru"],
               alpha=0.30, color="steelblue", s=10, label="Truncation 5' (sequence)")
    ax.scatter(off_df["delta_on_p14"], off_df["delta_off_p14"],
               alpha=0.30, color="crimson", s=10, label="do(pos_14 = 0) (DAG node)")
    ax.scatter(off_df["delta_on_divTC"], off_df["delta_off_divTC"],
               alpha=0.30, color="forestgreen", s=10, label="Diversity ACGT vs AAAA, T-C contrast")
    ax.scatter(off_df["delta_on_repTC"], off_df["delta_off_repTC"],
               alpha=0.30, color="darkorange", s=10, label="Repeat ATATATAT vs AAAATTTT, T-C contrast")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Delta On-Target Probability (cf - baseline) [%]")
    ax.set_ylabel("Delta Off-Target Probability (cf - baseline) [%]")
    ax.set_title(f"Pareto Trade-Off Causale ({args.dataset})\n"
                 f"Quadrante in basso-a-destra = ideale (off↓, on↑)")
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    pareto_path = output_dir / f"{args.dataset}_pareto.png"
    plt.savefig(pareto_path, dpi=200)
    plt.close()
    print(f"Salvato {pareto_path}")

    # 14. Plot distribuzione del rumore U
    has_u_on = args.on_target_mode != "drop"
    n_panels = 2 if has_u_on else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5), squeeze=False)
    panels = [(axes[0, 0], "U_off", "steelblue", "Rumore Esogeno U_off")]
    if has_u_on:
        panels.append((axes[0, 1], "U_on", "darkorange", f"Rumore Esogeno U_on ({args.on_target_mode})"))
    for ax, col, color, title in panels:
        vals = off_df[col].dropna().values
        ax.hist(vals, bins=60, color=color, alpha=0.75, edgecolor="white")
        ax.axvline(np.mean(vals), color="red", linestyle="--",
                   label=f"mean={np.mean(vals):+.3f}")
        ax.axvline(np.median(vals), color="black", linestyle=":",
                   label=f"median={np.median(vals):+.3f}")
        ax.set_xlabel(col)
        ax.set_title(title)
        ax.legend()
    plt.tight_layout()
    u_path = output_dir / f"{args.dataset}_U_distribution.png"
    plt.savefig(u_path, dpi=200)
    plt.close()
    print(f"Salvato {u_path}")

    # 15. Sommario su stdout — aggregato globale (per coppia)
    print("\n=== SOMMARIO GLOBALE (per coppia) ===")
    print(f"Coppie analizzate: {len(off_df)}")
    print(f"Guide uniche:      {off_df['name'].nunique()}")
    print(f"\npam_off  mean={off_df['pam_off_f'].mean():.3f}  std={off_df['pam_off_f'].std():.3f}")
    print(f"pam_on   mean={off_df['pam_on_f'].mean():.3f}  std={off_df['pam_on_f'].std():.3f}")
    print(f"\nU_off  mean={off_df['U_off'].mean():+.3f}  std={off_df['U_off'].std():.3f}  "
          f"median={off_df['U_off'].median():+.3f}")
    if has_u_on:
        print(f"U_on   mean={off_df['U_on'].mean():+.3f}  std={off_df['U_on'].std():.3f}  "
              f"median={off_df['U_on'].median():+.3f}")
    else:
        print("U_on   N/A (on-target mode = drop, nessuna abduzione)")

    interventions_summary = [
        ("Truncation 5' (sequence)", "delta_off_tru", "delta_on_tru"),
        ("do(pos_14 = 0) (DAG node)", "delta_off_p14", "delta_on_p14"),
        ("Diversity ACGT vs AAAA, T-C contrast", "delta_off_divTC", "delta_on_divTC"),
        ("Repeat ATATATAT vs AAAATTTT, T-C contrast", "delta_off_repTC", "delta_on_repTC"),
    ]
    for label, dcoff, dcon in interventions_summary:
        print(f"\n{label}:")
        print(f"  Delta off mean={off_df[dcoff].mean():+.2f}%  std={off_df[dcoff].std():.2f}")
        print(f"  Delta on  mean={off_df[dcon].mean():+.2f}%  std={off_df[dcon].std():.2f}")
        ideal = ((off_df[dcoff] < 0) & (off_df[dcon] >= -5)).sum()
        print(f"  Coppie nel quadrante ideale (Deltaoff<0 e Deltaon>=-5%): {ideal} ({100*ideal/len(off_df):.1f}%)")

    # Diagnostica supplementare: T e C separatamente per diversity e repeat
    print("\n--- Diagnostica T vs C per interventi diversity/repeat (per coppia) ---")
    for prefix, label in [("div", "Diversity"), ("rep", "Repeat")]:
        for side in ("off", "on"):
            T_col = f"delta_{side}_{prefix}T"
            C_col = f"delta_{side}_{prefix}C"
            print(f"  {label} {side}-target:  "
                  f"delta_T mean={off_df[T_col].mean():+.2f}%  "
                  f"delta_C mean={off_df[C_col].mean():+.2f}%  "
                  f"contrast(T-C) mean={(off_df[T_col]-off_df[C_col]).mean():+.2f}%")

    # 16. Sommario per-guida: mediana entro guida, poi statistiche su quei valori
    delta_cols = [
        "delta_off_tru", "delta_off_p14", "delta_off_divTC", "delta_off_repTC",
        "delta_on_tru", "delta_on_p14", "delta_on_divTC", "delta_on_repTC",
    ]
    u_cols = ["U_off"] + (["U_on"] if has_u_on else [])
    per_guide = off_df.groupby("name")[delta_cols + u_cols].median()

    print(f"\n=== SOMMARIO PER-GUIDA (mediana entro guida -> distribuzione su {len(per_guide)} guide) ===")
    print("(mitiga il bias delle guide con molti off-target nella media globale)")
    print()
    summary = per_guide.describe().loc[["mean", "std", "min", "25%", "50%", "75%", "max"]].round(3)
    print(summary.to_string())
    print()
    for label, dcoff, dcon in interventions_summary:
        n_g = len(per_guide)
        ideal_g = ((per_guide[dcoff] < 0) & (per_guide[dcon] >= -5)).sum()
        print(f"{label}: guide nel quadrante ideale (mediana Deltaoff<0 e Deltaon>=-5%): "
              f"{ideal_g}/{n_g} ({100*ideal_g/n_g:.1f}%)")

    per_guide_path = output_dir / f"{args.dataset}_per_guide_medians.csv"
    per_guide.to_csv(per_guide_path)
    print(f"\nSalvato {per_guide_path}")


if __name__ == "__main__":
    main()
