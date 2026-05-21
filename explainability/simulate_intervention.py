"""Single-pair counterfactual demo per il Neural SCM.

Default model: Exp18_Positional_AdditivePAM (pam_mode=additive).

Esegue la procedura Pearl in 3 step:
  1) Abduction:    U = logit(y_obs) - struct_logit  (formula additiva)
  2) Action:       do(...) — qui implementato come modifica della sequenza
                    della guida (intervento sul nodo radice esogeno)
  3) Prediction:   y_cf = σ(struct_logit_cf + U)

Per modelli multiplicativi (Run 15 e precedenti), passare --pam-mode multiplicative.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from models.deep.encoding import BiologicalMismatchEncoder
from models.deep.neural_scm import NeuralSCM


EPS = 1e-7


def calculate_prob(logit: float) -> float:
    return torch.sigmoid(torch.tensor(logit)).item() * 100


def logit_from_prob(prob_pct: float) -> float:
    """Ricava il logit dalla probabilità, evitando infiniti."""
    p = np.clip(prob_pct / 100.0, EPS, 1.0 - EPS)
    return float(np.log(p / (1.0 - p)))


def reads_to_prob(reads: int, max_reads: int, method: str = "log") -> float:
    """Converte i read counts in probabilità mitigando i bias di amplificazione PCR."""
    reads = max(0, reads)
    max_reads = max(reads, max_reads)
    if method == "linear":
        p = reads / max_reads
    elif method == "log":
        p = np.log1p(reads) / np.log1p(max_reads)
    else:
        raise ValueError(f"Metodo {method} non supportato.")
    return min(p * 100.0, 99.0)


def compute_gc_context(sgRNA: str, target: str, device: torch.device) -> torch.Tensor:
    gc_sg = sum(1 for c in sgRNA if c in 'GC') / len(sgRNA)
    gc_tg = sum(1 for c in target if c in 'GC') / len(target)
    delta = gc_sg - gc_tg
    return torch.tensor([[gc_sg, gc_tg, delta]], dtype=torch.float32, device=device)


def abduct_U(y_obs_prob_pct: float, struct_logit: float, pam_gate: float, pam_mode: str) -> float:
    """Abduction mode-aware. In additive il pam contribuisce già a struct_logit."""
    if pam_mode == "additive":
        p_unit = np.clip(y_obs_prob_pct / 100.0, EPS, 1.0 - EPS)
    elif pam_mode == "multiplicative":
        p_unit = np.clip(y_obs_prob_pct / 100.0 / pam_gate, EPS, 1.0 - EPS)
    else:
        raise ValueError(f"pam_mode non riconosciuto: {pam_mode}")
    return float(np.log(p_unit / (1.0 - p_unit)) - struct_logit)


def counterfactual_prob(struct_logit_cf: float, pam_gate_cf: float, U: float, pam_mode: str) -> float:
    """CF mode-aware."""
    if pam_mode == "additive":
        return float(1.0 / (1.0 + np.exp(-(struct_logit_cf + U))) * 100.0)
    elif pam_mode == "multiplicative":
        return float(pam_gate_cf * 1.0 / (1.0 + np.exp(-(struct_logit_cf + U))) * 100.0)
    raise ValueError(f"pam_mode non riconosciuto: {pam_mode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=Path,
        default=Path("experiments/results/Exp18_Positional_AdditivePAM/neural_scm.pt"),
    )
    parser.add_argument(
        "--pam-mode",
        choices=["additive", "multiplicative"],
        default="additive",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inizializzazione Oracolo Causale su device: {device}")
    print(f"PAM mode:  {args.pam_mode}")
    print(f"Model:     {args.model_path}\n")

    if not args.model_path.exists():
        raise FileNotFoundError(f"Modello non trovato in {args.model_path}")

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
        pam_mode=args.pam_mode,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # ==========================================
    # DATI FATTUALI (Esempio dal Dataset)
    # ==========================================
    guide_wt =   "GTCCCTAGTGGCCCCACTGT"
    off_target = "GTCCCCAAAGCCCCCACTGTGGG"

    off_target_reads = 193     # Off-target reads
    on_target_reads = 51571    # Max reads dell'esperimento

    y_obs_prob = reads_to_prob(off_target_reads, on_target_reads, method="log")

    print("--- 1. ABDUCTION (Calcolo del Rumore Cellulare U) ---")
    with torch.no_grad():
        ctx_factual = compute_gc_context(guide_wt, off_target, device)
        out_factual = model([guide_wt], [off_target], context_features=ctx_factual)
        y_pred_logit = out_factual['logit'].item()
        pam_gate_f = out_factual['pam_gate'].item()

        U = abduct_U(y_obs_prob, y_pred_logit, pam_gate_f, args.pam_mode)

        if args.pam_mode == "additive":
            y_pred_pct = calculate_prob(y_pred_logit)
        else:
            y_pred_pct = pam_gate_f * calculate_prob(y_pred_logit) / 100.0 * 100.0

        print(f"Dati Lab (Fattuali): {off_target_reads} reads (Efficienza: {y_obs_prob:.1f}%)")
        print(f"Predizione del modello: {y_pred_pct:.1f}%")
        print(f"pam_gate (diagnostico): {pam_gate_f:.3f}")
        print(f"-> Rumore Esogeno Inferito (U): {U:+.3f}\n")

    print("--- 2. INTERVENTO (Do-Calculus: Troncamento della Guida) ---")
    guide_tru = "NN" + guide_wt[2:]

    with torch.no_grad():
        ctx_tru = compute_gc_context(guide_tru, off_target, device)
        out_intervened = model([guide_tru], [off_target], context_features=ctx_tru)
        y_do_logit = out_intervened['logit'].item()
        pam_gate_tru = out_intervened['pam_gate'].item()

        print(f"Azione: do(Guida = Troncata di 2 nucleotidi al 5')")
        if args.pam_mode == "additive":
            impact = calculate_prob(y_do_logit)
        else:
            impact = pam_gate_tru * calculate_prob(y_do_logit) / 100.0 * 100.0
        print(f"Impatto post-intervento (senza U): {impact:.1f}%\n")

    print("--- 3. PREDICATO CONTROFATTUALE COMPLETO ---")
    y_counterfactual_prob = counterfactual_prob(y_do_logit, pam_gate_tru, U, args.pam_mode)

    print(f"Se in quello specifico esperimento avessimo usato la guida troncata,")
    print(f"i {off_target_reads} read sarebbero diventati un'efficienza del:")
    print(f"-> {y_counterfactual_prob:.1f}%\n")

    print("--- 4. INTERVENTO B (Do-Calculus: Heal posizione 15 via DAG node) ---")
    # Usa il vero do() sui nodi DAG invece della mutazione di sequenza.
    # `do({"pos_14": 0.0})` forza la penalità della posizione 15 (indice 14) a 0.
    with torch.no_grad():
        out_intervened_p14 = model.do([guide_wt], [off_target], {"pos_14": 0.0}, context_features=ctx_factual)
        y_do_p14_logit = out_intervened_p14['logit'].item()
        pam_gate_p14 = out_intervened_p14['pam_gate'].item()

        print(f"Azione: do(pos_14 = 0) — heal della penalità in posizione 15")
        if args.pam_mode == "additive":
            impact_p14 = calculate_prob(y_do_p14_logit)
        else:
            impact_p14 = pam_gate_p14 * calculate_prob(y_do_p14_logit) / 100.0 * 100.0
        print(f"Impatto post-intervento (senza U): {impact_p14:.1f}%\n")

    print("--- 5. PREDICATO CONTROFATTUALE (Intervento B) ---")
    y_cf_p14_prob = counterfactual_prob(y_do_p14_logit, pam_gate_p14, U, args.pam_mode)
    print(f"Se la posizione 15 fosse stata perfettamente appaiata,")
    print(f"l'efficienza sarebbe diventata:")
    print(f"-> {y_cf_p14_prob:.1f}%\n")

    print("--- 6. TRADE-OFF CLINICO CAUSALE (Impatto sull'On-Target) ---")
    on_target = "GTCCCTAGTGGCCCCACTGTGGG"
    true_on_target_reads = 51571

    # Abduzione On-Target
    y_obs_on_prob = reads_to_prob(true_on_target_reads, on_target_reads, method="log")

    with torch.no_grad():
        ctx_on = compute_gc_context(guide_wt, on_target, device)
        out_on_factual = model([guide_wt], [on_target], context_features=ctx_on)
        y_pred_on_logit = out_on_factual['logit'].item()
        pam_gate_on = out_on_factual['pam_gate'].item()

        U_on = abduct_U(y_obs_on_prob, y_pred_on_logit, pam_gate_on, args.pam_mode)

        if args.pam_mode == "additive":
            y_pred_on_pct = calculate_prob(y_pred_on_logit)
        else:
            y_pred_on_pct = pam_gate_on * calculate_prob(y_pred_on_logit) / 100.0 * 100.0

        print(f"Abduzione On-Target: {true_on_target_reads} reads (Efficienza: {y_obs_on_prob:.1f}%)")
        print(f"Predizione del modello (Match perfetto): {y_pred_on_pct:.1f}%")
        print(f"-> Rumore Esogeno On-Target (U_on): {U_on:+.3f}\n")

        # Controfattuale On-Target sotto intervento truncation
        out_on_tru = model([guide_tru], [on_target], context_features=ctx_on)
        y_do_on_logit = out_on_tru['logit'].item()
        pam_gate_on_tru = out_on_tru['pam_gate'].item()
        y_cf_on_prob = counterfactual_prob(y_do_on_logit, pam_gate_on_tru, U_on, args.pam_mode)

        # Controfattuale On-Target sotto intervento do(pos_14=0)
        out_on_p14 = model.do([guide_wt], [on_target], {"pos_14": 0.0}, context_features=ctx_on)
        y_do_on_p14_logit = out_on_p14['logit'].item()
        pam_gate_on_p14 = out_on_p14['pam_gate'].item()
        y_cf_on_p14_prob = counterfactual_prob(y_do_on_p14_logit, pam_gate_on_p14, U_on, args.pam_mode)

        print("--- VERDETTO FINALE DELL'ORACOLO ---")
        print(f"Strategia            |    On-Target    |    Off-Target")
        print(f"---------------------|-----------------|---------------")
        print(f"Fattuale             |  {y_obs_on_prob:5.1f}%        |  {y_obs_prob:5.1f}%")
        print(f"Truncation 5'        |  {y_cf_on_prob:5.1f}%        |  {y_counterfactual_prob:5.1f}%")
        print(f"do(pos_14 = 0)       |  {y_cf_on_p14_prob:5.1f}%        |  {y_cf_p14_prob:5.1f}%")
        print()

        for label, y_on, y_off in [
            ("Truncation 5'", y_cf_on_prob, y_counterfactual_prob),
            ("do(pos_14 = 0)", y_cf_on_p14_prob, y_cf_p14_prob),
        ]:
            if y_on > 50.0 and y_off < 20.0:
                print(f"[{label}] APPROVATA: efficacia on-target preservata, rischio off-target abbattuto.")
            elif y_off > y_obs_prob:
                print(f"[{label}] PEGGIORATIVA: l'off-target è aumentato.")
            else:
                print(f"[{label}] AMBIGUA: trade-off da valutare caso per caso.")


if __name__ == "__main__":
    main()
