import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from models.deep.encoding import BiologicalMismatchEncoder
from models.deep.neural_scm import NeuralSCM


def _mismatch_type_vec(guide_base: str, target_base: str) -> list[float]:
    """Restituisce il vettore one-hot 4D [Match, Wobble, Transition, Transversion]."""
    if guide_base == target_base:
        return [1.0, 0.0, 0.0, 0.0]
    pair = {guide_base, target_base}
    if pair == {"G", "T"}:
        return [0.0, 1.0, 0.0, 0.0]
    if pair in [{"A", "G"}, {"C", "T"}]:
        return [0.0, 0.0, 1.0, 0.0]
    return [0.0, 0.0, 0.0, 1.0]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Caricamento Modello
    # BiologicalMismatchEncoder è necessario per ricostruire le dimensioni PAM corrette
    model_path = Path("experiments/results/Exp15_Positional_ExtendedOneCycle/neural_scm.pt")
    state_dict = torch.load(model_path, map_location=device)

    # Ricava context_dim dal checkpoint per ricostruire il modello fedelmente
    context_dim = 0
    if "context_net.0.weight" in state_dict:
        context_dim = state_dict["context_net.0.weight"].shape[1]

    encoder = BiologicalMismatchEncoder()
    model = NeuralSCM(encoder=encoder, architecture="positional_mlp", hidden_dim=8, context_dim=context_dim)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 2. Estrazione del peso strutturale globale W_i (shape: [20])
    w_pos_eff = -F.softplus(model.w_pos).detach().cpu().numpy()

    # 3. Firme chimiche: coppie (guide_base, target_base) che rappresentano i 4 tipi
    mismatch_pairs = {
        "Match":        ("A", "A"),
        "Wobble":       ("G", "T"),
        "Transition":   ("A", "G"),
        "Transversion": ("A", "C"),
    }

    effective_penalties = {name: [] for name in mismatch_pairs.keys()}
    positions = np.arange(1, 21)

    print("Calcolo Analitico della Matrice 2D (Posizione x Chimica)...")

    with torch.no_grad():
        for name, (guide_base, target_base) in mismatch_pairs.items():
            # Costruiamo il tensore 4D per questa chimica, replicato su 20 posizioni: [1, 20, 4]
            # Stessa logica di _base_forward per positional_mlp
            type_vec = _mismatch_type_vec(guide_base, target_base)
            s_typed = torch.tensor([[type_vec] * 20], dtype=torch.float32, device=device)

            # pos_node (condiviso): [1, 20, 4] -> [1, 20, 1] -> [1, 20]
            pos_penalties = F.relu(model.pos_node(s_typed).squeeze(-1)).cpu().numpy()[0]

            for i in range(20):
                penalty = abs(w_pos_eff[i]) * pos_penalties[i]
                effective_penalties[name].append(penalty)

    # --- PLOTTING DEL PROFILO 2D ---
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))
    
    colors = {"Match": "gray", "Wobble": "blue", "Transition": "orange", "Transversion": "red"}
    styles = {"Match": "--", "Wobble": "-", "Transition": "-", "Transversion": "-"}

    for name, penalties in effective_penalties.items():
        plt.plot(positions, penalties, marker='o', linewidth=2.5, 
                 color=colors[name], linestyle=styles[name], label=name)

    plt.axvspan(0.5, 8.5, color='gray', alpha=0.1, label='Non-Seed')
    plt.axvspan(8.5, 16.5, color='gold', alpha=0.1, label='Seed')
    plt.axvspan(16.5, 20.5, color='red', alpha=0.1, label='PAM-Proximal')

    plt.title("Radiografia Causale: Impatto Termodinamico Strutturale (2D)", fontsize=16, fontweight='bold')
    plt.xlabel("Posizione", fontsize=13)
    plt.ylabel("Penalità Causale Assoluta (|W_i * s_i|)", fontsize=13)
    plt.xticks(positions)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Salvataggio del grafico
    output_path = Path("explainability/plots/thermodynamic_profile.png")
    plt.savefig(output_path, dpi=300)

    print(f"Grafico salvato in: {output_path.resolve()}")

if __name__ == "__main__":
    main()