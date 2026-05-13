"""Neural SCM architecture figure — the implementation that realises
the causal DAG (see thesis_fig0_dag.png).

Visual language stays consistent with the DAG figure:
  - light blue   = observed/input data
  - light orange = intermediate tensor / latent value
  - light yellow = trainable module (MLP, encoder)
  - light red    = outcome
  - blue edges   = sign-constrained weighted path
  - red edge     = multiplicative gate
  - green edge   = additive context shift
  - gray edge    = structural / functional dependency
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path("explainability/plots/thesis_fig0b_architecture.png")

# --- visual palette ----------------------------------------------------
COLOR_INPUT   = "#bcd6ec"   # light blue   — observed/input data
COLOR_MODULE  = "#fff2b3"   # light yellow — trainable module
COLOR_LATENT  = "#ffd9a8"   # light orange — intermediate tensor
COLOR_OUTCOME = "#f5a8a8"   # light red    — outcome
COLOR_EDGE_W  = "#1f3b6e"   # dark blue    — sign-constrained edge
COLOR_EDGE_M  = "#a02020"   # dark red     — multiplicative gate
COLOR_EDGE_C  = "#2a7a2a"   # dark green   — additive context
COLOR_EDGE_F  = "#666666"   # gray         — structural / functional


def box(ax, xy, text, kind, w=1.7, h=0.8, fontsize=10.5):
    color = {
        "input": COLOR_INPUT, "module": COLOR_MODULE,
        "latent": COLOR_LATENT, "outcome": COLOR_OUTCOME,
    }[kind]
    x, y = xy
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.05,rounding_size=0.10",
        linewidth=1.2, edgecolor="black", facecolor=color, zorder=3,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", zorder=4)


def arrow(ax, src, dst, color, label=None, label_pos=0.5,
          style="-", curvature=0.0, lw=1.6):
    a = FancyArrowPatch(
        src, dst,
        arrowstyle="-|>", mutation_scale=14,
        linestyle=style, linewidth=lw, color=color,
        connectionstyle=f"arc3,rad={curvature}", zorder=2,
        shrinkA=16, shrinkB=16,
    )
    ax.add_patch(a)
    if label:
        mx = src[0] + (dst[0] - src[0]) * label_pos
        my = src[1] + (dst[1] - src[1]) * label_pos
        ax.text(mx, my, label, ha="center", va="center",
                fontsize=8.5, color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                          edgecolor="none", alpha=0.88), zorder=5)


def main():
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(-1, 13)
    ax.set_ylim(-0.5, 11.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # ===================================================================
    # ROW 1 — raw inputs
    # ===================================================================
    box(ax, (3.5, 10.5), "sgRNA (20 nt)\n+ off-target (23 nt)", "input", w=3.4)
    box(ax, (10.0, 10.5), "GC composition\n[B × 3]", "input", w=2.6)

    # ===================================================================
    # ROW 2 — encoder
    # ===================================================================
    box(ax, (3.5, 8.8), "BiologicalMismatchEncoder", "module", w=4.2)

    # arrow inputs -> encoder
    arrow(ax, (3.5, 10.1), (3.5, 9.2), COLOR_EDGE_F, lw=1.4)

    # ===================================================================
    # ROW 3 — encoded tensors (split spacer vs PAM)
    # ===================================================================
    box(ax, (1.5, 7.3),
        "spacer encoding\n[B × 20 × 12]\n(4 type + 4 sgRNA + 4 target)",
        "latent", w=2.6, h=1.1)
    box(ax, (5.5, 7.3),
        "PAM encoding\n[B × 3 × 12]\n(5 one-hot + 7 pad)",
        "latent", w=2.6, h=1.1)

    arrow(ax, (3.0, 8.45), (1.5, 7.90), COLOR_EDGE_F, lw=1.4)
    arrow(ax, (4.0, 8.45), (5.5, 7.90), COLOR_EDGE_F, lw=1.4)

    # ===================================================================
    # ROW 4 — per-pathway processing modules
    # ===================================================================
    box(ax, (1.5, 5.8),  "pos_node\n(shared MLP,\nLinear-ReLU-Linear)", "module", w=2.4, h=1.2)
    box(ax, (5.5, 5.8),  "PAMModule\n(MLP + sigmoid)",                  "module", w=2.4, h=1.2)
    box(ax, (10.0, 5.8), "context_net\n(MLP)",                          "module", w=2.4, h=1.2)

    arrow(ax, (1.5, 6.75), (1.5, 6.45), COLOR_EDGE_F, lw=1.4,
          label="select type (4 dim)", label_pos=0.5)
    arrow(ax, (5.5, 6.75), (5.5, 6.45), COLOR_EDGE_F, lw=1.4)
    arrow(ax, (10.0, 10.05), (10.0, 6.45), COLOR_EDGE_F, lw=1.4)

    # ===================================================================
    # ROW 5 — per-pathway outputs (intermediate tensors)
    # ===================================================================
    box(ax, (1.5, 4.0),  "s_i  ≥ 0\n[B × 20]\n(per-position\nsensitivity)", "latent", w=2.4, h=1.2)
    box(ax, (5.5, 4.0),  "pam_gate ∈ [0,1]\n[B × 1]",                       "latent", w=2.4, h=1.0)
    box(ax, (10.0, 4.0), "context_logit\n[B × 1]",                          "latent", w=2.4, h=1.0)

    arrow(ax, (1.5, 5.20), (1.5, 4.60), COLOR_EDGE_F, lw=1.4, label="ReLU", label_pos=0.5)
    arrow(ax, (5.5, 5.20), (5.5, 4.50), COLOR_EDGE_F, lw=1.4)
    arrow(ax, (10.0, 5.20), (10.0, 4.50), COLOR_EDGE_F, lw=1.4)

    # ===================================================================
    # ROW 6 — thermo_logit (spacer pathway aggregation)
    # ===================================================================
    box(ax, (1.5, 2.3), "thermo_logit\n= Σᵢ Wᵢ · sᵢ  +  bias_eff\n"
                        "Wᵢ = −softplus(w_posᵢ) ≤ 0", "latent", w=3.4, h=1.3)

    arrow(ax, (1.5, 3.40), (1.5, 2.95), COLOR_EDGE_W, lw=2.0,
          label="× Wᵢ ≤ 0  +  bias", label_pos=0.5)

    # ===================================================================
    # ROW 7 — outcome (sigmoid + multiplicative PAM gate)
    # All three pathways (thermo, pam, context) converge here.
    # ===================================================================
    box(ax, (5.5, 0.6),
        "P(off-target activity)\n= pam_gate  ×  σ( thermo_logit  +  context_logit )",
        "outcome", w=6.2, h=1.0, fontsize=10.5)

    # thermo_logit -> outcome (blue diagonal from the left column)
    arrow(ax, (1.5, 1.65), (3.30, 0.95), COLOR_EDGE_W, lw=1.8)
    # pam_gate -> outcome (red diagonal from the middle column, multiplicative)
    arrow(ax, (5.5, 3.50), (5.5, 1.10), COLOR_EDGE_M, lw=2.0,
          label="× gate", label_pos=0.55, curvature=-0.15)
    # context_logit -> outcome (green diagonal from the right column)
    arrow(ax, (10.0, 3.50), (7.70, 0.95), COLOR_EDGE_C, lw=1.8,
          label="+ context", label_pos=0.45, curvature=-0.15)

    # ===================================================================
    # title + legend
    # ===================================================================
    ax.set_title("Neural SCM architecture — modular implementation of the causal DAG",
                 fontsize=13, fontweight="bold", pad=18)

    legend_handles = [
        FancyBboxPatch((0, 0), 1, 1, facecolor=COLOR_INPUT, edgecolor="black",
                       label="Input data (tensor)"),
        FancyBboxPatch((0, 0), 1, 1, facecolor=COLOR_MODULE, edgecolor="black",
                       label="Trainable module"),
        FancyBboxPatch((0, 0), 1, 1, facecolor=COLOR_LATENT, edgecolor="black",
                       label="Intermediate value"),
        FancyBboxPatch((0, 0), 1, 1, facecolor=COLOR_OUTCOME, edgecolor="black",
                       label="Outcome"),
    ]
    legend_edges = [
        plt.Line2D([0], [0], color=COLOR_EDGE_W, lw=2, label="Sign-constrained weighted path"),
        plt.Line2D([0], [0], color=COLOR_EDGE_M, lw=2, label="Multiplicative gate (PAM)"),
        plt.Line2D([0], [0], color=COLOR_EDGE_C, lw=2, label="Additive context shift"),
        plt.Line2D([0], [0], color=COLOR_EDGE_F, lw=2, label="Structural / functional"),
    ]
    leg1 = ax.legend(handles=legend_handles, loc="lower left",
                     bbox_to_anchor=(-0.02, -0.07), fontsize=9, frameon=True,
                     title="Nodes")
    ax.add_artist(leg1)
    ax.legend(handles=legend_edges, loc="lower right",
              bbox_to_anchor=(1.02, -0.07), fontsize=9, frameon=True,
              title="Edges")

    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
