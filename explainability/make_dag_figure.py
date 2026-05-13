"""Causal DAG figure for the thesis: the biological hypothesis
that the Neural SCM is asked to instantiate.

This is NOT the network architecture — it's the causal graph that
motivates the architecture. Drawing it separately makes the modelling
choices auditable: each arrow corresponds to a structural prior."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path("explainability/plots/thesis_fig0_dag.png")

# --- visual palette ----------------------------------------------------
COLOR_INPUT   = "#bcd6ec"   # light blue   — observed/exogenous
COLOR_LATENT  = "#ffd9a8"   # light orange — latent biological construct
COLOR_OUTCOME = "#f5a8a8"   # light red    — measured outcome
COLOR_EDGE_W  = "#1f3b6e"   # dark blue    — additive weighted edge
COLOR_EDGE_M  = "#a02020"   # dark red     — multiplicative gate
COLOR_EDGE_C  = "#2a7a2a"   # dark green   — context (additive shift)
COLOR_EDGE_F  = "#666666"   # gray         — structural / functional


def box(ax, xy, text, kind, w=1.7, h=0.8, fontsize=10.5):
    color = {"input": COLOR_INPUT, "latent": COLOR_LATENT, "outcome": COLOR_OUTCOME}[kind]
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
    rad = curvature
    connectionstyle = f"arc3,rad={rad}"
    a = FancyArrowPatch(
        src, dst,
        arrowstyle="-|>", mutation_scale=14,
        linestyle=style, linewidth=lw, color=color,
        connectionstyle=connectionstyle, zorder=2,
        shrinkA=18, shrinkB=18,
    )
    ax.add_patch(a)
    if label:
        # midpoint with offset
        mx = src[0] + (dst[0] - src[0]) * label_pos
        my = src[1] + (dst[1] - src[1]) * label_pos
        # offset perpendicular to direction for legibility
        ax.text(mx, my, label, ha="center", va="center",
                fontsize=9, color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                          edgecolor="none", alpha=0.85), zorder=5)


def main():
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xlim(-1, 12)
    ax.set_ylim(-0.5, 8.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # ---- layer 1: observed inputs -------------------------------------
    box(ax, (2.0, 7.5),  "sgRNA × off-target\n(20 nt spacer)",   "input", w=2.8)
    box(ax, (6.0, 7.5),  "off-target PAM\n(3 nt, NGG)",          "input", w=2.4)
    box(ax, (9.5, 7.5),  "GC composition\n(exogenous context)",  "input", w=2.8)

    # ---- layer 2: latent biological constructs ------------------------
    box(ax, (0.5, 5.2), "Non-seed\nmismatches\n(pos 1–8)",        "latent", w=2.0, h=1.1)
    box(ax, (3.0, 5.2), "Seed\nmismatches\n(pos 9–16)",           "latent", w=2.0, h=1.1)
    box(ax, (5.5, 5.2), "PAM-proximal\nmismatches\n(pos 17–20)",  "latent", w=2.0, h=1.1)
    box(ax, (8.0, 5.2), "PAM\ncompatibility",                     "latent", w=1.8, h=1.1)
    box(ax, (10.5, 5.2),"Context\noffset",                        "latent", w=1.8, h=1.1)

    # ---- layer 3: combined logit --------------------------------------
    box(ax, (4.5, 2.8), "Thermodynamic logit\n+ context",         "latent", w=3.2, h=0.9)

    # ---- layer 4: outcome ---------------------------------------------
    box(ax, (6.0, 0.4), "P(off-target activity)",                  "outcome", w=3.2, h=0.8, fontsize=11)

    # ---- edges, top -> middle -----------------------------------------
    for src, dst in [((2.0, 7.1), (0.5, 5.85)),
                     ((2.0, 7.1), (3.0, 5.85)),
                     ((2.0, 7.1), (5.5, 5.85))]:
        arrow(ax, src, dst, COLOR_EDGE_F, lw=1.4)
    arrow(ax, (6.0, 7.1), (8.0, 5.85), COLOR_EDGE_F, lw=1.4)
    arrow(ax, (9.5, 7.1), (10.5, 5.85), COLOR_EDGE_F, lw=1.4)

    # ---- edges, middle -> thermo logit (the three weighted region paths)
    arrow(ax, (0.5, 4.55), (4.5, 3.30), COLOR_EDGE_W,
          label="W_d ≤ 0", label_pos=0.45, curvature=0.05)
    arrow(ax, (3.0, 4.55), (4.5, 3.30), COLOR_EDGE_W,
          label="W_s ≤ 0", label_pos=0.55)
    arrow(ax, (5.5, 4.55), (4.5, 3.30), COLOR_EDGE_W,
          label="W_p ≤ 0", label_pos=0.45, curvature=-0.05)

    # context -> thermo logit (additive offset)
    arrow(ax, (10.5, 4.55), (5.7, 3.05), COLOR_EDGE_C,
          label="+", label_pos=0.55, curvature=-0.18)

    # ---- thermo logit -> outcome (sigmoid + PAM gate) ------------------
    arrow(ax, (4.5, 2.35), (5.5, 0.85), COLOR_EDGE_F,
          label="sigmoid(·)", label_pos=0.5)
    # PAM gate (multiplicative) directly modulates the outcome
    arrow(ax, (8.0, 4.55), (6.5, 0.85), COLOR_EDGE_M,
          label="× gate ∈ [0,1]", label_pos=0.55, curvature=-0.15)

    # # ---- structural priors annotation ---------------------------------
    # ax.text(3.0, 6.95, "Hard prior:  | W_s |  ≥  | W_d |   (seed dominance)",
    #         ha="center", va="center", fontsize=9, style="italic",
    #         color=COLOR_EDGE_W,
    #         bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
    #                   edgecolor=COLOR_EDGE_W, alpha=0.9), zorder=6)

    # ---- title and legend ---------------------------------------------
    ax.set_title("Causal hypothesis underlying the Neural SCM\n"
                 "(structural priors encoded into the architecture)",
                 fontsize=13, fontweight="bold", pad=20)

    legend_handles = [
        FancyBboxPatch((0, 0), 1, 1, facecolor=COLOR_INPUT, edgecolor="black",
                       label="Observed / exogenous"),
        FancyBboxPatch((0, 0), 1, 1, facecolor=COLOR_LATENT, edgecolor="black",
                       label="Latent biological construct"),
        FancyBboxPatch((0, 0), 1, 1, facecolor=COLOR_OUTCOME, edgecolor="black",
                       label="Observed outcome"),
    ]
    legend_edges = [
        plt.Line2D([0], [0], color=COLOR_EDGE_W, lw=2, label="Sign-constrained weighted edge"),
        plt.Line2D([0], [0], color=COLOR_EDGE_M, lw=2, label="Multiplicative gate (PAM)"),
        plt.Line2D([0], [0], color=COLOR_EDGE_C, lw=2, label="Additive context shift"),
        plt.Line2D([0], [0], color=COLOR_EDGE_F, lw=2, label="Structural / functional"),
    ]
    leg1 = ax.legend(handles=legend_handles, loc="lower left",
                     bbox_to_anchor=(-0.02, -0.05), fontsize=9, frameon=True,
                     title="Nodes")
    ax.add_artist(leg1)
    ax.legend(handles=legend_edges, loc="lower right",
              bbox_to_anchor=(1.02, -0.05), fontsize=9, frameon=True,
              title="Edges")

    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
