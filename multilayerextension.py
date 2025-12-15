#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 14 23:57:03 2025

@author: dushathangarajah
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

def _curved_edge(ax, p1, p2, lw=2.0, alpha=0.8, ls='-', rad=0.15, arrow=False):
    """Draw a curved edge between p1 and p2."""
    style = "-|>" if arrow else "-"
    patch = FancyArrowPatch(
        p1, p2,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle=style,
        mutation_scale=10 if arrow else 1,
        linewidth=lw,
        linestyle=ls,
        alpha=alpha,
        color="black"
    )
    ax.add_patch(patch)

def plot_multilayer_extension(
    fin_nodes=None,
    macro_nodes=None,
    fin_edges=None,
    macro_edges=None,
    inter_edges=None,
    title="Two-layer contagion extension (stylised multilayer network)",
    savepath="multilayer_extension.png",
    show=True
):
    """
    A cleaner multilayer schematic:
    - Layer panels (shaded boxes)
    - Weighted/curved edges
    - Optional directed inter-layer coupling
    - Proper legend
    """

    # --- defaults ---
    if fin_nodes is None:
        fin_nodes = ["JPM", "BAC", "WFC", "GS", "MS", "C"]

    # macro layer nodes can be different (recommended for the extension)
    if macro_nodes is None:
        macro_nodes = ["SL", "TE", "ROW"]  # Sri Lanka, Tamil region, Rest of World

    # intra-layer edges with weights
    if fin_edges is None:
        fin_edges = [
            ("JPM","BAC", 0.8), ("BAC","WFC", 0.9), ("WFC","C", 0.7),
            ("JPM","GS", 0.6), ("GS","MS", 0.8), ("MS","C", 0.5)
        ]

    if macro_edges is None:
        macro_edges = [
            ("ROW","SL", 0.8), ("SL","TE", 0.6)
        ]

    # inter-layer mapping finance -> macro (weights = “exposure/intensity”)
    if inter_edges is None:
        inter_edges = [
            ("JPM","SL", 0.7),
            ("BAC","SL", 0.6),
            ("WFC","SL", 0.4),
            ("GS","ROW", 0.5),
            ("MS","ROW", 0.4),
            ("C","SL", 0.5),
            ("C","TE", 0.3),
        ]

    # --- layout (hand-built, stable, no random spring-layout) ---
    xf = np.linspace(0.08, 0.92, len(fin_nodes))
    xm = np.linspace(0.15, 0.85, len(macro_nodes))

    y_fin = 0.72
    y_mac = 0.25

    pos_fin = {n: (xf[i], y_fin) for i, n in enumerate(fin_nodes)}
    pos_mac = {n: (xm[i], y_mac) for i, n in enumerate(macro_nodes)}

    # --- figure ---
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.set_axis_off()

    # Layer panels
    ax.add_patch(Rectangle((0.03, 0.56), 0.94, 0.32, linewidth=1.2, edgecolor="black", facecolor="0.95", zorder=0))
    ax.add_patch(Rectangle((0.03, 0.08), 0.94, 0.32, linewidth=1.2, edgecolor="black", facecolor="0.97", zorder=0))

    ax.text(0.04, 0.86, "Layer 1: Financial network", fontsize=11, weight="bold")
    ax.text(0.04, 0.38, "Layer 2: Macro / regional network", fontsize=11, weight="bold")

    # --- edges: financial ---
    for u, v, w in fin_edges:
        p1, p2 = pos_fin[u], pos_fin[v]
        _curved_edge(ax, p1, p2, lw=1.0 + 4.0*w, alpha=0.85, ls='-', rad=0.05, arrow=False)

    # --- edges: macro (dashed) ---
    for u, v, w in macro_edges:
        p1, p2 = pos_mac[u], pos_mac[v]
        _curved_edge(ax, p1, p2, lw=1.0 + 4.0*w, alpha=0.85, ls='--', rad=0.10, arrow=True)

    # --- inter-layer edges (dotted + curved) ---
    for u, v, w in inter_edges:
        p1, p2 = pos_fin[u], pos_mac[v]
        # alternate curvature direction for readability
        rad = 0.18 if (hash(u) % 2 == 0) else -0.18
        _curved_edge(ax, p1, p2, lw=0.6 + 2.5*w, alpha=0.55, ls=':', rad=rad, arrow=False)

    # --- nodes ---
    def _draw_nodes(nodes, pos, y, node_size=1200):
        for n in nodes:
            x, yy = pos[n]
            ax.scatter([x], [yy], s=node_size, edgecolors="black", linewidths=1.2, zorder=3)
            ax.text(x, yy, n, ha="center", va="center", fontsize=10, zorder=4)

    _draw_nodes(fin_nodes, pos_fin, y_fin, node_size=1400)
    _draw_nodes(macro_nodes, pos_mac, y_mac, node_size=1400)

    # --- title + legend ---
    ax.set_title(title, fontsize=14, pad=10)

    legend_elems = [
        Line2D([0], [0], color="black", lw=2.5, ls='-', label="Intra-layer (financial)"),
        Line2D([0], [0], color="black", lw=2.5, ls='--', label="Intra-layer (macro)"),
        Line2D([0], [0], color="black", lw=2.0, ls=':', label="Inter-layer coupling"),
    ]
    ax.legend(handles=legend_elems, loc="upper right", frameon=True)

    plt.tight_layout()
    fig.savefig(savepath, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved multilayer figure to: {savepath}")

if __name__ == "__main__":
    plot_multilayer_extension()
