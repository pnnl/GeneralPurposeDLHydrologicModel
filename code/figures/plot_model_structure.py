#!/usr/bin/env python3
"""Model-structure infographic matching Figure 2 visual language."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch

OUT = Path(__file__).resolve().parents[2] / "outputs" / "figures" / "model_structure.png"

NAVY = "#16325c"
GREEN = "#1f6b3a"
TEAL = "#156e75"
PURPLE = "#6a2f9a"
EDGE = "#2b6cb0"
INK = "#24303a"
MUTED = "#5b6770"


def rbox(ax, x, y, w, h, ec, lw=2.15, fc="white", rad=0.09, z=1):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.01,rounding_size={rad}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z,
    ))


def ndot(ax, x, y, n, color):
    ax.add_patch(Circle((x, y), 0.20, facecolor=color, edgecolor=color, zorder=8))
    ax.text(x, y - 0.01, str(n), ha="center", va="center",
            fontsize=11.5, fontweight="bold", color="white", zorder=9)


def darrow(ax, x, y0, y1, color, label):
    ax.annotate(
        "", xy=(x, y1), xytext=(x, y0),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=2.3, mutation_scale=15),
        zorder=6,
    )
    ax.text(x + 0.14, (y0 + y1) / 2.0, label, color=color, fontsize=9.4,
            fontstyle="italic", va="center")


def net(ax, ox, oy, s=1.0, target="e"):
    nodes = {
        "a": (0.10, 1.28), "b": (0.52, 1.48), "c": (0.32, 0.92),
        "d": (0.92, 1.18), "e": (0.70, 0.68), "f": (1.18, 0.50),
        "g": (0.38, 0.32), "h": (0.82, 0.12),
    }
    edges = [("a", "c"), ("b", "c"), ("c", "e"), ("d", "e"),
             ("e", "h"), ("g", "h"), ("f", "h")]
    for u, v in edges:
        x1, y1 = nodes[u]
        x2, y2 = nodes[v]
        ax.annotate(
            "",
            xy=(ox + x2 * s, oy + y2 * s),
            xytext=(ox + x1 * s, oy + y1 * s),
            arrowprops=dict(arrowstyle="-|>", color=EDGE, lw=1.2,
                            mutation_scale=8, shrinkA=4, shrinkB=4),
            zorder=3,
        )
    for name, (x, y) in nodes.items():
        xx, yy = ox + x * s, oy + y * s
        if name == target:
            ax.add_patch(Circle((xx, yy), 0.078, facecolor=GREEN,
                                edgecolor=GREEN, zorder=5))
        else:
            ax.add_patch(Circle((xx, yy), 0.068, facecolor="white",
                                edgecolor=NAVY, lw=1.1, zorder=4))


def spark_weather(ax, x0, y0):
    rng = np.random.default_rng(7)
    n = 16
    p = rng.gamma(1.1, 0.16, size=n)
    p[6], p[7] = 0.92, 0.62
    xs = x0 + np.arange(n) * 0.072
    ax.bar(xs, p * 0.42, width=0.055, bottom=y0, color="#4a90c8",
           align="center", zorder=4, linewidth=0)
    t = np.linspace(0, 1, n)
    ax.plot(xs, y0 + 0.58 + 0.09 * np.sin(2 * np.pi * t * 1.1),
            color="#c0392b", lw=1.15, zorder=5)


def hydro_obs(ax, xb, yb, w, h):
    ins = ax.inset_axes([xb, yb, w, h], transform=ax.transData)
    t = np.linspace(0, 10, 220)
    obs = 0.38 + 0.10 * np.sin(t) + 0.07 * np.sin(2.1 * t + 0.5)
    bump = 0.26 * np.exp(-((t - 6.0) ** 2) / 1.6)
    obs = obs + bump
    fc = np.where(t < 5.3, np.nan, obs * 0.92 + 0.04)
    ins.plot(t, obs, color="#222", lw=1.2)
    ins.plot(t, fc, color="#222", lw=1.15, ls="--")
    ins.axvline(5.3, color="#999", lw=0.7, ls=":")
    ins.set_xlim(0, 10)
    ins.set_ylim(0.12, 0.95)
    ins.set_xticks([])
    ins.set_yticks([])
    ins.set_ylabel("Streamflow", fontsize=6.5, color=MUTED, labelpad=1)
    ins.set_xlabel("Time", fontsize=6.5, color=MUTED, labelpad=1)
    for s in ins.spines.values():
        s.set_color("#777")
        s.set_linewidth(0.7)


def hydro_mix(ax, xb, yb, w, h):
    ins = ax.inset_axes([xb, yb, w, h], transform=ax.transData)
    t = np.linspace(0, 10, 220)
    med = 0.40 + 0.08 * np.sin(0.85 * t) + 0.16 * np.exp(-((t - 5.6) ** 2) / 1.8)
    p90 = med + 0.09
    p95 = med + 0.15
    p99 = med + 0.24
    lo = np.maximum(0.08, med - 0.07)
    ins.fill_between(t, lo, p99, color="#d3b3ea", alpha=0.7, lw=0)
    ins.fill_between(t, lo, p95, color="#b888dc", alpha=0.45, lw=0)
    ins.fill_between(t, lo, p90, color="#9660c8", alpha=0.28, lw=0)
    ins.plot(t, med, color=PURPLE, lw=1.4)
    ins.axvline(5.3, color="#999", lw=0.7, ls=":")
    ins.set_xlim(0, 10)
    ins.set_ylim(0.05, 0.98)
    ins.set_xticks([])
    ins.set_yticks([])
    ins.set_ylabel("Streamflow", fontsize=6.5, color=MUTED, labelpad=1)
    ins.set_xlabel("Time", fontsize=6.5, color=MUTED, labelpad=1)
    for s in ins.spines.values():
        s.set_color("#777")
        s.set_linewidth(0.7)


def main():
    fig, ax = plt.subplots(figsize=(13.6, 7.35), dpi=250)
    ax.set_xlim(0, 13.6)
    ax.set_ylim(0, 7.35)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # LEFT PANEL
    rbox(ax, 0.20, 0.22, 3.95, 6.90, NAVY, lw=2.2, rad=0.11)
    ax.text(2.18, 6.78, "Shared hydrologic\nrepresentation",
            ha="center", va="top", fontsize=13.2, fontweight="bold",
            color=NAVY, linespacing=1.15)
    ax.text(2.18, 6.18, "built on the MERIT Hydro network",
            ha="center", va="top", fontsize=8.0, color=MUTED, style="italic")

    net(ax, 0.85, 3.12, s=1.75)

    ax.text(0.42, 3.28, "Daily weather  (8 variables)",
            fontsize=7.8, color=NAVY, fontweight="medium")
    spark_weather(ax, 0.50, 2.58)

    ax.text(0.42, 2.40, "Static catchment attributes  (40)",
            fontsize=7.8, color=NAVY, fontweight="medium")
    chips = [("terrain", 0.42, 1.90), ("soils", 2.18, 1.90),
             ("vegetation", 0.42, 1.48), ("climate indices", 2.18, 1.48)]
    for lab, x, y in chips:
        rbox(ax, x, y, 1.55, 0.34, "#b7c0c8", lw=1.0, fc="#f5f7f9", rad=0.06)
        ax.text(x + 0.775, y + 0.17, lab, ha="center", va="center",
                fontsize=7.5, color=NAVY)

    rbox(ax, 0.40, 0.38, 3.55, 0.92, "#c5ced6", lw=1.05, fc="#fbfcfd", rad=0.06)
    ax.add_patch(Circle((0.68, 1.02), 0.075, facecolor=GREEN, edgecolor=GREEN, zorder=5))
    ax.text(0.86, 1.02, "Target catchment", va="center", fontsize=7.4, color=INK)
    ax.add_patch(Circle((0.68, 0.62), 0.068, facecolor="white", edgecolor=NAVY, lw=1.1, zorder=5))
    ax.text(0.86, 0.62, "Neighbor reach", va="center", fontsize=7.4, color=INK)
    ax.annotate("", xy=(2.95, 0.82), xytext=(2.35, 0.82),
                arrowprops=dict(arrowstyle="-|>", color=EDGE, lw=1.25, mutation_scale=8))
    ax.text(3.05, 0.82, "Directed\nriver link", va="center", fontsize=7.3, color=INK)

    ax.annotate(
        "", xy=(4.42, 5.85), xytext=(4.15, 5.85),
        arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2.5, mutation_scale=15),
    )

    # CARD 1
    rbox(ax, 4.42, 4.95, 8.95, 2.15, GREEN)
    ndot(ax, 4.78, 6.72, 1, GREEN)
    ax.text(5.10, 6.72, "LSTM temporal encoder", fontsize=12.6,
            fontweight="bold", color=GREEN, va="center")
    ax.text(5.10, 6.40, "(365-day warmup, then 180-day prediction)",
            fontsize=8.4, color=GREEN, style="italic", va="center")
    ax.text(4.68, 5.55,
            "Each catchment reads its own weather\n"
            "history and static attributes.\n"
            "512 hidden units, 3 layers.",
            fontsize=8.6, color=INK, va="center")
    hydro_obs(ax, 10.55, 5.12, 2.55, 1.68)

    darrow(ax, 8.90, 4.95, 4.62, TEAL, "graph update")

    # CARD 2
    rbox(ax, 4.42, 2.48, 8.95, 2.12, TEAL)
    ndot(ax, 4.78, 4.22, 2, TEAL)
    ax.text(5.10, 4.22, "Directed graph attention", fontsize=12.6,
            fontweight="bold", color=TEAL, va="center")
    ax.text(5.10, 3.90, "(upstream and downstream pathways, learned gate)",
            fontsize=8.4, color=TEAL, style="italic", va="center")
    net(ax, 5.15, 2.52, s=0.88)
    ax.text(6.85, 3.20,
            "3 layers, 256 hidden, 4-head attention.\n"
            "Neighbors exchange state along the river.\n"
            "A gate mixes the two directions.",
            fontsize=8.6, color=INK, va="center")
    rbox(ax, 10.55, 3.48, 2.55, 0.44, TEAL, lw=1.15, fc="#e7f4f5", rad=0.06)
    ax.text(11.82, 3.70, "upstream  to  downstream", ha="center", va="center",
            fontsize=7.3, color=TEAL)
    rbox(ax, 10.55, 2.82, 2.55, 0.44, TEAL, lw=1.15, fc="#e7f4f5", rad=0.06)
    ax.text(11.82, 3.04, "downstream  to  upstream", ha="center", va="center",
            fontsize=7.3, color=TEAL)

    darrow(ax, 8.90, 2.48, 2.15, PURPLE, "output heads")

    # CARD 3
    rbox(ax, 4.42, 0.22, 8.95, 1.91, PURPLE)
    ndot(ax, 4.78, 1.75, 3, PURPLE)
    ax.text(5.10, 1.75, "Two heads on the same state", fontsize=12.6,
            fontweight="bold", color=PURPLE, va="center")
    ax.text(5.10, 1.44, "(point forecast and LogNormal mixture envelope)",
            fontsize=8.4, color=PURPLE, style="italic", va="center")
    ax.text(4.68, 0.78,
            "Local LSTM state is concatenated with the\n"
            "post-graph state. Softplus gives non-negative\n"
            "specific discharge. The K = 3 mixture supplies P10 to P99.",
            fontsize=8.5, color=INK, va="center")
    hydro_mix(ax, 10.55, 0.38, 2.55, 1.50)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=270, bbox_inches="tight", facecolor="white", pad_inches=0.06)
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
