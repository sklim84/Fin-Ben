"""
Figure 1 (dataset category distribution) — redesigned palette.

Addresses reviewer ea77-W1: the original sunburst colored slices by *count*
(color_continuous_scale='RdBu'), so sub-categories of different domains with
similar counts received near-identical reds and became indistinguishable.

This version keeps each of the three domains visually distinct, but draws all
colors from the shared **viridis** scale used by the paper's other figures:
each domain occupies a separate, well-separated viridis band (Knowledge = the
blue/purple region, Reasoning = the teal/green region, Toxicity = the
green/yellow region), and sub-categories are shaded within the band. Because
viridis spans dark (purple) to light (yellow), each label's color (black or
white) is chosen from the fill luminance so text stays legible on every wedge.

Output: visualize/stats_simple_v2.pdf and .png
Data:   _datasets/0_integration/_results/category_stats_simple.csv
        (same aggregated top-4-plus-Others table as the original figure)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CSV = REPO / "_datasets" / "0_integration" / "_results" / "category_stats_simple.csv"

VIRIDIS = plt.cm.viridis

# Domain order + display name + viridis band (lo, hi). The three bands are
# well-separated so the domains remain immediately distinguishable while sharing
# the paper-wide viridis palette; sub-categories are shaded within the band.
DOMAINS = [
    ("Financial <br> Knowledge", "Financial\nKnowledge", (0.20, 0.40)),
    ("Financial <br> Reasoning", "Financial\nReasoning", (0.48, 0.66)),
    ("Financial <br> Toxicity",  "Financial\nToxicity",  (0.74, 0.92)),
]
DOMAIN_BASE = {  # inner-ring base = middle of each domain's viridis band
    "Financial <br> Knowledge": VIRIDIS(0.30),
    "Financial <br> Reasoning": VIRIDIS(0.57),
    "Financial <br> Toxicity":  VIRIDIS(0.83),
}


def _text_on(color):
    """Return black or white — whichever reads on the given fill color."""
    r, g, b = color[:3]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if lum < 0.55 else "0.12"


def clean(lbl: str) -> str:
    return lbl.replace("<br>", "\n").replace("  ", " ").strip()


def main():
    df = pd.read_csv(CSV)
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]

    inner_sizes, inner_colors, inner_labels = [], [], []
    outer_sizes, outer_colors, outer_labels = [], [], []

    SHORT = {  # short domain word for the inner-ring label (all are "Financial")
        "Financial <br> Knowledge": "Knowledge",
        "Financial <br> Reasoning": "Reasoning",
        "Financial <br> Toxicity": "Toxicity",
    }
    for main_key, disp, (lo, hi) in DOMAINS:
        sub = df[df["Main Category"] == main_key].copy()
        sub = sub.sort_values("Count", ascending=False).reset_index(drop=True)
        total = int(sub["Count"].sum())
        inner_sizes.append(total)
        inner_colors.append(DOMAIN_BASE[main_key])
        inner_labels.append(f"{SHORT[main_key]}\n({total})")

        # Shade sub-categories across the domain's viridis band (largest count ->
        # one end, smallest -> the other) so wedges are distinguishable.
        shades = np.linspace(lo, hi, len(sub))
        for (_, row), s in zip(sub.iterrows(), shades):
            outer_sizes.append(int(row["Count"]))
            outer_colors.append(VIRIDIS(s))
            name = " ".join(clean(row["Sub Category"]).replace("\n", " ").split())
            outer_labels.append(f"{name} ({int(row['Count'])})")

    import textwrap

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(aspect="equal"))
    W_OUT, W_IN = 0.40, 0.60  # inner band reaches near the center (sunburst-like)
    R_OUT_MID = 1.0 - W_OUT / 2.0            # mid-radius of outer band
    WIDE_DEG = 22.0                          # >= this: horizontal label; else radial

    # outer ring — sub-categories
    wedges, _ = ax.pie(
        outer_sizes, radius=1.0, colors=outer_colors,
        startangle=90, counterclock=False,
        wedgeprops=dict(width=W_OUT, edgecolor="white", linewidth=1.3),
    )
    # inner ring — domains (label centered on the band, bold; color adapts below)
    _, inner_texts = ax.pie(
        inner_sizes, radius=1.0 - W_OUT, colors=inner_colors,
        labels=inner_labels, labeldistance=0.66, rotatelabels=False,
        startangle=90, counterclock=False,
        wedgeprops=dict(width=W_IN, edgecolor="white", linewidth=2.0),
        textprops=dict(fontsize=21, fontweight="bold", ha="center", va="center"),
    )
    for t, col in zip(inner_texts, inner_colors):
        t.set_color(_text_on(col))

    # sub-category names INSIDE each outer wedge (sunburst style):
    #   wide wedges -> horizontal (wrapped); thin wedges -> radial (rotated).
    #   label color adapts to the wedge fill so it reads on dark and light bands.
    for w, lbl in zip(wedges, outer_labels):
        mid = (w.theta1 + w.theta2) / 2.0
        width = w.theta2 - w.theta1
        a = np.deg2rad(mid)
        x, y = R_OUT_MID * np.cos(a), R_OUT_MID * np.sin(a)
        tcol = _text_on(w.get_facecolor())
        if width >= WIDE_DEG:
            wrapped = "\n".join(textwrap.wrap(lbl, width=16))
            ax.text(x, y, wrapped, ha="center", va="center",
                    fontsize=16, color=tcol, linespacing=1.0)
        else:
            # thin wedge: radial text INSIDE the wedge (reads along the radius),
            # wrapped to short lines and shrunk so it fits within the band.
            rot = mid - 180 if 90 < (mid % 360) < 270 else mid
            fs = 12 if width >= 10 else 11
            wrapped = "\n".join(textwrap.wrap(lbl, width=13))
            ax.text(x, y, wrapped, ha="center", va="center", rotation=rot,
                    rotation_mode="anchor", fontsize=fs, color=tcol,
                    linespacing=0.9)

    ax.set(aspect="equal")
    # Tight limits so the ring fills the frame (nothing is drawn beyond r=1),
    # which maximizes the on-page size of the in-wedge labels at column width.
    ax.set_xlim(-1.03, 1.03)
    ax.set_ylim(-1.03, 1.03)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = HERE / f"stats_simple_v2.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"saved: {out}")


if __name__ == "__main__":
    main()
