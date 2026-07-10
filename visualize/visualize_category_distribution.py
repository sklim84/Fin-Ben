"""
Figure 1 (dataset category distribution) — redesigned palette.

Addresses reviewer ea77-W1: the original sunburst colored slices by *count*
(color_continuous_scale='RdBu'), so sub-categories of different domains with
similar counts received near-identical reds and became indistinguishable.

This version assigns each of the three domains a distinct color *family*
(Knowledge = blues, Reasoning = greens, Toxicity = oranges) and shades the
sub-categories within each family, so domain membership is immediately legible.

Output: visualize/stats_simple_v2.pdf and .png
Data:   _datasets/0_integration/_results/category_stats_simple.csv
        (same aggregated top-4-plus-Others table as the original figure)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Wedge  # noqa: F401 (kept for reference)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CSV = REPO / "_datasets" / "0_integration" / "_results" / "category_stats_simple.csv"

# Domain order + display name + colormap (distinct color families)
DOMAINS = [
    ("Financial <br> Knowledge", "Financial\nKnowledge", plt.cm.Blues),
    ("Financial <br> Reasoning", "Financial\nReasoning", plt.cm.Greens),
    ("Financial <br> Toxicity",  "Financial\nToxicity",  plt.cm.Oranges),
]
DOMAIN_BASE = {  # inner-ring base color (light enough for dark text)
    "Financial <br> Knowledge": plt.cm.Blues(0.62),
    "Financial <br> Reasoning": plt.cm.Greens(0.62),
    "Financial <br> Toxicity":  plt.cm.Oranges(0.62),
}


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
    for main_key, disp, cmap in DOMAINS:
        sub = df[df["Main Category"] == main_key].copy()
        sub = sub.sort_values("Count", ascending=False).reset_index(drop=True)
        total = int(sub["Count"].sum())
        inner_sizes.append(total)
        inner_colors.append(DOMAIN_BASE[main_key])
        inner_labels.append(f"{SHORT[main_key]}\n({total})")

        # shades: light-to-lighter within the family so a single dark text
        # colour reads on every wedge (keeps labels visually uniform).
        shades = np.linspace(0.55, 0.30, len(sub))
        for (_, row), s in zip(sub.iterrows(), shades):
            outer_sizes.append(int(row["Count"]))
            outer_colors.append(cmap(s))
            name = " ".join(clean(row["Sub Category"]).replace("\n", " ").split())
            outer_labels.append(f"{name} ({int(row['Count'])})")

    import textwrap

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(aspect="equal"))
    W_OUT, W_IN = 0.40, 0.60  # inner band reaches near the center (sunburst-like)
    R_OUT_MID = 1.0 - W_OUT / 2.0            # mid-radius of outer band
    WIDE_DEG = 22.0                          # >= this: horizontal label; else radial

    LABEL_COLOR = "0.12"   # uniform dark colour for all sub-category labels

    # outer ring — sub-categories
    wedges, _ = ax.pie(
        outer_sizes, radius=1.0, colors=outer_colors,
        startangle=90, counterclock=False,
        wedgeprops=dict(width=W_OUT, edgecolor="white", linewidth=1.3),
    )
    # inner ring — domains (label centered on the band, white bold)
    ax.pie(
        inner_sizes, radius=1.0 - W_OUT, colors=inner_colors,
        labels=inner_labels, labeldistance=0.66, rotatelabels=False,
        startangle=90, counterclock=False,
        wedgeprops=dict(width=W_IN, edgecolor="white", linewidth=2.0),
        textprops=dict(fontsize=18, fontweight="bold", ha="center", va="center",
                       color=LABEL_COLOR),
    )

    # sub-category names INSIDE each outer wedge (sunburst style):
    #   wide wedges -> horizontal (wrapped); thin wedges -> radial (rotated).
    for w, lbl in zip(wedges, outer_labels):
        mid = (w.theta1 + w.theta2) / 2.0
        width = w.theta2 - w.theta1
        a = np.deg2rad(mid)
        x, y = R_OUT_MID * np.cos(a), R_OUT_MID * np.sin(a)
        if width >= WIDE_DEG:
            wrapped = "\n".join(textwrap.wrap(lbl, width=16))
            ax.text(x, y, wrapped, ha="center", va="center",
                    fontsize=13, color=LABEL_COLOR, linespacing=1.0)
        else:
            # thin wedge: radial text INSIDE the wedge (reads along the radius),
            # wrapped to short lines and shrunk so it fits within the band.
            rot = mid - 180 if 90 < (mid % 360) < 270 else mid
            fs = 10 if width >= 10 else 9
            wrapped = "\n".join(textwrap.wrap(lbl, width=13))
            ax.text(x, y, wrapped, ha="center", va="center", rotation=rot,
                    rotation_mode="anchor", fontsize=fs, color=LABEL_COLOR,
                    linespacing=0.9)

    ax.set(aspect="equal")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = HERE / f"stats_simple_v2.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"saved: {out}")


if __name__ == "__main__":
    main()
