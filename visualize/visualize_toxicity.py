#!/usr/bin/env python3
"""
Toxicity Evaluation Results Visualization Script

Creates:
1. Radar charts for each category with A-G checklist items
2. Radar chart for attack methods with mean scores
3. Bar charts for toxicity_levels and score_distribution
"""

import json
import os
import csv
import textwrap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import platform


class _SnsShim:
    @staticmethod
    def color_palette(name, n):
        if name == "husl":
            return [plt.cm.hsv(i / n) for i in range(n)]
        if name == "viridis":
            return [plt.cm.viridis(i / max(n - 1, 1)) for i in range(n)]
        return [plt.cm.tab10(i % 10) for i in range(n)]


sns = _SnsShim()


# Set up Korean font
def setup_korean_font():
    """Configure matplotlib to display Korean characters."""
    if platform.system() == "Linux":
        font_paths = [
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        ]
        for path in font_paths:
            if os.path.exists(path):
                font_manager.fontManager.addfont(path)
                plt.rcParams["font.family"] = font_manager.FontProperties(
                    fname=path
                ).get_name()
                break
        else:
            plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False


setup_korean_font()

# Serif style matched to the paper's other vector figures; sized for placement
# so the on-page text stays ~8pt.
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.6,
    "legend.frameon": False,
    "pdf.fonttype": 42,
})

# JSON file paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(
    SCRIPT_DIR, "..", "eval", "_results", "3_fin_toxicity"
)
JSON_FILES = [
    "3_fin_toxicity_google_gemini-3-flash-preview_eval_stats.json",
    "3_fin_toxicity_gpt-4.1_eval_stats.json",
    "3_fin_toxicity_claude-haiku-4-5_eval_stats.json",
    "3_fin_toxicity_claude-sonnet-4-5_eval_stats.json",
    "3_fin_toxicity_Ministral-3-14B-Instruct-2512_eval_stats.json",
    "3_fin_toxicity_Mistral-Small-3.2-24B-Instruct-2506_eval_stats.json",
    "3_fin_toxicity_gpt-oss-20b_eval_stats.json",
    "3_fin_toxicity_EXAONE-4.0-1.2B_eval_stats.json",
    "3_fin_toxicity_gpt-5.2_eval_stats.json",
    "3_fin_toxicity_gpt-5.2_reasoning_eval_stats.json",
    "3_fin_toxicity_gpt-5-mini_eval_stats.json",
    "3_fin_toxicity_gpt-5-mini_reasoning_eval_stats.json",
]

# Short model names for display
MODEL_SHORT_NAMES = {
    "google_gemini-3-flash-preview": "Gemini 3 Flash",
    "gpt-4.1": "GPT-4.1",
    "claude-haiku-4-5": "Claude Haiku 4.5",
    "claude-sonnet-4-5": "Claude Sonnet 4.5",
    "Ministral-3-14B-Instruct-2512": "Ministral 3 14B",
    "Mistral-Small-3.2-24B-Instruct-2506": "Mistral Small 24B",
    "gpt-oss-20b": "GPT-OSS 20B",
    "EXAONE-4.0-1.2B": "EXAONE 4.0 1.2B",
    "gpt-5.2": "GPT-5.2",
    "gpt-5.2_reasoning": "GPT-5.2 (Reas)",
    "gpt-5-mini": "GPT-5 Mini",
    "gpt-5-mini_reasoning": "GPT-5 Mini (Reas)",
}

# Colors for each model
MODEL_COLORS = sns.color_palette("husl", len(JSON_FILES))

# Checklist item labels (A-G)
CHECKLIST_LABELS = [
    "A_명시적_거절",
    "B_법적_경고",
    "C_출처없는_주장",
    "D_실행가능_방법",
    "E_긴급성_공포",
    "F_정치편향",
    "G_수락의사",
]

CHECKLIST_SHORT = [
    "Explicit\nRefusal",
    "Legal\nWarning",
    "Uncited\nClaim",
    "Actionable\nMethod",
    "Urgency/Fear",
    "Political\nBias",
    "Acceptance",
]

# Single-letter axis labels; the caption maps A-G to the full dimension names.
CHECKLIST_ABBR = ["A", "B", "C", "D", "E", "F", "G"]


def load_all_data():
    """Load all JSON files and return data dict."""
    data = {}
    for fname in JSON_FILES:
        fpath = os.path.join(BASE_DIR, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            d = json.load(f)
            model_name = d.get("model", fname.replace(".json", ""))
            data[model_name] = d
    return data


def create_radar_chart(ax, labels, values_dict, title, show_legend=True, show_labels=True):
    """Create a radar chart on the given axes."""
    num_vars = len(labels)
    # 각도 계산 (라디안)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles_with_closure = angles + angles[:1]

    ax.set_theta_offset(np.pi / 2)  # 시작 위치를 12시 방향으로
    ax.set_theta_direction(-1)  # 시계 방향으로 진행
    ax.set_xticks(angles)

    # Axis (A-G) labels are drawn on the first panel only; all panels share the
    # same axis layout, so repeating them would only shrink the circles.
    ax.set_xticklabels(labels if show_labels else [""] * len(labels), fontsize=7)

    # --- 라벨 회전 로직 추가 ---
    # tick_params의 pad를 조절해 원과의 간격을 띄웁니다.
    ax.tick_params(axis="x", pad=2)

    labels_objs = ax.get_xticklabels()
    for i, label in enumerate(labels_objs):
        # 현재 항목의 각도 (라디안 -> 도)
        angle_rad = angles[i]
        angle_deg = np.rad2deg(angle_rad)

        # 텍스트가 거꾸로 뒤집히지 않게 처리 (하단 부분은 180도 회전)
        if 90 < angle_deg < 270:
            rotation = 180 - angle_deg
        else:
            rotation = -angle_deg

        label.set_rotation(rotation)
        # 라벨의 정렬 방향을 중앙으로 설정하여 원을 따라 깔끔하게 배치
        label.set_verticalalignment("center")
        label.set_horizontalalignment("center")
    # --------------------------

    lines = []
    for idx, (model_name, values) in enumerate(values_dict.items()):
        vals = values + values[:1]
        short_name = MODEL_SHORT_NAMES.get(model_name, model_name)
        (line,) = ax.plot(
            angles_with_closure,
            vals,
            "-",
            linewidth=0.9,
            label=short_name,
            color=MODEL_COLORS[idx % len(MODEL_COLORS)],
        )
        lines.append(line)
        ax.fill(
            angles_with_closure,
            vals,
            alpha=0.1,
            color=MODEL_COLORS[idx % len(MODEL_COLORS)],
        )

    wrapped_title = (
        "Fear-\nmongering" if title == "Fearmongering" else textwrap.fill(title, 18)
    )
    ax.set_title(wrapped_title, fontsize=8, pad=6)

    return lines


CATEGORY_TRANSLATION = {
    "불법 부정행위 조언": "Illegal/Unethical Advice",
    "허위정보 생성": "Misinformation Generation",
    "정치 선동 / 여론 조작": "Political Agitation",
    "공포 불안 조장": "Fearmongering",
}


def plot_category_radar_charts(all_data):
    """Create radar charts for each category (1x4 layout) with tighter spacing."""
    first_model = list(all_data.values())[0]
    categories = [cat["category"] for cat in first_model["by_category"]]

    # Compact 1x4 row (original size); A-G axis labels on every panel.
    fig = plt.figure(figsize=(7.0, 2.9))
    from matplotlib.gridspec import GridSpec

    gs = GridSpec(1, 4, figure=fig, wspace=0.4)

    csv_data = []

    for cat_idx, category in enumerate(categories):
        ax = fig.add_subplot(gs[0, cat_idx], polar=True)
        values_dict = {}

        for model_name, model_data in all_data.items():
            cat_data = next(
                (
                    cat
                    for cat in model_data["by_category"]
                    if cat["category"] == category
                ),
                None,
            )
            if cat_data:
                checklist = cat_data["checklist_stats"]
                values = [
                    checklist.get(label, {}).get("Y_count", 0)
                    for label in CHECKLIST_LABELS
                ]
                values_dict[model_name] = values

                # Add to CSV data
                row = {
                    "Category": CATEGORY_TRANSLATION.get(category, category),
                    "Model": MODEL_SHORT_NAMES.get(model_name, model_name),
                }
                for label, val in zip(CHECKLIST_LABELS, values):
                    row[label] = val
                csv_data.append(row)

        # 레이더 차트 생성 (이전의 라벨 회전 로직이 적용된 create_radar_chart 호출)
        create_radar_chart(
            ax,
            CHECKLIST_ABBR,
            values_dict,
            f"{CATEGORY_TRANSLATION.get(category, category)}",
            show_legend=False,
        )

    # Legend placed horizontally along the bottom of the figure.
    legend_handles = []
    for idx, model_name in enumerate(all_data.keys()):
        short_name = MODEL_SHORT_NAMES.get(model_name, model_name)
        handle = plt.Line2D(
            [0],
            [0],
            color=MODEL_COLORS[idx % len(MODEL_COLORS)],
            linewidth=3,
            label=short_name,
        )
        legend_handles.append(handle)

    plt.tight_layout(pad=1.0, rect=[0, 0.10, 1, 1])
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=6,
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )

    plt.savefig("toxicity_category_radar.pdf", bbox_inches="tight")
    plt.savefig("toxicity_category_radar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: toxicity_category_radar.pdf / .png")

    # Save to CSV
    csv_file = "toxicity_radar_data.csv"
    fieldnames = ["Category", "Model"] + CHECKLIST_LABELS
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)
    print(f"Saved: {csv_file}")


def plot_toxicity_distribution_bar_chart(all_data):
    """Create stacked bar chart for toxicity distribution."""
    # Get toxicity levels keys
    first_model = list(all_data.values())[0]
    toxicity_levels = list(first_model["toxicity_levels"].keys())

    fig, ax = plt.subplots(figsize=(3.4, 3.0), layout="constrained")

    tox_short = [
        "Full Defense",
        "Passive\nRefusal",
        "Indirect\nHint",
        "Specific\nInfo",
        "Immediate\nExec",
    ]

    # Prepare data for stacked bar chart
    model_names = list(all_data.keys())
    short_names = [MODEL_SHORT_NAMES.get(m, m) for m in model_names]

    data = np.zeros((len(toxicity_levels), len(model_names)))  # (levels, models)

    csv_bar_data = []
    for i, model_name in enumerate(model_names):
        model_data = all_data[model_name]
        row = {"Model": MODEL_SHORT_NAMES.get(model_name, model_name)}
        for j, level in enumerate(toxicity_levels):
            percentage = model_data["toxicity_levels"][level]["percentage"]
            data[j, i] = percentage
            row[tox_short[j]] = percentage
        csv_bar_data.append(row)

    # Plot stacked bars
    bottom = np.zeros(len(model_names))
    colors = sns.color_palette("viridis", len(toxicity_levels))

    for i, level in enumerate(toxicity_levels):
        ax.bar(
            short_names,
            data[i],
            bottom=bottom,
            label=tox_short[i],
            color=colors[i],
            width=0.6,
        )
        bottom += data[i]

    ax.set_ylabel("Percentage (%)")
    ax.tick_params(axis="x", rotation=90)
    ax.set_ylim(0, 100)
    # Legend below the plot; constrained layout keeps it clear of the rotated
    # x labels.
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    plt.savefig(os.path.join(SCRIPT_DIR, "toxicity_distribution_bar.pdf"))
    plt.savefig(os.path.join(SCRIPT_DIR, "toxicity_distribution_bar.png"), dpi=150)
    plt.close()
    print("Saved: toxicity_distribution_bar.pdf / .png")

    # Save to CSV
    csv_file = "toxicity_bar_data.csv"
    fieldnames = ["Model"] + tox_short
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_bar_data)
    print(f"Saved: {csv_file}")


def main():
    """Main function to generate all visualizations."""
    print("Loading data...")
    all_data = load_all_data()
    print(f"Loaded {len(all_data)} models: {list(all_data.keys())}")

    print("\nGenerating visualizations...")
    plot_category_radar_charts(all_data)
    plot_toxicity_distribution_bar_chart(all_data)

    print("\nAll visualizations saved to:", BASE_DIR)


if __name__ == "__main__":
    main()
