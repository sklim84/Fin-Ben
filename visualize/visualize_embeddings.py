import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.font_manager as fm
import platform
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE

# Define base path and file paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.join(SCRIPT_DIR, "..", "_datasets", "0_integration")
files_info = [
    {
        "filename": "1_fin_knowledge.csv",
        "group_col": "category",
        "title": "Fin-Knowledge: Semantic Space (t-SNE)",
        "output_filename": "fin_knowledge_embedding_tsne.png",
    },
    {
        "filename": "2_fin_reasoning.csv",
        "group_col": "category",
        "title": "Fin-Reasoning: Semantic Space (t-SNE)",
        "output_filename": "fin_reasoning_embedding_tsne.png",
    },
    {
        "filename": "3_fin_toxicity.csv",
        "group_col": "category",
        "title": "Fin-Toxicity: Semantic Space (t-SNE)",
        "output_filename": "fin_toxicity_embedding_tsne.png",
    },
]

output_dir = SCRIPT_DIR
os.makedirs(output_dir, exist_ok=True)


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
                fm.fontManager.addfont(path)
                plt.rcParams["font.family"] = fm.FontProperties(fname=path).get_name()
                break
        else:
            plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False


setup_korean_font()

# Colors palette (extending the one used before for more categories if needed)
# Using a large palette for scatter plots to distinguish many categories
SCATTER_PALETTE = sns.color_palette("husl", 20)

# Category order for 2_fin_reasoning (using translated display names)
FIN_REASONING_CATEGORY_ORDER = [
    "Relevant Info at Middle (EN Noise)",
    "Relevant Info Dispersed",
    "Relevant Info at Middle",
    "Relevant Info at End",
    "Relevant Info at Front",
    "Relevant Info Scattered",
    "Relevant Info Only",
    "Relevant Info Only (Shuffled)",
]

# Category order for 1_fin_knowledge (alphabetical as per user request)
FIN_KNOWLEDGE_CATEGORY_ORDER = [
    "Bond Market",
    "Derivatives",
    "Digital Finance",
    "Distribution Market",
    "Econometrics",
    "Financial Fundamentals",
    "Financial Institutions",
    "Financial Management",
    "Financial Products",
    "Insurance Products",
    "Intermediate Accounting",
    "International Economics",
    "International Financial Policy",
    "Macroeconomics",
    "Microeconomics",
    "Monetary Finance",
    "Production & Operations Management",
    "Real Estate Market",
    "Securities Market",
    "Tax Law",
]

# Category order for 3_fin_toxicity
FIN_TOXICITY_CATEGORY_ORDER = [
    "False Information Generation",
    "Inciting Fear & Anxiety",
    "Political Incitement / Opinion Manipulation",
    "Illegal Misconduct Advice",
]

TRANSLATION_MAP = {
    # Fin-Knowledge
    "중급회계": "Intermediate Accounting",
    "생산운영관리": "Production & Operations Management",
    "재무관리": "Financial Management",
    "미시경제학": "Microeconomics",
    "계량경제": "Econometrics",
    "거시경제학": "Macroeconomics",
    "국제경제학": "International Economics",
    "화폐금융": "Monetary Finance",
    "디지털 금융": "Digital Finance",
    "채권시장": "Bond Market",
    "국제금융정책": "International Financial Policy",
    "증권시장": "Securities Market",
    "보험상품": "Insurance Products",
    "금융상품": "Financial Products",
    "금융기관": "Financial Institutions",
    "유통시장": "Distribution Market",
    "파생상품": "Derivatives",
    "금융의 기초": "Financial Fundamentals",
    "세법": "Tax Law",
    "부동산시장": "Real Estate Market",
    # Fin-Toxicity
    "불법 부정행위 조언": "Illegal Misconduct Advice",
    "허위정보 생성": "False Information Generation",
    "정치 선동 / 여론 조작": "Political Incitement / Opinion Manipulation",
    "공포 불안 조장": "Inciting Fear & Anxiety",
    # Fin-Reasoning
    "context_relevant_middle_with_en_noise": "Relevant Info at Middle (EN Noise)",
    "context_relevant_dispersed": "Relevant Info Dispersed",
    "context_relevant_middle": "Relevant Info at Middle",
    "context_relevant_end": "Relevant Info at End",
    "context_relevant_front": "Relevant Info at Front",
    "context_relevant_scattered": "Relevant Info Scattered",
    "context_relevant_only": "Relevant Info Only",
    "context_relevant_only_shuffled": "Relevant Info Only (Shuffled)",
}


def main():
    print("Loading embedding model...")
    model = SentenceTransformer(
        "dragonkue/snowflake-arctic-embed-l-v2.0-ko", device="cuda:0"
    )

    sns.set_theme(style="whitegrid")
    # Re-apply font after set_theme
    setup_korean_font()
    # Serif style matched to the paper's other vector figures; sized for
    # single-column placement so on-page text stays ~8pt.
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.6,
        "legend.frameon": False,
        "pdf.fonttype": 42,
    })

    for info in files_info:
        file_path = os.path.join(base_path, info["filename"])
        print(f"Processing {file_path}...")

        try:
            df = pd.read_csv(file_path)

            # Drop rows with missing questions
            df = df.dropna(subset=["question"])

            # Apply translations to group column if applicable
            if info["group_col"] in df.columns:
                df[info["group_col"]] = df[info["group_col"]].map(
                    lambda x: TRANSLATION_MAP.get(x, x)
                )

            print(f"  Encoding {len(df)} questions...")
            embeddings = model.encode(
                df["question"].tolist(), show_progress_bar=True, batch_size=32
            )

            print("  Running t-SNE...")
            tsne = TSNE(
                n_components=2, random_state=42, perplexity=min(30, len(df) - 1)
            )
            reduced_embeddings = tsne.fit_transform(embeddings)

            df["x"] = reduced_embeddings[:, 0]
            df["y"] = reduced_embeddings[:, 1]

            # Single-column size; add height for the one-per-row legend below.
            n_groups = df[info["group_col"]].nunique()
            plt.figure(figsize=(3.5, 2.6 + 0.14 * n_groups), layout="constrained")

            # High-contrast palette for few categories, else perceptual "husl".
            if n_groups <= 8:
                palette = sns.color_palette("Set2", n_groups)
            else:
                palette = sns.color_palette("husl", n_groups)

            # Apply category order based on file type
            if info["filename"] == "1_fin_knowledge.csv":
                df[info["group_col"]] = pd.Categorical(
                    df[info["group_col"]],
                    categories=FIN_KNOWLEDGE_CATEGORY_ORDER,
                    ordered=True,
                )
            elif info["filename"] == "2_fin_reasoning.csv":
                df[info["group_col"]] = pd.Categorical(
                    df[info["group_col"]],
                    categories=FIN_REASONING_CATEGORY_ORDER,
                    ordered=True,
                )
            elif info["filename"] == "3_fin_toxicity.csv":
                df[info["group_col"]] = pd.Categorical(
                    df[info["group_col"]],
                    categories=FIN_TOXICITY_CATEGORY_ORDER,
                    ordered=True,
                )

            ax = sns.scatterplot(
                data=df,
                x="x",
                y="y",
                hue=info["group_col"],
                palette=palette,
                s=60,
                alpha=0.7,
                edgecolor="w",
                linewidth=0.5,
            )

            ax.set_xlabel("t-SNE Dimension 1", fontweight="bold")
            ax.set_ylabel("t-SNE Dimension 2", fontweight="bold")
            # Category legend outside, below the plot; one entry per row so long
            # names never overlap. constrained layout reserves the room, so the
            # legend never collides with the axis or its labels.
            handles, labels = ax.get_legend_handles_labels()
            if ax.get_legend() is not None:
                ax.get_legend().remove()
            plt.gcf().legend(
                handles,
                labels,
                loc="outside lower center",
                ncol=1,
                fontsize=6.5,
                frameon=False,
                handletextpad=0.3,
            )

            output_path = os.path.join(output_dir, info["output_filename"])
            plt.savefig(output_path[:-4] + ".pdf")
            plt.savefig(output_path, dpi=150)
            print(f"Saved plot to {output_path}")
            plt.close()

        except Exception as e:
            print(f"Error processing {info['filename']}: {e}")
            import traceback

            traceback.print_exc()

    print("All embedding visualizations generated.")


if __name__ == "__main__":
    main()
