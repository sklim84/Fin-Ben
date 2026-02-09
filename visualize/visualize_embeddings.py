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
base_path = os.path.join(SCRIPT_DIR, "..", "Fin-Ben_main", "_datasets", "0_integration")
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

            # Create scatter plot
            plt.figure(figsize=(14, 10))

            # Use a high-contrast palette if number of categories is small (<= 8), else default "husl"
            n_groups = df[info["group_col"]].nunique()
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

            sns.scatterplot(
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

            plt.title(info["title"], fontsize=18, fontweight="bold", pad=20)
            plt.xlabel("t-SNE Dimension 1", fontsize=14, fontweight="bold")
            plt.ylabel("t-SNE Dimension 2", fontsize=14, fontweight="bold")
            plt.legend(
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
                borderaxespad=0.0,
                title=info["group_col"].capitalize(),
            )
            plt.tight_layout()

            output_path = os.path.join(output_dir, info["output_filename"])
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Saved plot to {output_path}")
            plt.close()

        except Exception as e:
            print(f"Error processing {info['filename']}: {e}")
            import traceback

            traceback.print_exc()

    print("All embedding visualizations generated.")


if __name__ == "__main__":
    main()
