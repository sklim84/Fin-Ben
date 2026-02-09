import pandas as pd
import tiktoken
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.font_manager as fm
import platform

# Initialize tiktoken encoding
enc = tiktoken.get_encoding("o200k_base")

# Define base path and file paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.join(SCRIPT_DIR, "..", "Fin-Ben_main", "_datasets", "0_integration")
files_info = [
    {
        "filename": "1_fin_knowledge.csv",
        "group_col": "category",
        "title": "Fin-Knowledge: Question Length Distribution",
        "output_filename": "fin_knowledge_length_boxplot.png",
    },
    {
        "filename": "2_fin_reasoning.csv",
        "group_col": "category",
        "title": "Fin-Reasoning: Question Length Distribution",
        "output_filename": "fin_reasoning_length_boxplot.png",
    },
    {
        "filename": "3_fin_toxicity.csv",
        "group_col": "category",
        "title": "Fin-Toxicity: Question Length Distribution",
        "output_filename": "fin_toxicity_length_boxplot.png",
    },
]

output_dir = SCRIPT_DIR
os.makedirs(output_dir, exist_ok=True)


def calculate_token_length(text):
    if pd.isna(text):
        return 0
    return len(enc.encode(str(text), disallowed_special=()))


# Set up Korean font (consistent with visualize_toxicity.py)
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

# Colors for boxplots (using a palette similar to visualize_toxicity.py's MODEL_COLORS)
# Although boxplots aren't by model, using a nice palette enhances aesthetics.
BOXPLOT_PALETTE = [
    "#FF6B6B",  # Red
    "#4ECDC4",  # Teal
    "#45B7D1",  # Blue
    "#96CEB4",  # Green
    "#FFEAA7",  # Yellow
    "#D6A2E8",  # Purple
    "#FFBE76",  # Orange
    "#74b9ff",  # Light Blue
]

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
    "Political Incitement\n/ Opinion Manipulation",
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
    "정치 선동 / 여론 조작": "Political Incitement\n/ Opinion Manipulation",
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


for info in files_info:
    file_path = os.path.join(base_path, info["filename"])
    print(f"Processing {file_path}...")

    try:
        df = pd.read_csv(file_path)

        # Apply translations to group column if applicable
        if info["group_col"] in df.columns:
            df[info["group_col"]] = df[info["group_col"]].map(
                lambda x: TRANSLATION_MAP.get(x, x)
            )

        # Calculate token lengths
        df["token_length"] = df["question"].apply(calculate_token_length)

        # Apply category order based on file type
        if info["filename"] == "1_fin_knowledge.csv":
            df[info["group_col"]] = pd.Categorical(
                df[info["group_col"]],
                categories=FIN_KNOWLEDGE_CATEGORY_ORDER,
                ordered=True,
            )
            df = df.sort_values(info["group_col"])
        elif info["filename"] == "2_fin_reasoning.csv":
            df[info["group_col"]] = pd.Categorical(
                df[info["group_col"]],
                categories=FIN_REASONING_CATEGORY_ORDER,
                ordered=True,
            )
            df = df.sort_values(info["group_col"])
        elif info["filename"] == "3_fin_toxicity.csv":
            df[info["group_col"]] = pd.Categorical(
                df[info["group_col"]],
                categories=FIN_TOXICITY_CATEGORY_ORDER,
                ordered=True,
            )
            df = df.sort_values(info["group_col"])

        # Create boxplot
        plt.figure(figsize=(14, 8))  # Larger figure size
        unique_vals = df[info["group_col"]].unique()
        palette_to_use = (
            BOXPLOT_PALETTE[: len(unique_vals)]
            if len(unique_vals) <= len(BOXPLOT_PALETTE)
            else None
        )

        # Determine category order based on file type
        if info["filename"] == "1_fin_knowledge.csv":
            category_order = FIN_KNOWLEDGE_CATEGORY_ORDER
        elif info["filename"] == "2_fin_reasoning.csv":
            category_order = FIN_REASONING_CATEGORY_ORDER
        elif info["filename"] == "3_fin_toxicity.csv":
            category_order = FIN_TOXICITY_CATEGORY_ORDER
        else:
            category_order = None

        sns.boxplot(
            x=info["group_col"],
            y="token_length",
            data=df,
            order=category_order,
            hue=info["group_col"],
            hue_order=category_order,
            palette=palette_to_use,
            legend=False,
            dodge=False,
        )

        # Style enhancements
        plt.title(info["title"], fontsize=16, fontweight="bold", pad=20)
        plt.xlabel(info["group_col"].capitalize(), fontsize=14, fontweight="bold")
        plt.ylabel("Token Count (o200k_base)", fontsize=14, fontweight="bold")
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(axis="y", alpha=0.3, linestyle="--")

        plt.tight_layout()

        output_path = os.path.join(output_dir, info["output_filename"])
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
        plt.close()

    except Exception as e:
        print(f"Error processing {info['filename']}: {e}")

print("All plots generated.")
