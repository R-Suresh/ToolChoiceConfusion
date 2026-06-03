import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

TABLES = Path("tables")
FIGURES = Path("figures")
FIGURES.mkdir(exist_ok=True)

df = pd.read_csv(TABLES / "summary_aggregate.csv")

method_order = [
    "all_tools",
    "keyword_top_5",
    "keyword_top_10",
    "state_aware",
    "full_causal_path",
    "cmtf",
]

display_names = {
    "all_tools": "All tools",
    "keyword_top_5": "Keyword top-5",
    "keyword_top_10": "Keyword top-10",
    "state_aware": "State-aware",
    "full_causal_path": "Full causal path",
    "cmtf": "CMTF",
}

df["method"] = pd.Categorical(df["method"], categories=method_order, ordered=True)
df = df.sort_values("method")
labels = [display_names[m] for m in df["method"].astype(str)]

def save_bar(metric, ylabel, filename):
    plt.figure(figsize=(9, 4.8))
    plt.bar(labels, df[metric])
    plt.ylabel(ylabel)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES / filename, dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES / filename.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()

save_bar("success", "Task success rate", "success_by_method.png")
save_bar("wrong", "Wrong-tool calls per task", "wrong_tools_by_method.png")
save_bar("premature", "Premature actions per task", "premature_actions_by_method.png")
save_bar("tools_per_step", "Average visible tools per step", "tools_per_step_by_method.png")
save_bar("tokens", "Average tokens per task", "tokens_by_method.png")

print("Wrote figures:")
for p in sorted(FIGURES.iterdir()):
    print(" -", p)
