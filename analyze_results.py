import pandas as pd
from pathlib import Path

INPUT = Path("analysis/task_metrics.csv")
OUT_TABLES = Path("tables")
OUT_TABLES.mkdir(exist_ok=True)

df = pd.read_csv(INPUT)

print("Columns:")
print(df.columns.tolist())
print()
print("Rows:", len(df))
print()

group_cols = ["model", "method"]

summary = (
    df.groupby(group_cols)
    .agg(
        tasks=("task_id", "count"),
        success=("success", "mean"),
        wrong=("wrong_tool_count", "mean"),
        premature=("premature_action_count", "mean"),
        tools_per_step=("avg_tools_per_step", "mean"),
        tokens=("total_tokens", "mean"),
        steps=("steps", "mean"),
    )
    .reset_index()
)

method_order = [
    "all_tools",
    "keyword_top_5",
    "keyword_top_10",
    "state_aware",
    "full_causal_path",
    "cmtf",
]

summary["method"] = pd.Categorical(summary["method"], categories=method_order, ordered=True)
summary = summary.sort_values(["model", "method"])

print("=== Summary by model and method ===")
print(summary.to_string(index=False))

summary.to_csv(OUT_TABLES / "summary_by_model_method.csv", index=False)

aggregate = (
    df.groupby("method")
    .agg(
        tasks=("task_id", "count"),
        success=("success", "mean"),
        wrong=("wrong_tool_count", "mean"),
        premature=("premature_action_count", "mean"),
        tools_per_step=("avg_tools_per_step", "mean"),
        tokens=("total_tokens", "mean"),
        steps=("steps", "mean"),
    )
    .reset_index()
)

aggregate["method"] = pd.Categorical(aggregate["method"], categories=method_order, ordered=True)
aggregate = aggregate.sort_values("method")

print()
print("=== Aggregate summary ===")
print(aggregate.to_string(index=False))

aggregate.to_csv(OUT_TABLES / "summary_aggregate.csv", index=False)

latex_cols = ["method", "success", "wrong", "premature", "tools_per_step", "tokens"]
latex_table = aggregate[latex_cols].copy()

for col in ["success", "wrong", "premature", "tools_per_step", "tokens"]:
    latex_table[col] = latex_table[col].map(lambda x: f"{x:.2f}")

latex = latex_table.to_latex(index=False, escape=False)
(OUT_TABLES / "summary_aggregate.tex").write_text(latex)

print()
print("Wrote:")
print(" - tables/summary_by_model_method.csv")
print(" - tables/summary_aggregate.csv")
print(" - tables/summary_aggregate.tex")
