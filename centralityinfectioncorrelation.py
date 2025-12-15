
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr


# ------------------------------------------------------------
# 1) File name (change this if your file has a different name)
# ------------------------------------------------------------
METRICS_FILE = "fin_network_infection_metrics.csv"


# ------------------------------------------------------------
# 2) Main program
# ------------------------------------------------------------
def main():
    # ---- Step A: Check the file exists ----
    if not Path(METRICS_FILE).exists():
        raise FileNotFoundError(
            f"Could not find '{METRICS_FILE}'.\n"
            "Run your network metrics script first to create it."
        )

    # ---- Step B: Load the CSV into a pandas DataFrame ----
    # index_col=0 means the first column is treated as the row labels (node names)
    df = pd.read_csv(METRICS_FILE, index_col=0)

    # ---- Step C: Check the columns we need exist ----
    required_columns = [
        "infection_fraction",
        "degree",
        "betweenness",
        "eigenvector",
        "clustering"
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            "Your CSV is missing these required columns:\n"
            f"{missing}\n"
            "Make sure your metrics script saved them."
        )

    # ---- Step D: Decide which relationships to test ----
    # (x_column, y_column) pairs
    pairs_to_test = [
        ("degree", "infection_fraction"),
        ("betweenness", "infection_fraction"),
        ("eigenvector", "infection_fraction"),
        ("clustering", "infection_fraction"),
    ]

    # ---- Step E: Compute correlations and store results ----
    results = []

    for x_col, y_col in pairs_to_test:
        x_vals = df[x_col].values
        y_vals = df[y_col].values

        # Pearson correlation (linear relationship)
        pear_r, pear_p = pearsonr(x_vals, y_vals)

        # Spearman correlation (rank-based relationship)
        spear_r, spear_p = spearmanr(x_vals, y_vals)

        results.append({
            "Centrality": x_col,
            "Infection metric": y_col,
            "Pearson r": pear_r,
            "Pearson p": pear_p,
            "Spearman rho": spear_r,
            "Spearman p": spear_p
        })

    # Convert list of dictionaries -> DataFrame table
    summary = pd.DataFrame(results)

    # ---- Step F: Save the summary table as CSV ----
    summary.to_csv("fin_centrality_summary.csv", index=False)

    # Print to console (nice quick check)
    print("\nCentrality–infection correlation summary:")
    print(summary.to_string(index=False))

    # ---- Step G: Also save as a LaTeX table ----
    latex_table = summary.to_latex(
        index=False,
        float_format="%.3f",
        caption=(
            "Centrality--infection correlations for the financial network "
            "at correlation threshold $\\tau = 0.60$."
        ),
        label="tab:fin_centrality_corr"
    )

    with open("fin_centrality_summary.tex", "w", encoding="utf-8") as f:
        f.write(latex_table)

    print("\nSaved outputs:")
    print("  - fin_centrality_summary.csv")
    print("  - fin_centrality_summary.tex")


# ------------------------------------------------------------
# Run the script
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
