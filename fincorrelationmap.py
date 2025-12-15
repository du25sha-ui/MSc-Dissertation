

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ------------------------------------------------------------
# 1) FILE NAMES (edit if needed)
# ------------------------------------------------------------
CORR_FILE = "fin_corr_matrix.csv"   # input correlation matrix
OUT_FIG = "fin_corr_heatmap.png"    # output image file


# ------------------------------------------------------------
# 2) MAIN PROGRAM
# ------------------------------------------------------------
def main():

    # ---- Step A: Check that the correlation file exists ----
    if not Path(CORR_FILE).exists():
        raise FileNotFoundError(
            f"File '{CORR_FILE}' not found.\n"
            "Make sure your correlation matrix CSV exists."
        )

    # ---- Step B: Load the correlation matrix ----
    # index_col=0 means the first column contains row labels (tickers)
    corr = pd.read_csv(CORR_FILE, index_col=0)

    # Extract ticker names (used for axis labels)
    tickers = corr.columns.tolist()

    # ---- Step C: Create the figure ----
    fig, ax = plt.subplots(figsize=(6.5, 6))

    # Show the correlation matrix as an image
    im = ax.imshow(
        corr.values,
        vmin=-1,          # correlations range from -1 to 1
        vmax=1,
        cmap="coolwarm",  # red-blue color map
        aspect="equal"
    )

    # ---- Step D: Set axis ticks and labels ----
    ax.set_xticks(np.arange(len(tickers)))
    ax.set_yticks(np.arange(len(tickers)))

    ax.set_xticklabels(tickers, rotation=45, ha="right")
    ax.set_yticklabels(tickers)

    ax.set_title("Financial sector correlation matrix", fontsize=15)

    # ---- Step E: Add a colorbar ----
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Correlation", rotation=270, labelpad=15)

    # ---- Step F: Add light gridlines between cells ----
    # These help visually separate each correlation value
    ax.set_xticks(np.arange(-0.5, len(tickers), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(tickers), 1), minor=True)

    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)

    # ---- Step G: Adjust layout so labels are not cut off ----
    fig.tight_layout()

    # ---- Step H: Save and show the figure ----
    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    print(f"Saved correlation heatmap to: {OUT_FIG}")

    plt.show()


# ------------------------------------------------------------
# 3) RUN SCRIPT
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
