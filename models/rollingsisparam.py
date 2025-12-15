
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ------------------------------------------------------------
# 0) SETTINGS YOU MAY WANT TO CHANGE
# ------------------------------------------------------------
INF_PANEL_PARQUET = Path("infection_panel.parquet")
INF_PANEL_CSV = Path("infection_panel.csv")

WINDOW = 60                  # rolling window length (trading days)
MIN_EFFECTIVE_POINTS = 10    # minimum number of days with I(t) > 0 inside window
SECTOR_NAME = "Financial sector"

OUT_FRAC_FIG = "fin_infection_fraction_check.png"
OUT_PARAMS_FIG = "rolling_sis_params_plot.png"
OUT_CSV = "rolling_sis_params.csv"


# ------------------------------------------------------------
# 1) LOAD THE INFECTION PANEL
# ------------------------------------------------------------
def load_infection_panel():
    """
    Load the infection panel.

    Returns:
      DataFrame where:
        - index is datetime (dates)
        - columns are tickers/nodes
        - values are 0/1
    """
    # Try Parquet first (faster, smaller)
    if INF_PANEL_PARQUET.exists():
        df = pd.read_parquet(INF_PANEL_PARQUET)

    # If Parquet doesn't exist, try CSV
    elif INF_PANEL_CSV.exists():
        df = pd.read_csv(INF_PANEL_CSV, index_col=0)

    # If neither exists, stop with a clear error message
    else:
        raise FileNotFoundError(
            "No infection panel found. Expected one of:\n"
            f"  - {INF_PANEL_PARQUET.resolve()}\n"
            f"  - {INF_PANEL_CSV.resolve()}"
        )

    # Make sure index is datetime (needed for plotting + rolling work)
    if not np.issubdtype(df.index.dtype, np.datetime64):
        df.index = pd.to_datetime(df.index, errors="coerce")

    # Sort by date (oldest -> newest)
    df = df.sort_index()

    # Remove rows where date parsing failed (index is NaT)
    df = df.loc[df.index.notna()]

    return df


def compute_infection_fraction(panel_df):
    """
    Compute I(t) = fraction infected per date.

    Example:
      If a day has 100 tickers and 20 infected, I(t)=0.20
    """
    frac = panel_df.mean(axis=1).astype(float)
    frac.name = "infection_fraction"
    return frac


# ------------------------------------------------------------
# 2) FIT SIS PARAMETERS IN ONE WINDOW
# ------------------------------------------------------------
def fit_sis_window(I_window):
    """
    Fit beta and gamma using ONE window of infection fraction values.

    Model (discrete approximation):
      dI_t = I_{t+1} - I_t
      dI_t ≈ beta * I_t * (1 - I_t) - gamma * I_t

    This becomes a simple least squares regression:
      dI_t ≈ beta * X_t + gamma * (-Y_t)
      where:
        X_t = I_t * (1 - I_t)
        Y_t = I_t

    Returns:
      beta_hat, gamma_hat
      or (nan, nan) if window is not fit-able.
    """
    I_window = np.asarray(I_window, dtype=float)

    # Need at least 2 points to compute differences
    if I_window.size < 2:
        return np.nan, np.nan

    # dI_t = I_{t+1} - I_t
    dI = I_window[1:] - I_window[:-1]

    # Build regressors
    X = I_window[:-1] * (1.0 - I_window[:-1])  # infection pressure term
    Y = I_window[:-1]                           # recovery term

    # Design matrix A = [X, -Y]
    A = np.vstack([X, -Y]).T

    # If everything is basically zero/constant, least squares is meaningless
    if np.allclose(A, 0) or np.allclose(dI, 0):
        return np.nan, np.nan

    # Least squares solution: minimize ||A*theta - dI||
    theta, *_ = np.linalg.lstsq(A, dI, rcond=None)

    beta_hat = float(theta[0])
    gamma_hat = float(theta[1])
    return beta_hat, gamma_hat


# ------------------------------------------------------------
# 3) ROLLING FIT OVER TIME
# ------------------------------------------------------------
def rolling_sis_fit(frac_series, window, min_points):
    """
    Perform rolling SIS estimation over the whole time series.

    For each rolling window:
      - take window values
      - fit beta/gamma
      - store results at the "centre" date of the window

    min_points means:
      - require at least min_points days with I(t) > 0 in the window
        otherwise store NaN (skip)
    """
    frac_series = frac_series.dropna()
    values = frac_series.values
    dates = frac_series.index

    if len(values) < window:
        raise ValueError(f"Time series too short (n={len(values)}) for window={window}")

    betas = []
    gammas = []
    centre_dates = []

    # Slide the window along the series
    for start in range(0, len(values) - window + 1):
        end = start + window
        w = values[start:end]

        # Skip window if it barely has any infection activity
        if np.sum(w > 0) < min_points:
            betas.append(np.nan)
            gammas.append(np.nan)
        else:
            b, g = fit_sis_window(w)
            betas.append(b)
            gammas.append(g)

        # Store at centre date so plots align nicely
        centre_dates.append(dates[start + window // 2])

    # Build DataFrame of rolling estimates
    roll_df = pd.DataFrame(
        {"beta_hat": betas, "gamma_hat": gammas},
        index=pd.to_datetime(centre_dates)
    )

    return roll_df


# ------------------------------------------------------------
# 4) PLOTTING (publication-style)
# ------------------------------------------------------------
def plot_infection_fraction(frac):
    """Save a clean plot of infection fraction."""
    fig, ax = plt.subplots(figsize=(8.5, 3.0))

    ax.plot(frac.index, frac.values)
    ax.set_ylabel("Fraction infected")
    ax.set_xlabel("Date")
    ax.set_title(f"{SECTOR_NAME}: infection fraction over time")
    ax.grid(True, alpha=0.3)

    # Auto-format date labels
    fig.autofmt_xdate()
    fig.tight_layout()

    fig.savefig(OUT_FRAC_FIG, dpi=300)
    plt.close(fig)


def plot_rolling_params(roll_df):
    """Save a clean two-panel plot of beta(t) and gamma(t)."""
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 4.8), sharex=True)

    # Beta panel
    axes[0].plot(roll_df.index, roll_df["beta_hat"], label=r"$\hat{\beta}(t)$")
    axes[0].axhline(0, linestyle="--", linewidth=1)
    axes[0].set_ylabel(r"Infection rate $\hat{\beta}(t)$")
    axes[0].legend(frameon=False)
    axes[0].grid(True, alpha=0.3)

    # Gamma panel
    axes[1].plot(roll_df.index, roll_df["gamma_hat"], label=r"$\hat{\gamma}(t)$")
    axes[1].axhline(0, linestyle="--", linewidth=1)
    axes[1].set_ylabel(r"Recovery rate $\hat{\gamma}(t)$")
    axes[1].set_xlabel("Date")
    axes[1].legend(frameon=False)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        f"{SECTOR_NAME}: rolling SIS parameter estimates\n"
        f"window = {WINDOW} trading days",
        y=1.02
    )

    fig.autofmt_xdate()
    fig.tight_layout()

    fig.savefig(OUT_PARAMS_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------
# 5) MAIN PROGRAM
# ------------------------------------------------------------
def main(verbose=True):
    # Step 1: Load infection panel
    panel = load_infection_panel()

    # Step 2: Compute daily infection fraction
    frac = compute_infection_fraction(panel)

    # Step 3: Rolling SIS estimation
    roll_df = rolling_sis_fit(frac, window=WINDOW, min_points=MIN_EFFECTIVE_POINTS)

    # Step 4: Save CSV results
    roll_df.to_csv(OUT_CSV, index_label="date")

    # Step 5: Save figures
    plot_infection_fraction(frac)
    plot_rolling_params(roll_df)

    # Step 6: Print basic info to console
    if verbose:
        print(f"Loaded infection panel: {panel.shape}")
        print("\nInfection fraction summary:")
        print(frac.describe())
        print("\nSaved outputs:")
        print("  -", OUT_FRAC_FIG)
        print("  -", OUT_PARAMS_FIG)
        print("  -", OUT_CSV)


if __name__ == "__main__":
    main()
