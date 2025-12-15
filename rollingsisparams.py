

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ------------------------------------------------------------
# 0) USER SETTINGS (edit these)
# ------------------------------------------------------------
INF_PANEL_FILE = "infection_panel.parquet"   # your input file (Parquet)
WINDOW = 60                                  # rolling window size in days
MIN_EFFECTIVE_POINTS = 10                    # minimum “useful” points per window
SECTOR_NAME = "Financial sector"             # used for plot titles


# ------------------------------------------------------------
# 1) LOAD DATA AND CREATE INFECTION FRACTION
# ------------------------------------------------------------
def load_infection_panel(file_path):
    """
    Load the infection panel (0/1 values) from a Parquet file.

    Returns:
      DataFrame indexed by date, sorted from oldest to newest.
    """
    df = pd.read_parquet(file_path)

    # Make sure the index is datetime (important for plots)
    if not np.issubdtype(df.index.dtype, np.datetime64):
        df.index = pd.to_datetime(df.index)

    df = df.sort_index()
    return df


def compute_infection_fraction(panel_df):
    """
    Compute the daily fraction infected.

    Example:
      If you have 100 columns (assets) and 20 are infected (1),
      the infection fraction that day = 20/100 = 0.20
    """
    frac = panel_df.mean(axis=1)
    frac.name = "infection_fraction"
    return frac


# ------------------------------------------------------------
# 2) FIT SIS PARAMETERS IN ONE WINDOW
# ------------------------------------------------------------
def fit_sis_window(I_window):
    """
    Fit beta and gamma using one rolling window of infection fractions.

    We use this relationship:
      dI_t = I_{t+1} - I_t
      dI_t ≈ beta * I_t * (1 - I_t) - gamma * I_t

    This can be written as a linear regression:
      dI_t ≈ beta * X_t + gamma * (-Y_t)
      where:
        X_t = I_t * (1 - I_t)
        Y_t = I_t

    Returns:
      (beta_hat, gamma_hat)
      or (np.nan, np.nan) if the window has no variation / not fit-able.
    """
    I_window = np.asarray(I_window, dtype=float)

    # Differences: dI_t = I_{t+1} - I_t
    dI = I_window[1:] - I_window[:-1]

    # Regressors
    X = I_window[:-1] * (1.0 - I_window[:-1])  # infection term
    Y = I_window[:-1]                           # recovery term

    # Create design matrix A = [X, -Y]
    A = np.vstack([X, -Y]).T

    # If everything is basically zero/constant, we cannot fit parameters
    if np.allclose(A, 0) or np.allclose(dI, 0):
        return np.nan, np.nan

    # Solve least squares: minimize ||A * theta - dI||
    # theta = [beta_hat, gamma_hat]
    theta, residuals, rank, s = np.linalg.lstsq(A, dI, rcond=None)

    beta_hat = theta[0]
    gamma_hat = theta[1]
    return beta_hat, gamma_hat


# ------------------------------------------------------------
# 3) ROLLING FIT OVER TIME
# ------------------------------------------------------------
def rolling_sis_fit(frac_series, window=60, min_points=10):
    """
    Run the SIS fit in a rolling window.

    For each window:
      - take values from start to start+window
      - fit (beta_hat, gamma_hat)
      - store results at the window "centre" date

    Returns:
      DataFrame with columns ['beta_hat', 'gamma_hat']
    """
    frac_series = frac_series.dropna()
    values = frac_series.values
    dates = frac_series.index

    n = len(values)
    if n < window:
        raise ValueError(f"Time series is too short (n={n}) for window={window}")

    beta_list = []
    gamma_list = []
    centre_dates = []

    # Slide the window from start=0 to start=n-window
    for start in range(0, n - window + 1):
        end = start + window
        window_vals = values[start:end]

        # If there are too few non-zero infection points, skip (not enough info)
        if np.sum(window_vals > 0) < min_points:
            beta_list.append(np.nan)
            gamma_list.append(np.nan)
            centre_dates.append(dates[start + window // 2])
            continue

        beta_hat, gamma_hat = fit_sis_window(window_vals)

        beta_list.append(beta_hat)
        gamma_list.append(gamma_hat)
        centre_dates.append(dates[start + window // 2])

    result = pd.DataFrame(
        {"beta_hat": beta_list, "gamma_hat": gamma_list},
        index=pd.to_datetime(centre_dates)
    )

    return result


# ------------------------------------------------------------
# 4) PLOTS
# ------------------------------------------------------------
def plot_infection_fraction(frac_series):
    """Plot infection fraction over time."""
    plt.figure(figsize=(10, 4))
    plt.plot(frac_series.index, frac_series.values)
    plt.ylabel("Fraction infected")
    plt.xlabel("Date")
    plt.title(f"{SECTOR_NAME}: infection fraction over time\n"
              "(empirical, from volatility-return rule)")
    plt.tight_layout()
    plt.show()


def plot_rolling_params(roll_df):
    """Plot beta(t) and gamma(t) over time."""
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax[0].plot(roll_df.index, roll_df["beta_hat"], label=r"$\hat{\beta}(t)$")
    ax[0].axhline(0, linestyle="--")
    ax[0].set_ylabel(r"Infection rate $\hat{\beta}(t)$")
    ax[0].legend()

    ax[1].plot(roll_df.index, roll_df["gamma_hat"], label=r"$\hat{\gamma}(t)$")
    ax[1].axhline(0, linestyle="--")
    ax[1].set_ylabel(r"Recovery rate $\hat{\gamma}(t)$")
    ax[1].set_xlabel("Date")
    ax[1].legend()

    fig.suptitle(f"{SECTOR_NAME}: rolling SIS parameter estimates\n"
                 f"window = {WINDOW} trading days", y=1.02)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 5) MAIN SCRIPT
# ------------------------------------------------------------
def main():
    # Step 1: Check file exists
    if not Path(INF_PANEL_FILE).exists():
        raise FileNotFoundError(
            f"File '{INF_PANEL_FILE}' not found.\n"
            "Check the path or generate/export your infection_panel first."
        )

    # Step 2: Load panel
    panel = load_infection_panel(INF_PANEL_FILE)
    print(f"Loaded infection panel with shape {panel.shape}")

    # Step 3: Compute infection fraction
    frac = compute_infection_fraction(panel)
    print("\nInfection fraction stats:")
    print(frac.describe())

    # Step 4: Plot infection fraction
    plot_infection_fraction(frac)

    # Step 5: Rolling SIS fit
    roll_df = rolling_sis_fit(
        frac_series=frac,
        window=WINDOW,
        min_points=MIN_EFFECTIVE_POINTS
    )

    print("\nRolling fit summary (first few rows):")
    print(roll_df.head())

    # Step 6: Plot rolling beta and gamma
    plot_rolling_params(roll_df)

    # Step 7: Save results
    out_csv = "rolling_sis_params.csv"
    roll_df.to_csv(out_csv, index_label="date")
    print(f"\nSaved rolling parameter estimates to: {out_csv}")


if __name__ == "__main__":
    main()
