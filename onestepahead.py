
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


# ------------------------------------------------------------
# 0) SETTINGS (edit these)
# ------------------------------------------------------------
TICKERS = ["JPM", "BAC", "WFC", "C", "GS", "MS", "BK", "BLK", "AXP", "SCHW", "USB", "PNC"]
START = "2015-01-01"
END = "2025-01-01"

VOL_WINDOW = 20     # rolling volatility window (in days)
EMA_SPAN = 250      # smoothing window for threshold (EMA)
ROLL_W = 60         # rolling window length for beta/gamma estimation


# ------------------------------------------------------------
# 1) DOWNLOAD PRICES
# ------------------------------------------------------------
def download_prices(tickers, start, end):
    """
    Download price data from Yahoo Finance using yfinance.

    Returns:
      DataFrame of prices (index = dates, columns = tickers)

    We try to use "Adj Close" if available; otherwise use "Close".
    """
    df = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        progress=False,
        group_by="column",
        auto_adjust=False
    )

    if df is None or df.empty:
        raise RuntimeError("No data downloaded. Check internet, tickers, or date range.")

    # yfinance often returns MultiIndex columns like ('Adj Close','JPM')
    if isinstance(df.columns, pd.MultiIndex):
        top_level = df.columns.get_level_values(0).unique()

        if "Adj Close" in top_level:
            prices = df["Adj Close"].copy()
        elif "Close" in top_level:
            prices = df["Close"].copy()
        else:
            raise KeyError("Could not find 'Adj Close' or 'Close' in downloaded data.")

    # Sometimes yfinance returns plain columns (less common)
    else:
        if "Adj Close" in df.columns:
            prices = df["Adj Close"]
        elif "Close" in df.columns:
            prices = df["Close"]
        else:
            raise KeyError("Could not find 'Adj Close' or 'Close' in downloaded data.")

        # Ensure DataFrame format
        if isinstance(prices, pd.Series):
            prices = prices.to_frame()

    # Clean: sort, forward-fill missing values, drop empty rows
    prices = prices.sort_index().ffill()
    prices = prices.dropna(how="all")

    # Keep only requested tickers (sometimes extra columns appear)
    prices = prices.loc[:, [c for c in prices.columns if c in tickers]]

    if prices.empty:
        raise RuntimeError("Prices became empty after cleaning. Check ticker symbols.")

    return prices


# ------------------------------------------------------------
# 2) RETURNS AND VOLATILITY
# ------------------------------------------------------------
def compute_log_returns(prices):
    """Log returns: r_t = log(P_t) - log(P_{t-1})."""
    return np.log(prices).diff()


def rolling_volatility(returns, window):
    """
    Rolling annualised volatility:
      sigma(t) = sqrt(252 * Var(returns over last 'window' days))
    """
    return np.sqrt(252.0 * returns.rolling(window).var())


def ema(series, span):
    """Exponential moving average (smooths a noisy series)."""
    return series.ewm(span=span, adjust=False).mean()


# ------------------------------------------------------------
# 3) ROLLING SIS PARAMETER ESTIMATION (beta(t), gamma(t))
# ------------------------------------------------------------
def rolling_sis_params(rho_obs, k_mean, window_W):
    """
    Estimate beta(t) and gamma(t) using rolling regression.

    We approximate:
      drho(t) = rho(t+1) - rho(t)
      drho(t) ≈ beta(t)*k(t)*rho(t)(1-rho(t)) - gamma(t)*rho(t)

    Regression form inside each window:
      y = drho
      x1 = rho(1-rho)
      x2 = rho

    We estimate:
      theta1 ≈ beta*k
      theta2 ≈ -gamma
    so:
      beta_hat = theta1 / k
      gamma_hat = -theta2
    """
    rho = rho_obs.dropna().astype(float)
    drho = rho.shift(-1) - rho

    # Combine into one table and drop missing rows
    data = pd.DataFrame({"rho": rho, "drho": drho, "k": k_mean}).dropna()

    beta_hat = pd.Series(index=data.index, dtype=float)
    gamma_hat = pd.Series(index=data.index, dtype=float)

    # Start after we have enough points for a rolling window
    for i in range(window_W, len(data)):
        win = data.iloc[i - window_W:i]

        y = win["drho"].values
        x1 = (win["rho"] * (1.0 - win["rho"])).values
        x2 = win["rho"].values
        X = np.column_stack([x1, x2])

        # Safety check: if X'X is nearly singular, skip
        XtX = X.T @ X
        if np.linalg.cond(XtX) > 1e12:
            continue

        theta1, theta2 = np.linalg.solve(XtX, X.T @ y)

        k_t = data["k"].iloc[i]
        if (not np.isfinite(k_t)) or (k_t <= 0):
            continue

        beta_hat.iloc[i] = theta1 / k_t
        gamma_hat.iloc[i] = -theta2

    return beta_hat, gamma_hat


# ------------------------------------------------------------
# 4) ONE-STEP-AHEAD FORECAST
# ------------------------------------------------------------
def one_step_forecast(rho_obs, beta_hat, gamma_hat, k_mean):
    """
    One-step-ahead forecast using fitted beta(t), gamma(t):

      rho_hat(t+1|t) = rho(t) + beta(t)*k(t)*rho(t)(1-rho(t)) - gamma(t)*rho(t)

    Returns:
      rho_fc indexed by time t (forecast origin).
    """
    # Use only dates where we have beta and gamma estimates
    idx = rho_obs.index.intersection(beta_hat.dropna().index).intersection(gamma_hat.dropna().index)
    idx = idx.sort_values()

    rho_fc = pd.Series(index=idx, dtype=float)

    for t in idx:
        rho_t = float(rho_obs.loc[t])
        b = float(beta_hat.loc[t])
        g = float(gamma_hat.loc[t])
        k = float(k_mean.loc[t])

        pred = rho_t + (b * k * rho_t * (1.0 - rho_t) - g * rho_t)

        # Keep forecast inside [0, 1]
        rho_fc.loc[t] = float(np.clip(pred, 0.0, 1.0))

    return rho_fc


# ------------------------------------------------------------
# 5) MAIN PROGRAM
# ------------------------------------------------------------
def main():
    # ---- Step 1: Download prices and compute returns ----
    prices = download_prices(TICKERS, START, END)
    returns = compute_log_returns(prices)

    # ---- Step 2: Compute volatility ----
    sigma = rolling_volatility(returns, VOL_WINDOW)

    # Sector volatility = average vol across all tickers
    sigma_sector = sigma.mean(axis=1)

    # ---- Step 3: Create a volatility threshold (EMA) ----
    theta = ema(sigma_sector, EMA_SPAN)

    # ---- Step 4: Create infection panel X (0/1) ----
    # X[i,t] = 1 if ticker volatility > threshold at time t
    X = (sigma.gt(theta, axis=0)).astype(int)

    # Observed prevalence rho(t) = mean infected at time t
    rho_obs = X.mean(axis=1).dropna()

    # ---- Step 5: Mean degree proxy ----
    # Here we use a constant "fully connected" proxy: k = (N-1).
    # You can replace this later with time-varying mean degree from a correlation network.
    k_const = len(TICKERS) - 1
    k_mean = pd.Series(k_const, index=rho_obs.index, name="k_mean")

    # ---- Step 6: Rolling estimates of beta(t), gamma(t) ----
    beta_hat, gamma_hat = rolling_sis_params(rho_obs, k_mean, window_W=ROLL_W)

    usable = beta_hat.dropna().index.intersection(gamma_hat.dropna().index)
    if len(usable) < 10:
        raise RuntimeError(
            "Too few usable beta/gamma estimates.\n"
            "Try reducing ROLL_W or using a longer sample."
        )

    # ---- Step 7: One-step-ahead forecasts ----
    rho_fc = one_step_forecast(rho_obs, beta_hat, gamma_hat, k_mean)

    # Compare rho_fc(t) against realised rho(t+1)
    realised_next = rho_obs.shift(-1).loc[rho_fc.index]

    # Drop final points where realised_next is missing
    ok = realised_next.notna()
    rho_fc = rho_fc.loc[ok]
    realised_next = realised_next.loc[ok]

    # Forecast errors and RMSE
    error = rho_fc - realised_next
    rmse = float(np.sqrt(np.mean(error ** 2)))
    print(f"One-step-ahead RMSE: {rmse:.4f}")

    # --------------------------------------------------------
    # 6) PLOTS (shown, not saved)
    # --------------------------------------------------------

    # Plot 1: Forecast vs realised next-step
    plt.figure(figsize=(12, 4))
    plt.plot(realised_next.index, realised_next.values, label="Realised (t+1)", color="black")
    plt.plot(rho_fc.index, rho_fc.values, label="SIS forecast (t+1|t)", color="hotpink", linestyle="--")
    plt.title("One-step-ahead SIS forecasts vs realised prevalence")
    plt.xlabel("Date")
    plt.ylabel("Infection fraction")
    plt.ylim(-0.02, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot 2: Forecast error over time
    plt.figure(figsize=(12, 3.5))
    plt.plot(error.index, error.values, color="firebrick")
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.title("One-step-ahead forecast error (forecast − realised)")
    plt.xlabel("Date")
    plt.ylabel("Forecast error")
    plt.tight_layout()
    plt.show()

    # Plot 3: Error magnitude vs volatility (optional)
    abs_err = error.abs()
    vol_align = sigma_sector.loc[abs_err.index]

    plt.figure(figsize=(6, 4))
    plt.scatter(vol_align.values, abs_err.values)
    plt.title("Forecast error vs realised sector volatility")
    plt.xlabel("Sector volatility")
    plt.ylabel("Absolute forecast error")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
