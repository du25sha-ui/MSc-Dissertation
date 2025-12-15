
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import yfinance as yf


# ------------------------------------------------------------
# 0) USER SETTINGS (edit these)
# ------------------------------------------------------------
TICKERS = ["JPM", "BAC", "WFC", "C", "GS", "MS", "BK", "BLK", "AXP", "SCHW", "USB", "PNC"]
START = "2015-01-01"
END = "2025-01-01"

VOL_WINDOW = 20   # rolling volatility window (days)
EMA_SPAN = 250    # smoothing for threshold (EMA)
TAU = 0.60        # correlation threshold for the network
ROLL_W = 60       # rolling window for beta/gamma estimation


# ------------------------------------------------------------
# 1) DOWNLOAD PRICES AND BUILD RETURNS
# ------------------------------------------------------------
def download_prices(tickers, start, end):
    """
    Download prices from Yahoo Finance (Adj Close).
    Returns DataFrame: index=dates, columns=tickers
    """
    df = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False
    )

    # We expect a multi-column format and want Adjusted Close prices
    prices = df["Adj Close"]

    # If only one ticker, yfinance might return a Series
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    # Forward-fill missing values; remove fully empty rows
    prices = prices.dropna(how="all").ffill()
    return prices


def compute_returns(prices):
    """Log returns."""
    return np.log(prices).diff()


# ------------------------------------------------------------
# 2) VOLATILITY AND THRESHOLD
# ------------------------------------------------------------
def rolling_volatility(returns, window=20):
    """
    Annualised rolling volatility:
      sigma(t) = sqrt(252 * rolling variance of returns)
    """
    return np.sqrt(252.0 * returns.rolling(window).var())


def ema(series, span=250):
    """Exponential moving average (smooth threshold)."""
    return series.ewm(span=span, adjust=False).mean()


# ------------------------------------------------------------
# 3) BUILD NETWORK FROM CORRELATIONS
# ------------------------------------------------------------
def build_graph_from_corr(corr_df, tau=0.60):
    """
    Build an undirected graph:
      - nodes are tickers
      - connect i-j if corr(i,j) >= tau
      - store weight = correlation
    """
    tickers = list(corr_df.columns)

    G = nx.Graph()
    G.add_nodes_from(tickers)

    corr = corr_df.copy()
    np.fill_diagonal(corr.values, 0.0)

    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            w = corr.iloc[i, j]
            if pd.notna(w) and w >= tau:
                G.add_edge(tickers[i], tickers[j], weight=float(w))

    return G


# ------------------------------------------------------------
# 4) INFECTION STATS
# ------------------------------------------------------------
def infection_stats(X_df):
    """
    Given infection panel X (0/1):
    - infection fraction per firm (average across time)
    - rho(t): prevalence each day (average across firms)
    """
    inf_frac = X_df.mean(axis=0)  # per ticker
    rho_obs = X_df.mean(axis=1)   # per date
    return inf_frac, rho_obs


# ------------------------------------------------------------
# 5) ROLLING SIS PARAMETER ESTIMATION
# ------------------------------------------------------------
def rolling_sis_params(rho_obs, k_mean, W=60):
    """
    Rolling regression on:
      Δρ(t) = θ1 * ρ(t)(1-ρ(t)) + θ2 * ρ(t) + error

    Interpretation:
      θ1 = beta(t) * <k>(t)
      θ2 = -gamma(t)
    """
    rho = rho_obs.astype(float).dropna()

    # One-step change: rho(t+1) - rho(t)
    drho = rho.shift(-1) - rho

    data = pd.DataFrame({"rho": rho, "drho": drho, "k": k_mean}).dropna()

    beta_hat = pd.Series(index=data.index, dtype=float)
    gamma_hat = pd.Series(index=data.index, dtype=float)

    # Start from W so we have a full rolling window
    for i in range(W, len(data)):
        win = data.iloc[i - W:i]

        y = win["drho"].values
        x1 = (win["rho"] * (1.0 - win["rho"])).values
        x2 = win["rho"].values

        X = np.column_stack([x1, x2])

        # Skip ill-conditioned cases
        XtX = X.T @ X
        if np.linalg.cond(XtX) > 1e12:
            continue

        theta1, theta2 = np.linalg.solve(XtX, X.T @ y)

        k_t = float(data["k"].iloc[i])
        if np.isnan(k_t) or k_t <= 0:
            continue

        beta_hat.iloc[i] = theta1 / k_t
        gamma_hat.iloc[i] = -theta2

    return beta_hat, gamma_hat


def simulate_mean_field_discrete(rho0, beta_hat, gamma_hat, k_mean):
    """
    Discrete-time forward simulation:
      ρ(t+1) = ρ(t) + beta(t)<k>(t)ρ(t)(1-ρ(t)) - gamma(t)ρ(t)

    Returns:
      rho_model indexed by time (same index as beta/gamma/k overlap).
    """
    idx = beta_hat.index.intersection(gamma_hat.index).intersection(k_mean.index)
    idx = idx.sort_values()

    if len(idx) == 0:
        raise RuntimeError("No overlap in indices for beta/gamma/k series.")

    rho_model = pd.Series(index=idx, dtype=float)
    rho = float(rho0)

    for t in idx:
        b = beta_hat.loc[t]
        g = gamma_hat.loc[t]
        k = k_mean.loc[t]

        if pd.isna(b) or pd.isna(g) or pd.isna(k):
            rho_model.loc[t] = np.nan
            continue

        # store current rho, then step forward
        rho_model.loc[t] = rho
        rho = rho + (b * k * rho * (1.0 - rho) - g * rho)
        rho = float(np.clip(rho, 0.0, 1.0))

    return rho_model


# ------------------------------------------------------------
# 6) MAIN PIPELINE
# ------------------------------------------------------------
def main():
    # ----- Step 1: prices -> returns -----
    prices = download_prices(TICKERS, START, END)
    returns = compute_returns(prices)

    # ----- Step 2: rolling volatility -----
    sigma = rolling_volatility(returns, VOL_WINDOW)

    # Sector volatility = average across firms
    sigma_sector = sigma.mean(axis=1)

    # ----- Step 3: threshold (EMA) -----
    theta = ema(sigma_sector, EMA_SPAN)

    # ----- Step 4: infection panel -----
    # X[ticker,t] = 1 if sigma[ticker,t] > theta[t]
    X = (sigma.gt(theta, axis=0)).astype(int)

    firm_inf_frac, rho_obs = infection_stats(X)

    # Plot 1: infection fraction time series
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(rho_obs.index, rho_obs.values, color="black", linewidth=1.6)
    ax.set_title("Financial sector infection fraction")
    ax.set_xlabel("Date")
    ax.set_ylabel("Fraction infected")
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    plt.show()

    # ----- Step 5: correlation network from full-sample returns -----
    returns_clean = returns.dropna(how="any")
    corr_full = returns_clean.corr()
    G = build_graph_from_corr(corr_full, TAU)

    # Centralities
    degree = pd.Series(dict(G.degree()), name="degree").astype(float)
    betweenness = pd.Series(nx.betweenness_centrality(G, normalized=True), name="betweenness")

    # Combine firm infection fraction + network metrics
    metrics = pd.DataFrame({
        "infection_fraction": firm_inf_frac,
        "degree": degree,
        "betweenness": betweenness
    }).dropna()

    # ----- Plot 2: network plot (node size ∝ infection fraction) -----
    nodes = list(metrics.index)
    subG = G.subgraph(nodes).copy()

    inf = metrics["infection_fraction"].values
    sizes = 300 + 2500 * (inf - inf.min()) / (np.ptp(inf) + 1e-9)

    pos = nx.spring_layout(subG, seed=0)

    fig, ax = plt.subplots(figsize=(9, 6))
    nx.draw_networkx_edges(subG, pos, alpha=0.35, width=0.9, ax=ax)
    nx.draw_networkx_nodes(subG, pos, node_size=sizes, node_color="#1f77b4", ax=ax)
    nx.draw_networkx_labels(subG, pos, font_size=10, ax=ax)
    ax.set_title(
        rf"Financial sector network at $\tau={TAU:.2f}$" + "\n" +
        "Node size ∝ infection fraction"
    )
    ax.axis("off")
    fig.tight_layout()
    plt.show()

    # ----- Plot 3: degree vs infection -----
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(metrics["degree"], metrics["infection_fraction"])
    for tkr, row in metrics.iterrows():
        ax.annotate(
            tkr,
            (row["degree"], row["infection_fraction"]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=10
        )
    ax.set_title("Degree vs infection")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Infection fraction")
    fig.tight_layout()
    plt.show()

    # ----- Plot 4: betweenness vs infection -----
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(metrics["betweenness"], metrics["infection_fraction"])
    for tkr, row in metrics.iterrows():
        ax.annotate(
            tkr,
            (row["betweenness"], row["infection_fraction"]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=10
        )
    ax.set_title("Betweenness vs infection")
    ax.set_xlabel("Betweenness centrality")
    ax.set_ylabel("Infection fraction")
    fig.tight_layout()
    plt.show()

    # ----- Step 6: mean degree series <k>_t -----
    # Simple version: constant mean degree from the thresholded network
    k_const = float(np.mean([d for _, d in subG.degree()]))
    k_mean = pd.Series(k_const, index=rho_obs.index, name="k_mean")

    # ----- Step 7: rolling beta(t), gamma(t) -----
    beta_hat, gamma_hat = rolling_sis_params(rho_obs, k_mean, W=ROLL_W)

    usable = rho_obs.index.intersection(beta_hat.dropna().index).intersection(gamma_hat.dropna().index)
    if len(usable) == 0:
        print("WARNING: No usable beta/gamma estimates. Model-vs-data plot skipped.")
        return

    start_t = usable.min()
    rho0 = float(rho_obs.loc[start_t])

    rho_model = simulate_mean_field_discrete(
        rho0=rho0,
        beta_hat=beta_hat.dropna(),
        gamma_hat=gamma_hat.dropna(),
        k_mean=k_mean
    )

    # ----- Plot 5: model vs data (Empirical black, SIS pink dashed) -----
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(
        rho_obs.index, rho_obs.values,
        color="black", linewidth=1.8,
        label="Empirical"
    )

    ax.plot(
        rho_model.index, rho_model.values,
        color="#e377c2", linestyle="--", linewidth=2.0,
        label="SIS model (mean-field)"
    )

    ax.set_title("Financial sector: model vs data")
    ax.set_xlabel("Date")
    ax.set_ylabel("Infection fraction")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()


