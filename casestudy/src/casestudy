
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path


# ============================================================
# 0) SETTINGS / CONFIG
# ============================================================

START_YEAR = 1983
END_YEAR = 2009
YEARS = np.arange(START_YEAR, END_YEAR + 1)

# Conflict timeline (used only for plotting + shaping synthetic data)
WAR_START = 1983
WAR_END = 2009
CEASEFIRE_YEAR = 2002
WAR4_YEAR = 2006
WAR_END_EVENT = 2009

# Random generator (fixed seed so results are repeatable)
RNG = np.random.default_rng(7)

# Stress threshold: 75th percentile
THRESHOLD_Q = 0.75

# Tamil region “penalties” and volatility factor (synthetic)
DELTA_GDP = 0.03
TE_VOL_MULT = 1.35

# SIS fitting settings (grid search)
N_MC = 500
BETA_GRID = np.linspace(0.05, 1.20, 48)
GAMMA_GRID = np.linspace(0.05, 1.20, 48)

# Output folder
OUTDIR = Path("sri_case_outputs")
OUTDIR.mkdir(exist_ok=True)

# If True, plots pop up. If False, plots just save to disk.
SHOW_PLOTS = False

# Colors
SL_COLOR = "black"
TE_COLOR = "red"
EXT_COLOR = "#2b6cb0"
GRID_ALPHA = 0.25


# ============================================================
# 1) PLOT STYLE
# ============================================================

def set_style():
    """Set matplotlib style so figures look consistent and thesis-safe."""
    plt.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "axes.grid": True,
        "grid.alpha": GRID_ALPHA,
    })


# ============================================================
# 2) CREATE SYNTHETIC MACRO DATA
# ============================================================

def generate_sri_lanka_macro(years):
    """
    Create synthetic macro data for Sri Lanka:
      - gdp_growth
      - inflation
      - fx_dep
      - debt_ratio

    These are “stylised” and not official data.
    """
    n = len(years)
    df = pd.DataFrame(index=pd.Index(years, name="year"))

    # Helper “flags” (0 or 1) for war periods
    war = ((years >= WAR_START) & (years <= WAR_END)).astype(float)
    war4 = ((years >= WAR4_YEAR) & (years <= WAR_END_EVENT)).astype(float)
    post_ceasefire = (years >= CEASEFIRE_YEAR).astype(float)

    # GDP growth: trend + war penalty + some post-ceasefire improvement + noise
    base_growth = 4.2 + 0.7 * np.tanh((years - 1996) / 6)
    gdp_growth = base_growth - 1.2 * war + 0.8 * post_ceasefire + RNG.normal(0, 1.2, n)
    df["gdp_growth"] = gdp_growth

    # Inflation: higher during war, extra boost during late intense war4
    infl_base = 8.5 + 2.0 * np.sin((years - 1983) / 3.3)
    inflation = infl_base + 3.5 * war + 2.5 * war4 + RNG.normal(0, 2.0, n)
    df["inflation"] = np.clip(inflation, 0.5, None)

    # FX depreciation: higher during war and war4
    fx_base = 4.0 + 1.0 * np.tanh((years - 1992) / 5.5)
    fx_dep = fx_base + 2.3 * war + 1.8 * war4 + RNG.normal(0, 1.5, n)
    df["fx_dep"] = np.clip(fx_dep, -5.0, None)

    # Debt ratio: rises over time, extra war pressure, slight easing after ceasefire
    debt = 65 + 0.9 * (years - START_YEAR) + 8 * war - 5 * post_ceasefire + RNG.normal(0, 3.0, n)
    df["debt_ratio"] = np.clip(debt, 30, 140)

    return df


def generate_tamil_region_macro(sl_df):
    """
    Create a synthetic Tamil region (TE) series from the SL series:
      - GDP growth slightly lower and noisier during war
      - inflation/fx/debt more volatile
    """
    years = sl_df.index.values
    n = len(years)
    te = pd.DataFrame(index=sl_df.index)

    war = ((years >= WAR_START) & (years <= WAR_END)).astype(float)
    war4 = ((years >= WAR4_YEAR) & (years <= WAR_END_EVENT)).astype(float)

    te["gdp_growth"] = (
        (1 - DELTA_GDP) * sl_df["gdp_growth"].values
        - 0.6 * war
        + RNG.normal(0, 1.4, n)
    )

    te_infl = sl_df["inflation"].values + (1.8 * war + 1.2 * war4) + RNG.normal(0, 2.0 * TE_VOL_MULT, n)
    te["inflation"] = np.clip(te_infl, 0.5, None)

    te["fx_dep"] = sl_df["fx_dep"].values + (1.2 * war + 1.0 * war4) + RNG.normal(0, 1.4 * TE_VOL_MULT, n)

    te_debt = sl_df["debt_ratio"].values + (6.0 * war + 4.0 * war4) + RNG.normal(0, 4.0, n)
    te["debt_ratio"] = np.clip(te_debt, 30, 170)

    return te


# ============================================================
# 3) STRESS INDEX + BINARY STATE
# ============================================================

def compute_stress_index(df):
    """
    Build a stress index:
      - weighted combination of inflation, fx depreciation, and debt pressure
      - then standardise (z-score)
    """
    debt_centered = (df["debt_ratio"] - df["debt_ratio"].median()) / 10.0

    raw = (
        0.45 * df["inflation"]
        + 0.35 * df["fx_dep"]
        + 0.20 * debt_centered
    )

    # z-score: (x - mean) / std
    stress = (raw - raw.mean()) / (raw.std(ddof=0) + 1e-12)
    return stress.rename("stress_index")


def stress_to_state(stress_series, q=THRESHOLD_Q):
    """
    Convert stress index into a binary “infected” state:
      infected = 1 if stress > threshold
    """
    theta = float(np.quantile(stress_series.values, q))
    infected = (stress_series > theta).astype(int).rename("infected")
    return infected, theta


# ============================================================
# 4) BUILD NETWORK + SIS SIMULATION
# ============================================================

def build_macro_network():
    """
    Build an illustrative macro network.

    Nodes:
      SL  = Sri Lanka
      TE  = Tamil region
      ROW = Rest of world
      RP  = Regional partners

    A is an adjacency/weight matrix (bigger = stronger connection).
    """
    nodes = ["SL", "TE", "ROW", "RP"]
    A = np.zeros((4, 4), dtype=float)

    # SL connected strongly to everyone
    A[0, 1] = A[1, 0] = 1.0
    A[0, 2] = A[2, 0] = 1.0
    A[0, 3] = A[3, 0] = 1.0

    # TE has weaker ties externally
    A[1, 3] = A[3, 1] = 0.4   # TE-RP weak tie
    # A[1, 2] = A[2, 1] = 0.1 # optional weak TE-ROW tie

    # RP connected moderately to ROW
    A[3, 2] = A[2, 3] = 0.6

    return nodes, A


def sis_one_step(x, A, beta, gamma, rng):
    """
    One step of a discrete-time stochastic SIS model.

    x[i] = 0 means node i is healthy (susceptible)
    x[i] = 1 means node i is stressed (infected)

    Infection probability for node i:
      p_inf[i] = 1 - exp(-beta * sum_j A[i,j]*x[j])

    Recovery probability for an infected node i:
      gamma (simple probability, not exp form here)
    """
    infection_pressure = A @ x
    lam = beta * infection_pressure
    p_inf = 1.0 - np.exp(-lam)

    x_next = x.copy()

    for i in range(len(x)):
        if x[i] == 0:
            # susceptible -> infected?
            x_next[i] = 1 if rng.random() < p_inf[i] else 0
        else:
            # infected -> recover?
            x_next[i] = 0 if rng.random() < gamma else 1

    return x_next


def simulate_sis(A, beta, gamma, T, x0, rng):
    """
    Simulate SIS for T time steps.
    Returns a T x N array of states.
    """
    x = x0.copy()
    history = np.zeros((T, len(x0)), dtype=int)
    history[0] = x

    for t in range(1, T):
        x = sis_one_step(x, A, beta, gamma, rng)
        history[t] = x

    return history


# ============================================================
# 5) ESTIMATE (BETA, GAMMA) USING GRID SEARCH
# ============================================================

def estimate_beta_gamma(X_obs, nodes, A):
    """
    Brute-force grid search:
    For each (beta, gamma):
      - run N_MC simulations
      - average prevalence across simulations
      - compute MSE vs observed prevalence
    Choose the pair with smallest MSE.
    """
    T = X_obs.shape[0]
    idx_sl = nodes.index("SL")
    idx_te = nodes.index("TE")

    # Observed prevalence = mean infected in SL and TE
    y_obs = X_obs[["SL", "TE"]].mean(axis=1).values.astype(float)

    # Initial state: set SL and TE based on first observed year
    x0 = np.zeros(len(nodes), dtype=int)
    x0[idx_sl] = int(X_obs["SL"].iloc[0])
    x0[idx_te] = int(X_obs["TE"].iloc[0])

    mse_grid = np.full((len(BETA_GRID), len(GAMMA_GRID)), np.nan)

    best_mse = np.inf
    best_beta = None
    best_gamma = None
    best_prev_hat = None

    for i, beta in enumerate(BETA_GRID):
        for j, gamma in enumerate(GAMMA_GRID):

            # Monte Carlo average prevalence curve
            prev_sum = np.zeros(T, dtype=float)

            for k in range(N_MC):
                sim = simulate_sis(
                    A=A,
                    beta=beta,
                    gamma=gamma,
                    T=T,
                    x0=x0,
                    rng=np.random.default_rng(1000 + k)  # reproducible seeds
                )

                prev = (sim[:, idx_sl] + sim[:, idx_te]) / 2.0
                prev_sum += prev

            prev_hat = prev_sum / N_MC

            mse = float(np.mean((prev_hat - y_obs) ** 2))
            mse_grid[i, j] = mse

            if mse < best_mse:
                best_mse = mse
                best_beta = float(beta)
                best_gamma = float(gamma)
                best_prev_hat = prev_hat

    return {
        "best": {
            "beta": best_beta,
            "gamma": best_gamma,
            "mse": best_mse,
            "prev_hat": best_prev_hat
        },
        "mse_grid": mse_grid,
        "beta_grid": BETA_GRID,
        "gamma_grid": GAMMA_GRID,
        "y_obs": y_obs,
    }


# ============================================================
# 6) PLOTTING HELPERS
# ============================================================

def save_or_show(fig, outfile):
    """Save plot to file, and optionally show it."""
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()

    plt.close(fig)


def plot_gdp_structural_change(years, sl_gdp, te_gdp, outfile):
    """GDP plot with ceasefire line and pre/post means."""
    years = np.asarray(years)

    fig, ax = plt.subplots(figsize=(10.5, 4.2))

    ax.plot(years, sl_gdp, color=SL_COLOR, marker="o", linewidth=1.8,
            label="Sri Lanka (annual GDP growth)")
    ax.plot(years, te_gdp, color=TE_COLOR, marker="s", linewidth=1.8,
            label="Tamil region (synthetic)")

    pre = years < CEASEFIRE_YEAR
    post = years >= CEASEFIRE_YEAR

    sl_pre_mean = np.mean(sl_gdp[pre])
    sl_post_mean = np.mean(sl_gdp[post])
    te_pre_mean = np.mean(te_gdp[pre])
    te_post_mean = np.mean(te_gdp[post])

    ax.axhline(sl_pre_mean, color=SL_COLOR, linestyle=":", linewidth=1.5, label="SL pre-2002 mean")
    ax.axhline(sl_post_mean, color=SL_COLOR, linestyle="--", linewidth=1.5, label="SL post-2002 mean")
    ax.axhline(te_pre_mean, color=TE_COLOR, linestyle=":", linewidth=1.5, label="TE pre-2002 mean")
    ax.axhline(te_post_mean, color=TE_COLOR, linestyle="--", linewidth=1.5, label="TE post-2002 mean")

    ax.axvline(CEASEFIRE_YEAR, color="black", linestyle="--", linewidth=1.5)
    ax.text(CEASEFIRE_YEAR + 0.1, ax.get_ylim()[0] + 0.05, "Ceasefire 2002",
            rotation=90, va="bottom")

    ax.set_title("Structural Change in GDP Growth: Pre-/Post-Ceasefire")
    ax.set_xlabel("Year")
    ax.set_ylabel("Real GDP growth (%)")
    ax.legend(ncol=2, frameon=True)

    save_or_show(fig, outfile)


def plot_timeline_and_states(years, x_sl, x_te, outfile):
    """Timeline plot (war shading) + step plot for SL/TE stress states."""
    fig, ax = plt.subplots(2, 1, figsize=(10.5, 5.5), sharex=True,
                           gridspec_kw={"height_ratios": [1, 2]})

    # Top panel: timeline shading + event lines
    ax[0].axvspan(WAR_START, WAR_END, alpha=0.15, label="Civil war period")
    for year, label in [(CEASEFIRE_YEAR, "Ceasefire 2002"), (WAR4_YEAR, "Eelam War IV"), (WAR_END_EVENT, "End 2009")]:
        ax[0].axvline(year, linestyle="--", linewidth=1, color="black")
        ax[0].text(year + 0.1, 0.02, label, rotation=90, va="bottom")
    ax[0].set_yticks([])
    ax[0].set_title("Conflict timeline and macroeconomic high-stress states (1983–2009)")
    ax[0].legend(loc="upper left", frameon=True)

    # Bottom panel: stress state steps
    ax[1].step(years, x_sl, where="mid", color=SL_COLOR, label="Sri Lanka (SL)")
    ax[1].step(years, x_te, where="mid", color=TE_COLOR, label="Tamil region (TE)")
    ax[1].set_yticks([0, 1], ["Healthy", "High stress"])
    ax[1].set_ylabel("State")
    ax[1].set_xlabel("Year")
    ax[1].legend(loc="upper right", frameon=True)

    save_or_show(fig, outfile)


def plot_stress_index(years, s_sl, th_sl, s_te, th_te, outfile):
    """Stress indices for SL and TE with thresholds."""
    fig, ax = plt.subplots(figsize=(10.5, 4.2))

    ax.plot(years, s_sl, color=SL_COLOR, label="SL stress index")
    ax.axhline(th_sl, color=SL_COLOR, linestyle="--", linewidth=1.3, label="SL threshold (75th pct)")

    ax.plot(years, s_te, color=TE_COLOR, label="TE stress index")
    ax.axhline(th_te, color=TE_COLOR, linestyle="--", linewidth=1.3, label="TE threshold (75th pct)")

    ax.axvspan(WAR_START, WAR_END, alpha=0.12)
    ax.set_title("Composite stress indices and region-specific thresholds")
    ax.set_xlabel("Year")
    ax.set_ylabel("Standardised stress index")
    ax.legend(ncol=2, frameon=True)

    save_or_show(fig, outfile)


def plot_model_vs_data(years, y_obs, y_hat, outfile):
    """Observed prevalence vs fitted SIS prevalence."""
    fig, ax = plt.subplots(figsize=(10.5, 4.0))

    ax.plot(years, y_obs, color=SL_COLOR, marker="o", linewidth=1.8,
            label="Empirical prevalence (mean of SL & TE)")
    ax.plot(years, y_hat, color=TE_COLOR, linestyle="--", linewidth=2.2,
            label="SIS model prevalence (MC mean)")

    ax.set_title("Macroeconomic contagion: model vs data")
    ax.set_xlabel("Year")
    ax.set_ylabel("Prevalence (fraction infected)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=True)

    save_or_show(fig, outfile)


def plot_mse_heatmap(beta_grid, gamma_grid, mse_grid, outfile):
    """Heatmap showing MSE across beta/gamma grid."""
    fig, ax = plt.subplots(figsize=(7.2, 5.6))

    im = ax.imshow(
        mse_grid.T,
        origin="lower",
        aspect="auto",
        extent=[beta_grid[0], beta_grid[-1], gamma_grid[0], gamma_grid[-1]],
        cmap="viridis",
    )

    ax.set_title("Parameter fit surface (MSE)")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\gamma$")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("MSE")

    save_or_show(fig, outfile)


def plot_macro_network(nodes, A, outfile):
    """Draw the macro network graph with SL black, TE red, externals blue."""
    G = nx.Graph()
    for name in nodes:
        G.add_node(name)

    # Add edges where adjacency matrix is > 0
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if A[i, j] > 0:
                G.add_edge(nodes[i], nodes[j], weight=A[i, j])

    # Fixed positions for a clean figure
    pos = {"SL": (0.0, 0.0), "TE": (1.0, -0.6), "RP": (1.0, 0.7), "ROW": (2.0, 0.0)}

    # Node colors
    node_colors = []
    for n in G.nodes():
        if n == "SL":
            node_colors.append(SL_COLOR)
        elif n == "TE":
            node_colors.append(TE_COLOR)
        else:
            node_colors.append(EXT_COLOR)

    fig, ax = plt.subplots(figsize=(6.8, 3.8))

    # Draw edges (dashed style if TE is involved)
    for (u, v) in G.edges():
        w = G[u][v]["weight"]
        width = 1.3 + 3.8 * w
        style = "--" if ("TE" in (u, v)) else "-"

        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v)],
            width=width,
            style=style,
            edge_color="black",
            alpha=0.85,
            ax=ax
        )

    nx.draw_networkx_nodes(G, pos, node_size=1400, node_color=node_colors, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=11, font_color="white", ax=ax)

    ax.set_title("Macroeconomic network representation (illustrative)")
    ax.axis("off")

    save_or_show(fig, outfile)


# ============================================================
# 7) MAIN PROGRAM
# ============================================================

def main():
    set_style()

    # ----- Step A: Create synthetic macro data -----
    sl = generate_sri_lanka_macro(YEARS)
    te = generate_tamil_region_macro(sl)

    # ----- Step B: Stress index + binary states -----
    s_sl = compute_stress_index(sl)
    s_te = compute_stress_index(te)

    x_sl, th_sl = stress_to_state(s_sl, q=THRESHOLD_Q)
    x_te, th_te = stress_to_state(s_te, q=THRESHOLD_Q)

    # Store observed states (panel)
    X_obs = pd.DataFrame({"SL": x_sl.values, "TE": x_te.values}, index=YEARS)
    X_obs.to_csv(OUTDIR / "macro_states_panel.csv", index_label="year")

    # ----- Step C: Build network -----
    nodes, A = build_macro_network()

    # ----- Step D: Estimate SIS parameters -----
    est = estimate_beta_gamma(X_obs, nodes, A)
    best = est["best"]

    # Save estimation summary
    summary = {
        "beta_hat": best["beta"],
        "gamma_hat": best["gamma"],
        "mse_min": best["mse"],
        "threshold_q": THRESHOLD_Q,
        "mc_runs_per_pair": N_MC,
        "beta_grid_n": len(BETA_GRID),
        "gamma_grid_n": len(GAMMA_GRID),
    }
    pd.Series(summary).to_csv(OUTDIR / "sis_estimation_summary.csv")

    # ----- Step E: Save figures -----
    plot_gdp_structural_change(
        YEARS,
        sl["gdp_growth"].values,
        te["gdp_growth"].values,
        OUTDIR / "sri_te_gdp_ceasefire.png",
    )

    plot_timeline_and_states(
        YEARS, X_obs["SL"].values, X_obs["TE"].values,
        OUTDIR / "timeline_and_states.png"
    )

    plot_stress_index(
        YEARS, s_sl.values, th_sl, s_te.values, th_te,
        OUTDIR / "stress_index_thresholds.png"
    )

    plot_model_vs_data(
        YEARS, est["y_obs"], best["prev_hat"],
        OUTDIR / "model_vs_data_prevalence.png"
    )

    plot_mse_heatmap(
        est["beta_grid"], est["gamma_grid"], est["mse_grid"],
        OUTDIR / "mse_surface.png"
    )

    plot_macro_network(
        nodes, A, OUTDIR / "macro_network.png"
    )

    # ----- Step F: Print key results -----
    print("Saved outputs to:", OUTDIR.resolve())
    print(f"Best fit: beta = {best['beta']:.4f}, gamma = {best['gamma']:.4f}, MSE = {best['mse']:.6f}")


if __name__ == "__main__":
    main()
