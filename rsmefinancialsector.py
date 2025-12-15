
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


# ------------------------------------------------------------
# 0) SETTINGS (change these if you like)
# ------------------------------------------------------------
N = 200                               # number of nodes
p_list = np.linspace(0.01, 0.20, 12)  # ER densities to test

gamma = 1.0                           # recovery rate
beta_ODE = 0.8                        # mean-field infection rate (ODE)

T = 40.0                              # total simulation time
dt_record = 0.2                       # record prevalence every dt_record units

n_reps = 50                           # number of Monte Carlo simulations per p
init_infected_frac = 0.05             # initial infected fraction (5%)

SAVE_FIG = True
FIG_PATH = "scaling_rmse.png"


# ------------------------------------------------------------
# 1) GILLESPIE SIS SIMULATION (continuous-time Markov chain)
# ------------------------------------------------------------
def gillespie_sis(G, beta, gamma, T, init_infected_frac=0.05, dt_record=0.2, rng=None):
    """
    Simulate SIS on a network using Gillespie's direct method.

    States:
      X[i] = 0 means node i is Susceptible
      X[i] = 1 means node i is Infected

    Events:
      - Recovery: infected node recovers at rate gamma
      - Infection: susceptible node becomes infected at rate beta * (# infected neighbours)

    Returns:
      t_grid (times where we record)
      I_grid (fraction infected at those times)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Make an ordered list of nodes and map node -> index 0..N-1
    nodes = list(G.nodes())
    node_to_idx = {u: i for i, u in enumerate(nodes)}
    N_local = len(nodes)

    # Neighbour list as indices (faster to update)
    neigh = []
    for u in nodes:
        neigh.append(np.array([node_to_idx[v] for v in G.neighbors(u)], dtype=int))

    # Initial infection state vector (0/1)
    X = np.zeros(N_local, dtype=np.int8)

    n0 = max(1, int(round(init_infected_frac * N_local)))
    infected0 = rng.choice(N_local, size=n0, replace=False)
    X[infected0] = 1

    # inf_nbrs[i] = number of infected neighbours of node i
    inf_nbrs = np.zeros(N_local, dtype=int)
    for i in np.where(X == 1)[0]:
        inf_nbrs[neigh[i]] += 1

    # Times at which we record prevalence
    t_grid = np.arange(0.0, T + 1e-12, dt_record)
    I_grid = np.zeros_like(t_grid)

    t = 0.0
    rec_ptr = 0  # where we are in t_grid

    # Record prevalence values until a given time
    def record_until(time_now):
        nonlocal rec_ptr
        while rec_ptr < len(t_grid) and t_grid[rec_ptr] <= time_now + 1e-15:
            I_grid[rec_ptr] = X.mean()  # fraction infected
            rec_ptr += 1

    # Record at t=0
    record_until(t)

    # Main Gillespie loop
    while t < T:
        infected = np.where(X == 1)[0]
        susceptible = np.where(X == 0)[0]

        # Recovery rates: gamma for each infected node
        rec_rates = gamma * np.ones(len(infected), dtype=float)

        # Infection rates: beta * (infected neighbour count) for each susceptible node
        inf_rates = beta * inf_nbrs[susceptible].astype(float)

        # Total event rate
        R = rec_rates.sum() + inf_rates.sum()
        if R <= 0:
            # No more events can happen
            record_until(T)
            break

        # Time to next event (Exponential with mean 1/R)
        dt = rng.exponential(1.0 / R)
        t_next = t + dt

        # Record prevalence up to the next event time
        record_until(min(t_next, T))

        # Choose which event happens
        u = rng.random() * R

        if u < rec_rates.sum():
            # A recovery event happens
            j = np.searchsorted(np.cumsum(rec_rates), u)
            node = infected[j]
            X[node] = 0
            inf_nbrs[neigh[node]] -= 1

        else:
            # An infection event happens
            u2 = u - rec_rates.sum()
            cinf = np.cumsum(inf_rates)

            # Safety check
            if cinf[-1] <= 0:
                t = t_next
                continue

            j = np.searchsorted(cinf, u2)
            node = susceptible[j]
            X[node] = 1
            inf_nbrs[neigh[node]] += 1

        t = t_next

    # If we didn't fill the whole grid, fill remaining with last prevalence
    if rec_ptr < len(t_grid):
        I_grid[rec_ptr:] = X.mean()

    return t_grid, I_grid


# ------------------------------------------------------------
# 2) MEAN-FIELD ODE SOLUTION (Euler method on same time grid)
# ------------------------------------------------------------
def mean_field_ode(beta_ODE, gamma, rho0, t_grid):
    """
    Solve the mean-field SIS ODE using simple Euler steps:

      dρ/dt = beta_ODE * ρ(1-ρ) - gamma * ρ

    Returns:
      rho(t) values on t_grid
    """
    rho = np.zeros_like(t_grid)
    rho[0] = rho0

    for k in range(1, len(t_grid)):
        dt = t_grid[k] - t_grid[k - 1]
        r = rho[k - 1]
        dr = beta_ODE * r * (1.0 - r) - gamma * r
        rho[k] = np.clip(r + dt * dr, 0.0, 1.0)

    return rho


# ------------------------------------------------------------
# 3) RMSE FUNCTION
# ------------------------------------------------------------
def rmse(a, b):
    """Root Mean Squared Error between two same-length arrays."""
    a = np.asarray(a)
    b = np.asarray(b)
    mask = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2)))


# ------------------------------------------------------------
# 4) MAIN PROGRAM
# ------------------------------------------------------------
def main():
    rng = np.random.default_rng(0)

    # Create the common time grid and ODE curve once (same for all p)
    t_grid = np.arange(0.0, T + 1e-12, dt_record)
    rho_ode = mean_field_ode(beta_ODE=beta_ODE, gamma=gamma,
                             rho0=init_infected_frac, t_grid=t_grid)

    # We will store results here
    k_vals = []
    rmse_vals = []

    # Loop over different ER densities
    for p in p_list:
        # Build an ER graph with N nodes and edge probability p
        G = nx.erdos_renyi_graph(n=N, p=float(p), seed=1)

        # Compute mean degree <k>
        k_mean = np.mean([d for _, d in G.degree()])
        if k_mean <= 0:
            continue

        # Degree scaling rule: beta_net = beta_ODE / <k>
        beta_net = beta_ODE / k_mean

        # Run Monte Carlo simulations and average them
        I_mat = np.zeros((n_reps, len(t_grid)))

        for r in range(n_reps):
            _, I = gillespie_sis(
                G=G,
                beta=beta_net,
                gamma=gamma,
                T=T,
                init_infected_frac=init_infected_frac,
                dt_record=dt_record,
                rng=rng
            )
            I_mat[r, :] = I

        I_avg = I_mat.mean(axis=0)

        # Compute RMSE vs the ODE curve
        error_val = rmse(I_avg, rho_ode)

        k_vals.append(k_mean)
        rmse_vals.append(error_val)

    # Convert lists to arrays (easier to plot)
    k_vals = np.asarray(k_vals)
    rmse_vals = np.asarray(rmse_vals)

    # Plot RMSE vs mean degree
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(k_vals, rmse_vals, marker="o", linewidth=1.8)
    ax.set_title(
        r"RMSE across ER densities under degree scaling "
        r"($\beta_{\rm net}=\beta_{\rm ODE}/\langle k\rangle$)"
    )
    ax.set_xlabel(r"Mean degree $\langle k\rangle$")
    ax.set_ylabel("RMSE (simulation vs ODE)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    # Save if requested
    if SAVE_FIG:
        fig.savefig(FIG_PATH, dpi=300)
        print(f"Saved figure to: {FIG_PATH}")

    plt.show()


if __name__ == "__main__":
    main()
