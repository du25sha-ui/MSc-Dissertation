

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


# ------------------------------------------------------------
# 0) SETTINGS (change these if needed)
# ------------------------------------------------------------
N = 200
p_list = [0.03, 0.05, 0.08, 0.12]  # ER densities to compare

gamma = 1.0       # recovery rate
beta_ODE = 0.8    # transmission parameter in the ODE

T = 40.0          # simulation horizon (time)
dt_record = 0.2   # record prevalence every dt_record time units

n_reps = 50               # Monte Carlo repetitions per density
init_infected_frac = 0.05 # initial infected fraction (5%)

SAVE_FIG = True
FIG_PATH = "scaling_validation.png"


# ------------------------------------------------------------
# 1) GILLESPIE SIS SIMULATION (continuous-time, exact)
# ------------------------------------------------------------
def gillespie_sis(G, beta, gamma, T, init_infected_frac=0.05, dt_record=0.2, rng=None):
    """
    Simulate SIS on a network using Gillespie's direct method.

    States:
      X[i] = 0 means Susceptible
      X[i] = 1 means Infected

    Events:
      - Recovery: infected node i recovers at rate gamma
      - Infection: susceptible node i is infected at rate beta * (# infected neighbours)

    Returns:
      t_grid: times where we record prevalence
      I_grid: fraction infected at those times
    """
    if rng is None:
        rng = np.random.default_rng()

    # Convert node labels -> 0..N-1 indices (faster arrays)
    nodes = list(G.nodes())
    node_to_idx = {u: i for i, u in enumerate(nodes)}
    N_local = len(nodes)

    # Neighbour list stored as indices
    neigh = []
    for u in nodes:
        neigh.append(np.array([node_to_idx[v] for v in G.neighbors(u)], dtype=int))

    # Initial state: all susceptible
    X = np.zeros(N_local, dtype=np.int8)

    # Infect a fraction of nodes at t=0
    n0 = max(1, int(round(init_infected_frac * N_local)))
    infected0 = rng.choice(N_local, size=n0, replace=False)
    X[infected0] = 1

    # Count how many infected neighbours each node has
    inf_nbrs = np.zeros(N_local, dtype=int)
    for i in np.where(X == 1)[0]:
        inf_nbrs[neigh[i]] += 1

    # Recording grid (fixed time points)
    t_grid = np.arange(0.0, T + 1e-12, dt_record)
    I_grid = np.zeros_like(t_grid)

    t = 0.0
    rec_ptr = 0  # where we are on the record grid

    def record_until(time_now):
        """Fill I_grid values up to time_now."""
        nonlocal rec_ptr
        while rec_ptr < len(t_grid) and t_grid[rec_ptr] <= time_now + 1e-15:
            I_grid[rec_ptr] = X.mean()  # fraction infected
            rec_ptr += 1

    # Record at time 0
    record_until(t)

    # Gillespie loop (jump from event to event)
    while t < T:
        infected = np.where(X == 1)[0]
        susceptible = np.where(X == 0)[0]

        # Recovery rates (one per infected node)
        rec_rates = gamma * np.ones(len(infected), dtype=float)

        # Infection rates (one per susceptible node)
        inf_rates = beta * inf_nbrs[susceptible].astype(float)

        # Total rate
        R = rec_rates.sum() + inf_rates.sum()
        if R <= 0:
            # No events can occur anymore
            record_until(T)
            break

        # Time to next event
        dt = rng.exponential(1.0 / R)
        t_next = t + dt

        # Record prevalence up to the next event
        record_until(min(t_next, T))

        # Choose event type (recovery vs infection)
        u = rng.random() * R
        total_rec = rec_rates.sum()

        if u < total_rec:
            # RECOVERY event: pick which infected node recovers
            j = np.searchsorted(np.cumsum(rec_rates), u)
            node = infected[j]
            X[node] = 0
            inf_nbrs[neigh[node]] -= 1  # node no longer infects neighbours
        else:
            # INFECTION event: pick which susceptible node gets infected
            u2 = u - total_rec
            cum_inf = np.cumsum(inf_rates)

            # Safety check
            if cum_inf[-1] <= 0:
                t = t_next
                continue

            j = np.searchsorted(cum_inf, u2)
            node = susceptible[j]
            X[node] = 1
            inf_nbrs[neigh[node]] += 1  # node now infects neighbours

        t = t_next

    # Fill any remaining record points with last value
    if rec_ptr < len(t_grid):
        I_grid[rec_ptr:] = X.mean()

    return t_grid, I_grid


# ------------------------------------------------------------
# 2) MEAN-FIELD ODE CURVE (Euler method)
# ------------------------------------------------------------
def mean_field_ode(beta_ODE, gamma, rho0, t_grid):
    """
    Solve:
      dρ/dt = beta_ODE * ρ(1-ρ) - gamma * ρ
    using Euler steps on the same t_grid.
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
# 3) MAIN PROGRAM
# ------------------------------------------------------------
def main():
    rng = np.random.default_rng(0)

    # Common time grid
    t_grid = np.arange(0.0, T + 1e-12, dt_record)

    # ODE reference curve
    rho0 = init_infected_frac
    rho_ode = mean_field_ode(beta_ODE=beta_ODE, gamma=gamma, rho0=rho0, t_grid=t_grid)

    # Create figure
    plt.figure(figsize=(9, 5))

    # Plot ODE curve (dashed black)
    plt.plot(
        t_grid, rho_ode,
        linestyle="--", color="black", linewidth=2.0,
        label="Mean-field ODE"
    )

    # Run scaled simulations for each density p
    for p in p_list:
        # Build ER graph
        G = nx.erdos_renyi_graph(n=N, p=p, seed=1)

        # Mean degree <k>
        k_mean = np.mean([d for _, d in G.degree()])
        if k_mean <= 0:
            continue

        # Scaling rule
        beta_net = beta_ODE / k_mean

        # Average prevalence across many runs
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

        # Plot averaged simulation curve
        plt.plot(
            t_grid, I_avg,
            linewidth=1.6,
            label=rf"ER $p={p:.2f}$, $\langle k\rangle \approx {k_mean:.1f}$"
        )

    # Titles and labels
    plt.title(r"Scaling validation on ER networks: $\beta_{\rm net}=\beta_{\rm ODE}/\langle k\rangle$")
    plt.xlabel("Time")
    plt.ylabel("Prevalence (fraction infected)")
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()

    # Save if wanted
    if SAVE_FIG:
        plt.savefig(FIG_PATH, dpi=300)
        print(f"Saved figure to: {FIG_PATH}")

    plt.show()


if __name__ == "__main__":
    main()
