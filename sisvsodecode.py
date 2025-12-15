
import os
import math
import random

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# -----------------------------
# 1) SIS ODE
# -----------------------------
def sis_ode_rhs(t, i, beta, gamma):
    """
    Right-hand side of the SIS ODE:

        di/dt = beta * i * (1 - i) - gamma * i

    where:
      i     = fraction infected (between 0 and 1)
      beta  = infection rate
      gamma = recovery rate

    Note: 't' is required by solve_ivp even if we don't use it directly.
    """
    return beta * i * (1 - i) - gamma * i


def solve_sis_ode(beta, gamma, I0, N, steps=200):
    """
    Solve the SIS ODE from time 0 to 'steps' (integer times).

    Inputs:
      beta, gamma : ODE parameters
      I0          : initial infected count (people/nodes)
      N           : population size (used to convert I0 -> fraction)
      steps       : number of time steps (we evaluate at 0,1,2,...,steps)

    Returns:
      t_ode : time points (0..steps)
      i_ode : infected fraction at each time
    """
    # We'll evaluate at integer times: 0, 1, 2, ..., steps
    t_eval = np.arange(steps + 1)

    # Initial infected fraction
    i0_fraction = I0 / N

    sol = solve_ivp(
        fun=sis_ode_rhs,             # differential equation
        t_span=(0, steps),           # time interval
        y0=[i0_fraction],            # initial condition (as a list/array)
        args=(beta, gamma),          # extra parameters passed into sis_ode_rhs
        t_eval=t_eval,               # times where we want results
        rtol=1e-8, atol=1e-8         # tighter tolerances for accuracy
    )

    # sol.y[0] is the first (and only) state variable i(t)
    return sol.t, sol.y[0]


# -----------------------------
# 2) SIS simulation on a network
#    (with "dense scaling")
# -----------------------------
def run_sis(G, beta, gamma, steps=200, I0=5, seed=None):
    """
    Run ONE stochastic SIS simulation on graph G.

    Idea:
      - Each node is either Susceptible 'S' or Infected 'I'
      - At each time step:
          * Infected nodes may recover with probability p_rec
          * Susceptible nodes may become infected depending on how many
            infected neighbors they have.

    Dense scaling:
      For dense graphs, we scale infection probability using:
        tau = beta / kbar
      where kbar = average degree, so infection pressure stays comparable
      when N changes.

    Returns:
      infected_fraction_history : numpy array of length (steps+1)
    """
    rng = random.Random(seed)

    N = G.number_of_nodes()
    nodes = list(G.nodes())

    # Start with everyone susceptible
    states = {u: "S" for u in nodes}

    # Infect I0 random nodes (or fewer if N < I0)
    for u in rng.sample(nodes, k=min(I0, N)):
        states[u] = "I"

    # Average degree kbar (mean number of neighbors)
    kbar = np.mean([deg for _, deg in G.degree()])

    # Infection "per-contact" scaling
    tau = beta / kbar

    # Recovery probability per step (derived from rate gamma)
    p_rec = 1 - math.exp(-gamma)

    # Store infected counts over time
    infected_count = sum(s == "I" for s in states.values())
    I_hist = [infected_count]

    # Main time loop
    for _ in range(steps):
        # Copy current states so updates happen "simultaneously"
        new_states = states.copy()

        for u in nodes:
            if states[u] == "I":
                # Infected node may recover
                if rng.random() < p_rec:
                    new_states[u] = "S"
            else:
                # Susceptible node: check infected neighbors
                infected_neighbors = sum(states[v] == "I" for v in G.neighbors(u))

                if infected_neighbors > 0:
                    # Infection probability based on # infected neighbors
                    # (again using rate-style probability conversion)
                    p_inf = 1 - math.exp(-tau * infected_neighbors)

                    if rng.random() < p_inf:
                        new_states[u] = "I"

        # Commit all updates
        states = new_states

        infected_count = sum(s == "I" for s in states.values())
        I_hist.append(infected_count)

    # Convert counts -> fraction infected
    return np.array(I_hist) / N


def average_sis(G, beta, gamma, steps=200, I0=5, runs=80):
    """
    Run the simulation multiple times and average the infected fraction.
    This reduces random noise.

    Returns:
      mean_infected_fraction_history
    """
    all_runs = []
    for i in range(runs):
        sim = run_sis(G, beta, gamma, steps=steps, I0=I0, seed=i)
        all_runs.append(sim)

    return np.mean(all_runs, axis=0)


# -----------------------------
# 3) Graph builders (dense networks)
# -----------------------------
def dense_ER(N):
    """Dense Erdős–Rényi random graph (p=0.8)."""
    return nx.erdos_renyi_graph(N, 0.8, seed=42)


def dense_WS(N):
    """
    Dense Watts–Strogatz small-world graph.
    k is chosen ~ 0.8N (but must be even and < N).
    """
    k = int(0.8 * N)
    if k % 2 == 1:
        k += 1
    if k >= N:
        k = N - 2

    return nx.watts_strogatz_graph(N, k, 0.3, seed=42)


def dense_BA(N):
    """
    Dense Barabási–Albert preferential attachment graph.
    m is chosen ~ 0.5N (but must be < N).
    """
    m = int(0.5 * N)
    if m >= N:
        m = N - 2

    return nx.barabasi_albert_graph(N, m, seed=42)


def dense_Lattice(N):
    """
    "Dense lattice-like" graph on a 2D torus (wrap-around grid),
    then connect nodes within a large Chebyshev radius.

    Steps:
      1) Build n x n periodic grid (n ~ sqrt(N)).
      2) Add extra edges within radius R.
      3) Relabel nodes to integers.
      4) If we made more than N nodes, keep the first N.
    """
    n = int(math.ceil(math.sqrt(N)))

    # Periodic grid (a torus)
    G = nx.grid_2d_graph(n, n, periodic=True)

    # Big radius to make it dense
    R = int(0.9 * math.sqrt(N))

    for x in range(n):
        for y in range(n):
            for dx in range(-R, R + 1):
                for dy in range(-R, R + 1):
                    if dx == 0 and dy == 0:
                        continue
                    # Chebyshev distance constraint
                    if max(abs(dx), abs(dy)) <= R:
                        G.add_edge((x, y), ((x + dx) % n, (y + dy) % n))

    # Convert (x,y) nodes -> 0..(n*n-1)
    G = nx.convert_node_labels_to_integers(G)

    # If we created too many nodes, cut down to N
    if n * n > N:
        return G.subgraph(range(N)).copy()

    return G


# -----------------------------
# 4) Plot ODE vs simulation for one topology
#    and compute RMSE
# -----------------------------
def plot_topology(name, builder, beta=0.25, gamma=0.10, Ns=(50, 100, 200)):
    """
    For a given topology builder (function that makes a graph of size N):
      - run simulation (averaged)
      - solve ODE
      - plot both for each N
      - compute RMSE between the two curves
      - save the figure to results/<name>_separate.png

    Returns:
      list of dict records (for RMSE table)
    """
    fig, axes = plt.subplots(1, len(Ns), figsize=(15, 4), sharey=True)

    rmse_records = []

    for idx, N in enumerate(Ns):
        # Build graph and run simulation average
        G = builder(N)
        i_sim = average_sis(G, beta, gamma, steps=200, I0=5, runs=80)
        t_sim = np.arange(201)  # 0..200

        # Solve ODE
        t_ode, i_ode = solve_sis_ode(beta, gamma, I0=5, N=N, steps=200)

        # RMSE (root mean squared error) over all time points
        rmse = np.sqrt(np.mean((i_sim - i_ode) ** 2))

        rmse_records.append({"topology": name, "N": N, "RMSE": rmse})

        # Plot
        ax = axes[idx]
        ax.plot(t_ode, i_ode, "k--", lw=2, label="ODE")
        ax.plot(t_sim, i_sim, lw=2, label="Simulation")

        ax.set_title(f"{name}, N={N}\nRMSE={rmse:.4f}")
        ax.set_xlabel("Time")
        if idx == 0:
            ax.set_ylabel("Fraction infected")
        ax.grid(alpha=0.3)

    # Put one shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    fig.suptitle(f"SIS vs ODE — {name}", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    # Save figure
    os.makedirs("results", exist_ok=True)
    fig.savefig(f"results/{name}_separate.png", dpi=200)

    plt.show()

    return rmse_records


# -----------------------------
# 5) Lattice convergence experiment:
#    vary the neighborhood radius R
# -----------------------------
def lattice_with_radius(N, R):
    """
    Build a 2D periodic grid (torus) and connect nodes within Chebyshev radius R.

    Chebyshev radius R means:
      max(|dx|, |dy|) <= R
    """
    n = int(math.ceil(math.sqrt(N)))
    G = nx.grid_2d_graph(n, n, periodic=True)

    for x in range(n):
        for y in range(n):
            for dx in range(-R, R + 1):
                for dy in range(-R, R + 1):
                    if dx == 0 and dy == 0:
                        continue
                    if max(abs(dx), abs(dy)) <= R:
                        G.add_edge((x, y), ((x + dx) % n, (y + dy) % n))

    G = nx.convert_node_labels_to_integers(G)

    # Trim if too large
    if n * n > N:
        G = G.subgraph(range(N)).copy()

    return G


def plot_lattice_convergence(beta=0.25, gamma=0.10,
                             N=200, R_list=(1, 2, 3, 5, 8), steps=200, runs=80):
    """
    Show how the lattice simulation approaches the ODE as R increases.
    Also compute RMSE for each R.

    Saves figure to results/Lattice_convergence.png

    Returns:
      list of dict records (for RMSE table)
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    # ODE curve
    t_ode, i_ode = solve_sis_ode(beta, gamma, I0=5, N=N, steps=steps)
    ax.plot(t_ode, i_ode, "k--", lw=2, label="ODE")

    rmse_records = []

    for R in R_list:
        G = lattice_with_radius(N, R)

        i_sim = average_sis(G, beta, gamma, steps=steps, I0=5, runs=runs)
        t_sim = np.arange(steps + 1)

        avg_deg = np.mean([d for _, d in G.degree()])
        ax.plot(t_sim, i_sim, lw=2, label=f"R={R} (deg≈{avg_deg:.1f})")

        rmse = np.sqrt(np.mean((i_sim - i_ode) ** 2))
        rmse_records.append({"topology": "Lattice_convergence", "N": N, "R": R, "RMSE": rmse})

    ax.set_xlabel("Time")
    ax.set_ylabel("Fraction infected")
    ax.set_title(f"Lattice convergence to ODE (N={N})")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()

    os.makedirs("results", exist_ok=True)
    fig.savefig("results/Lattice_convergence.png", dpi=200)

    plt.show()

    return rmse_records


# -----------------------------
# 6) Run everything + save RMSE table
# -----------------------------
if __name__ == "__main__":
    all_rmse_records = []

    # Compare ODE vs simulation for each topology
    all_rmse_records += plot_topology("ER", dense_ER)
    all_rmse_records += plot_topology("WS", dense_WS)
    all_rmse_records += plot_topology("BA", dense_BA)
    all_rmse_records += plot_topology("Lattice", dense_Lattice)

    # Extra: lattice convergence test (increasing neighborhood radius)
    all_rmse_records += plot_lattice_convergence(N=200, R_list=[1, 2, 3, 5, 8])

    # Save RMSE results to CSV
    df_rmse = pd.DataFrame(all_rmse_records)
    os.makedirs("results", exist_ok=True)
    df_rmse.to_csv("results/sis_vs_ode_rmse.csv", index=False)

    # Print to console so you can see it immediately
    print(df_rmse)

