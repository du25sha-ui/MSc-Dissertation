
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D


# ------------------------------------------------------------
# 0) Reproducibility: set random seed
# ------------------------------------------------------------
np.random.seed(42)

# ------------------------------------------------------------
# 1) Define regions and their colors (for plotting)
# ------------------------------------------------------------
regions = ["North America", "Asia", "Europe", "Africa", "South America", "Oceania"]

region_colors = {
    "North America": "#6b6b9f",
    "Asia":          "#2b6cb0",
    "Europe":        "#e55353",
    "Africa":        "#333333",
    "South America": "#f6ad55",
    "Oceania":       "#38a169",
}

# ------------------------------------------------------------
# 2) Build a modular network using a Stochastic Block Model (SBM)
# ------------------------------------------------------------
# sizes = number of nodes in each region (total = 250)
sizes = [60, 35, 70, 25, 30, 30]
k = len(sizes)  # number of regions

# Probability of edges:
p_in = 0.10    # chance of edge INSIDE the same region
p_out = 0.008  # chance of edge BETWEEN different regions

# Create a k x k matrix of edge probabilities
# Start with all p_out, then replace diagonal with p_in.
P = np.full((k, k), p_out)
np.fill_diagonal(P, p_in)

# Build the graph
G = nx.stochastic_block_model(sizes, P, seed=1)

# ------------------------------------------------------------
# 3) Attach a "region" label to each node (so we can color nodes)
# ------------------------------------------------------------
node_regions = {}
node_id = 0

for region_name, region_size in zip(regions, sizes):
    for _ in range(region_size):
        node_regions[node_id] = region_name
        node_id += 1

# Save node attribute "region"
nx.set_node_attributes(G, node_regions, "region")

# ------------------------------------------------------------
# 4) Helper function: what fraction of edges are cross-region?
# ------------------------------------------------------------
def inter_edge_share(graph):
    """
    Return the proportion of edges that connect two DIFFERENT regions.
    """
    inter_edges = 0

    for u, v in graph.edges():
        if graph.nodes[u]["region"] != graph.nodes[v]["region"]:
            inter_edges += 1

    total_edges = graph.number_of_edges()
    if total_edges == 0:
        return 0.0

    return inter_edges / total_edges


# ------------------------------------------------------------
# 5) Create two scenarios: (a) baseline, (b) fragmentation
# ------------------------------------------------------------
# (a) Baseline: just copy the original graph
G_a = G.copy()

# (b) Fragmentation: remove many cross-region edges
G_b = G.copy()

# Find all edges that connect different regions
cross_edges = []
for u, v in G_b.edges():
    if G_b.nodes[u]["region"] != G_b.nodes[v]["region"]:
        cross_edges.append((u, v))

# Shuffle cross edges and remove a large fraction of them
np.random.shuffle(cross_edges)

remove_frac = 0.80  # remove 80% of cross-region edges
num_to_remove = int(remove_frac * len(cross_edges))

edges_to_remove = cross_edges[:num_to_remove]
G_b.remove_edges_from(edges_to_remove)


# ------------------------------------------------------------
# 6) Choose a layout (node positions) once, reuse for both plots
# ------------------------------------------------------------
# Using positions from baseline so both panels look comparable.
pos = nx.spring_layout(G_a, seed=2, k=0.12, iterations=200)


# ------------------------------------------------------------
# 7) Helper function to draw a graph on an axis
# ------------------------------------------------------------
def draw_graph(ax, graph, title):
    """
    Draw the given graph on the matplotlib axis ax.
    """
    # Build list of node colors based on region
    node_colors = [region_colors[graph.nodes[n]["region"]] for n in graph.nodes()]

    # Draw edges first (so nodes appear on top)
    nx.draw_networkx_edges(
        graph, pos, ax=ax,
        width=0.35, alpha=0.35, edge_color="black"
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        graph, pos, ax=ax,
        node_color=node_colors, node_size=18, linewidths=0
    )

    # Title and formatting
    ax.set_title(title, loc="left")
    ax.set_axis_off()


# ------------------------------------------------------------
# 8) Plot side-by-side
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

draw_graph(axes[0], G_a, "(a) Baseline (higher cross-region connectivity)")
draw_graph(axes[1], G_b, "(b) Fragmentation (inter-region links reduced)")

# Build a legend (one marker per region)
legend_handles = []
for r in regions:
    handle = Line2D(
        [0], [0],
        marker="o", color="w",
        markerfacecolor=region_colors[r],
        markersize=6,
        label=r
    )
    legend_handles.append(handle)

fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=3,
    frameon=False,
    fontsize=9
)

plt.tight_layout()
plt.subplots_adjust(bottom=0.18)

# Save and show
plt.savefig("region_network_extension.png", dpi=300, bbox_inches="tight")
plt.show()


# ------------------------------------------------------------
# 9) Print metrics (useful in write-up)
# ------------------------------------------------------------
print("Edges (a):", G_a.number_of_edges(),
      " inter-edge share:", round(inter_edge_share(G_a), 3))

print("Edges (b):", G_b.number_of_edges(),
      " inter-edge share:", round(inter_edge_share(G_b), 3))

print("Largest CC size (a):", len(max(nx.connected_components(G_a), key=len)))
print("Largest CC size (b):", len(max(nx.connected_components(G_b), key=len)))

