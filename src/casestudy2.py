

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1) SETTINGS (things you might change)
# ------------------------------------------------------------
OUTPUT_FILE = "sri_synth_gdp_ceasefire.png"
CEASEFIRE_YEAR = 2002

# Years we want on the x-axis
years = np.arange(1983, 2010)  # 1983, 1984, ..., 2009

# ------------------------------------------------------------
# 2) DATA (replace this block if you already have real data)
# ------------------------------------------------------------
# If you already have your own arrays (gdp_sl and gdp_te),
# you can delete this synthetic-data section and paste yours.

np.random.seed(0)  # makes the random numbers the same every run (for reproducibility)

# Example Sri Lanka GDP growth:
# base + a wavy pattern + random noise
gdp_sl = 3.5 + 1.0 * np.sin(0.35 * (years - 1983)) + np.random.normal(0, 0.6, len(years))

# Create a "synthetic" Tamil region series from Sri Lanka series:
delta = 0.25  # how much smaller (on average) Tamil growth is compared to SL
noise = np.random.normal(0, 0.5, len(years))  # extra random noise
gdp_te = (1 - delta) * gdp_sl + noise

# Optional: keep GDP growth non-negative (remove if you want negative values allowed)
gdp_te = np.clip(gdp_te, 0, None)

# ------------------------------------------------------------
# 3) COMPUTE PRE- AND POST-CEASEFIRE MEANS
# ------------------------------------------------------------
# Make True/False masks (filters) for years before and after 2002
pre_mask = years < CEASEFIRE_YEAR
post_mask = years >= CEASEFIRE_YEAR

# Compute averages for each period and each series
sl_pre_mean = gdp_sl[pre_mask].mean()
sl_post_mean = gdp_sl[post_mask].mean()

te_pre_mean = gdp_te[pre_mask].mean()
te_post_mean = gdp_te[post_mask].mean()

# ------------------------------------------------------------
# 4) PLOT THE DATA
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10.5, 3.7), dpi=150)

# Plot Sri Lanka series (black line with circles)
ax.plot(
    years,
    gdp_sl,
    color="black",
    marker="o",
    linewidth=2,
    markersize=5,
    label="Sri Lanka (annual GDP growth)"
)

# Plot Tamil region series (red line with squares)
ax.plot(
    years,
    gdp_te,
    color="red",
    marker="s",
    linewidth=2,
    markersize=5,
    label="Tamil region (synthetic)"
)

# ------------------------------------------------------------
# 5) ADD CEASEFIRE LINE + LABEL
# ------------------------------------------------------------
ax.axvline(CEASEFIRE_YEAR, color="black", linestyle="--", linewidth=1.5)

# Put a vertical label near the top of the plot
top_y = ax.get_ylim()[1]  # current top of y-axis
ax.text(
    CEASEFIRE_YEAR + 0.15,
    top_y * 0.95,
    "Ceasefire 2002",
    rotation=90,
    va="top",
    ha="left"
)

# ------------------------------------------------------------
# 6) DRAW PRE/POST AVERAGE LINES
# ------------------------------------------------------------
# Sri Lanka averages
ax.hlines(sl_pre_mean, years[0], CEASEFIRE_YEAR, linestyles=":", linewidth=1.5,
          label="SL pre-2002 mean")
ax.hlines(sl_post_mean, CEASEFIRE_YEAR, years[-1], linestyles="--", linewidth=1.5,
          label="SL post-2002 mean")

# Tamil region averages
ax.hlines(te_pre_mean, years[0], CEASEFIRE_YEAR, linestyles=":", linewidth=1.5,
          label="TE pre-2002 mean")
ax.hlines(te_post_mean, CEASEFIRE_YEAR, years[-1], linestyles="-.", linewidth=1.5,
          label="TE post-2002 mean")

# ------------------------------------------------------------
# 7) TITLES, LABELS, GRID, LEGEND
# ------------------------------------------------------------
ax.set_title("Structural Change in GDP Growth: Pre-/Post-Ceasefire", fontsize=12)
ax.set_xlabel("Year")
ax.set_ylabel("Real GDP growth (%)")

ax.grid(True, alpha=0.3)
ax.legend(loc="lower left", fontsize=8, ncol=2, frameon=True)

# Make layout fit nicely (avoid cut-off labels)
fig.tight_layout()

# ------------------------------------------------------------
# 8) SAVE + SHOW
# ------------------------------------------------------------
fig.savefig(OUTPUT_FILE, bbox_inches="tight")
plt.show()

print(f"Saved: {OUTPUT_FILE}")
