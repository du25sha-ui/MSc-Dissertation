
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date


# ------------------------------------------------------------
# 1) EXAMPLE INFECTION SERIES (replace with your real one)
# ------------------------------------------------------------
# idx = business-day dates from 2015-01-01 to 2025-01-01
idx = pd.date_range("2015-01-01", "2025-01-01", freq="B")

# Fake infection count series:
# baseline + wave + random noise, then clipped so it never goes below 0
infection_count = pd.Series(
    np.clip(
        40
        + 10 * np.sin(np.linspace(0, 20, len(idx)))
        + 15 * np.random.randn(len(idx)),
        0,
        None
    ),
    index=idx,
    name="infected"
).round()


# ------------------------------------------------------------
# 2) CHOOSE CRISIS WINDOWS AND KEY EVENTS
# ------------------------------------------------------------
# Two time windows (as slices)
period_pre = slice("2015-06-01", "2017-01-01")
period_covid = slice("2020-02-01", "2023-01-01")

# Events: label -> exact date
events_pre = {
    "China devaluation":   date(2015, 8, 24),
    "2016 risk-off":       date(2016, 2, 11),
    "Brexit referendum":   date(2016, 6, 23),
}

events_covid = {
    "COVID-19 crash":          date(2020, 3, 16),
    "Fed emergency cuts":      date(2020, 3, 15),
    "Vaccine news":            date(2020, 11, 9),
    "Russia–Ukraine invasion": date(2022, 2, 24),
    "Rate hikes begin":        date(2022, 3, 16),
}

# Cut the big series into the two smaller crisis series
series_pre = infection_count.loc[period_pre]
series_covid = infection_count.loc[period_covid]


# ------------------------------------------------------------
# 3) HELPER FUNCTION: PLOT ONE PANEL
# ------------------------------------------------------------
def plot_crisis_panel(ax, series, events, title, ymax):
    """
    Plot one crisis window:
    - the infection series (black line)
    - vertical dashed lines for events (blue)
    - a legend listing the events
    """

    # Plot infection count
    ax.plot(
        series.index,
        series.values,
        lw=2.0,
        color="black",
        label="infected institutions"
    )

    # Add vertical event lines
    for event_date in events.values():
        ax.axvline(pd.to_datetime(event_date), ls="--", lw=1.1, color="C0")

    # Create a custom legend (so every event appears in the legend)
    legend_handles = []

    # One handle for the infection line
    legend_handles.append(
        plt.Line2D([0], [0], color="black", lw=2, label="infected institutions")
    )

    # One handle per event (same style as the event lines)
    for label in events.keys():
        legend_handles.append(
            plt.Line2D([0], [0], color="C0", ls="--", lw=1.1, label=label)
        )

    ax.legend(handles=legend_handles, loc="upper right", fontsize=7)

    # Titles and labels
    ax.set_title(title, fontsize=12, loc="left", fontstyle="italic")
    ax.set_ylabel("Infected financial institutions")
    ax.set_xlabel("Date")

    # Formatting: grid, limits, ticks
    ax.grid(alpha=0.25)
    ax.set_ylim(0, ymax)

    # Fewer x-ticks so the axis isn't cluttered
    ax.xaxis.set_major_locator(plt.MaxNLocator(6))

    # Rotate x tick labels
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)
        tick.set_horizontalalignment("right")

    # Add padding so the line isn't touching the edges
    ax.set_xlim(
        series.index.min() - pd.Timedelta(days=20),
        series.index.max() + pd.Timedelta(days=20)
    )


# ------------------------------------------------------------
# 4) BUILD THE TWO-PANEL FIGURE
# ------------------------------------------------------------
fig, axes = plt.subplots(
    1, 2,
    figsize=(15, 5),        # wider figure = more space for dates + legend
    sharey=True,
    gridspec_kw={"wspace": 0.15}  # space between panels
)

# Use one shared y-axis max so the panels are comparable
ymax = max(series_pre.max(), series_covid.max()) * 1.15

# Plot the two panels
plot_crisis_panel(
    axes[0], series_pre, events_pre,
    "(a) Pre-COVID stress episodes (2015–2016)",
    ymax=ymax
)

plot_crisis_panel(
    axes[1], series_covid, events_covid,
    "(b) COVID-19 and inflationary shock (2020–2022)",
    ymax=ymax
)

# Add an overall title (suptitle)
fig.suptitle(
    "Financial-sector contagion during major crises",
    fontsize=15,
    fontweight="bold",
    y=1.03
)

# Make layout tight and save
plt.tight_layout()
plt.savefig("fig_financial_crises_contagion.png", dpi=300, bbox_inches="tight")
plt.show()
