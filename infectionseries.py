#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date

# -------------------------------------------------
# Example infection series (replace with real one)
# -------------------------------------------------
idx = pd.date_range("2015-01-01", "2025-01-01", freq="B")
infection_count = pd.Series(
    np.clip(40 + 10*np.sin(np.linspace(0, 20, len(idx))) +
            15*np.random.randn(len(idx)), 0, None),
    index=idx, name="infected"
).round()

# -------------------------------------------------
# Crisis windows + events
# -------------------------------------------------
period_pre = slice("2015-06-01", "2017-01-01")
period_covid = slice("2020-02-01", "2023-01-01")

events_pre = {
    "China devaluation":          date(2015, 8, 24),
    "2016 risk-off":              date(2016, 2, 11),
    "Brexit referendum":          date(2016, 6, 23),
}

events_covid = {
    "COVID-19 crash":             date(2020, 3, 16),
    "Fed emergency cuts":         date(2020, 3, 15),
    "Vaccine news":               date(2020, 11, 9),
    "Russia–Ukraine invasion":    date(2022, 2, 24),
    "Rate hikes begin":           date(2022, 3, 16),
}

series_pre   = infection_count.loc[period_pre]
series_covid = infection_count.loc[period_covid]

# -------------------------------------------------
# Helper to plot one crisis panel
# -------------------------------------------------
def plot_crisis(ax, series, events, title, ymax=None):
    ax.plot(series.index, series.values,
            lw=2.0, color="black", label="infected institutions")

    # vertical event lines
    for d in events.values():
        ax.axvline(pd.to_datetime(d), ls="--", lw=1.1, color="C0")

    # legend (proxy handles)
    handles = [plt.Line2D([0], [0], color="black", lw=2,
                           label="infected institutions")]
    handles += [plt.Line2D([0], [0], color="C0", ls="--", lw=1.1,
                           label=lab) for lab in events.keys()]

    ax.legend(handles=handles, loc="upper right", fontsize=7)

    ax.set_title(title, fontsize=12, loc="left", fontstyle="italic")
    ax.set_ylabel("Infected financial institutions")
    ax.set_xlabel("Date")

    # Spacing & formatting improvements
    ax.grid(alpha=0.25)
    ax.set_ylim(0, ymax)

    # Spread out ticks
    ax.xaxis.set_major_locator(plt.MaxNLocator(6))

    # Rotate ticks for readability
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_horizontalalignment("right")

    # Add date padding
    ax.set_xlim(series.index.min() - pd.Timedelta(days=20),
                series.index.max() + pd.Timedelta(days=20))

# -------------------------------------------------
# Build two-panel figure with more horizontal space
# -------------------------------------------------
fig, axes = plt.subplots(
    1, 2,
    figsize=(15, 5),        # wider figure
    sharey=True,
    gridspec_kw={"wspace": 0.15}   # more horizontal spacing
)

ymax = max(series_pre.max(), series_covid.max()) * 1.15

plot_crisis(axes[0], series_pre,   events_pre,
            "(a) Pre-COVID stress episodes (2015–2016)", ymax=ymax)

plot_crisis(axes[1], series_covid, events_covid,
            "(b) COVID-19 and inflationary shock (2020–2022)", ymax=ymax)

fig.suptitle("Financial-sector contagion during major crises",
             fontsize=15, fontweight="bold", y=1.03)

plt.tight_layout()
plt.savefig("fig_financial_crises_contagion.png", dpi=300, bbox_inches="tight")
plt.show()
