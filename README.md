# MSc-Dissertation
# Network-Based SIS Contagion Models for Financial and Macroeconomic Stress

This repository contains the full source code used in the MSc dissertation:

**“How effectively can network-based SIS contagion models explain the propagation and persistence of stress in financial systems and conflict-affected regional macroeconomic networks?”**

*MSc Financial Mathematics, University of Nottingham (2025)*  
Author: **Dusha Thangarajah**

---

## Overview

This project applies **network epidemiology**, specifically the **Susceptible–Infected–Susceptible (SIS)** contagion model, to study stress propagation in:

- Interconnected **financial networks** constructed from asset return correlations, and
- **Conflict-affected regional macroeconomic networks**, using a stylised case study of Sri Lanka during the civil conflict period (1983–2009).

The code supports both:
- **Stochastic network simulations**, and
- **Mean-field approximations and parameter estimation**,

allowing analysis of stress transmission, persistence, recovery dynamics, and forecasting performance.

---

## Repository Structure

```text
.
├── financial/
│   ├── download_prices.py        # Data acquisition (financial time series)
│   ├── volatility_states.py      # Stress / infection state construction
│   ├── network_construction.py   # Correlation-based network building
│   ├── sis_estimation.py         # Rolling SIS parameter estimation
│   └── forecasting.py            # One-step-ahead SIS forecasts
│
├── macro/
│   ├── synthetic_macro_data.py   # Stylised Sri Lanka & Tamil region data
│   ├── stress_index.py           # Composite stress index construction
│   ├── macro_network.py          # Illustrative macroeconomic network
│   ├── sis_simulation.py         # Discrete-time SIS simulation
│   └── grid_search.py            # Monte Carlo SIS parameter estimation
│
├── simulations/
│   ├── gillespie_sis.py           # Continuous-time SIS (Gillespie)
│   ├── mean_field_ode.py          # Mean-field SIS ODE
│   └── scaling_validation.py     # Network scaling validation
│
├── figures/
│   └── *.png                      # All figures used in the dissertation
│
├── data/
│   ├── infection_panel.csv        # Binary infection states
│   ├── rolling_sis_params.csv     # Estimated parameters
│   └── macro_states_panel.csv
│
└── README.md
