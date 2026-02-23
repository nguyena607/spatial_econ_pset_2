# Spatial Economics Pset 2

Empirical analysis of the labor market effects of immigration enforcement using a Difference-in-Differences design and a dynamic spatial equilibrium model (CDP-Lite).

## Overview

This project estimates how the 2017 immigration enforcement shock affected wages and employment across U.S. states, distinguishing between states with high vs. low exposure to undocumented immigrants. The analysis proceeds in two parts:

1. **Reduced-Form DiD** — Two-way fixed effects regression with an event study to verify parallel trends
2. **CDP-Lite Model** — A dynamic spatial model with forward-looking workers that decomposes the enforcement effect into wage and fear/disutility channels, and tests for anticipation effects

## Repository Structure

```
├── cdf_model.ipynb        # Main analysis notebook
├── datasets/              # Input data
│   ├── state_panel.csv         # State-year panel (wages, employment, exposure)
│   ├── state_migration_annual.csv  # Annual state-to-state migration flows
│   ├── mu_minus1.csv           # Baseline migration transition matrix
│   ├── state_qcew_annual.csv   # QCEW employment data
│   └── treated_states.csv      # High-exposure state definitions
├── figures/               # Output figures
└── tables/                # Output LaTeX tables
```

## Main Results

- **Wage effect**: ~1.4% increase in high-exposure states (not statistically significant)
- **Employment effect**: ~3.2% increase in high-exposure states (significant at 5%)
- **Channel decomposition**: The fear/disutility channel dominates the wage channel in driving out-migration from high-exposure states
- **Anticipation**: Workers who anticipate the shock from 2015 show earlier pre-migration relative to the immediate-shock counterfactual
