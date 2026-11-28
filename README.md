# Kernel-Based Nonparametric Tests for Shape Constraints

The package reproduces the article's empirical applications (equity returns and Treasury bills), its sensitivity analysis and its Monte Carlo study. The Monte Carlo study requires no data.

## Requirements

- **MATLAB:** R2026b. No toolboxes are required.
- **Font (optional):** the figures use Times New Roman when it is installed. Otherwise MATLAB substitutes a default font; the numbers are unaffected.
- **Data:** CRSP and JKP data through a WRDS subscription, for the empirical applications only (below).

## Data

The data are licensed and are not included. Download them under your own WRDS subscription ([wrds-www.wharton.upenn.edu](https://wrds-www.wharton.upenn.edu)) and save them as CSV files, as follows. `scripts/build_data.m` creates the folders if they do not exist.

| File | Source | Used by |
|---|---|---|
| `data/raw/jkp/jkp_global_factor_usa_YYYY.csv`, one file per year 1964–2025 | JKP Global Factor Data (Jensen, Kelly and Pedersen, 2023), WRDS table `contrib.global_factor`, US stocks (`excntry = 'USA'`). Access needs CRSP and Compustat; see the [JKP/WRDS guide](https://jkpfactors.com/jkp-wrds-guide). | Equity applications |
| `data/raw/crsp/crsp_treasury_monthly_1964_2025.csv` | CRSP US Treasury Database ([crsp.org](https://www.crsp.org)) via WRDS: monthly series of individual issues | Treasury-bill applications |
| `data/raw/crsp/crsp_riskfree_monthly_1964_2025.csv` | CRSP US Treasury Database via WRDS: monthly risk-free series | Treasury-bill applications |

The code reads the following columns:

- **JKP files:** `id`, `eom`, `permno`, `me`, `ret_exc_lead1m`, `beta_60m`, `be_me`, `ocf_me`, `ni_me`, `div12m_me`, `ret_12_1`, `ret_1_0`, `ret_60_12`.
- **Treasury file:** `KYTREASNO`, `TMATDT`, `ITYPE`, `MCALDT`, `TMRETNUA`, `TMTOTOUT`. Bills are the rows with `ITYPE` = 4.
- **Risk-free file:** `TTERMTYPE`, `MCALDT`, `TMYTM`. The one-month rate is the series with `TTERMTYPE` = 30.

Exact replication of the empirical results requires the data vintage used in the article (CRSP and JKP files downloaded from WRDS in 2026); later revisions of these data may change the results. The beta-decile panel used for the market-beta application is built only from JKP files that contain the `id` column.

## Repository structure

| Path | Content |
|---|---|
| `src/` | Functions: `kernels`, `estimation` (kernel ridge fit and penalty selection), `inference` (scores, Newey–West covariance, cone-Wald tests), `panels` (data construction), `monte_carlo` |
| `scripts/` | The scripts listed under "Replication" below |
| `reference/` | The article's figures (`figures/`), shape-test tables (`tables/`) and Monte Carlo summaries (`monte_carlo/`), for comparison |

The scripts create `data/` (inputs), `results/` (estimates), `figures/` and `tables/`, which are not tracked. Every script finds the repository from its own location, so no paths need to be set.

## Replication

Run the scripts from the MATLAB Command Window in the repository root. Scripts read optional settings from the workspace, so start each call with `clear`. Run times are for a single MATLAB process on a recent laptop.

1. **Prepare the data** (several minutes):
   ```matlab
   clear; run("scripts/build_data.m")
   ```
   This builds `data/processed/` and `results/e2_distribution.mat`.
2. **Estimate the empirical applications** (about 30 minutes):
   ```matlab
   clear; run("scripts/run_empirical.m")
   ```
   Results are saved in `results/`, and completed files are reused.
3. **Run the sensitivity analysis** (Table F.1, about 2 minutes):
   ```matlab
   clear; run("scripts/run_sensitivity.m")
   ```
   The table is printed in the Command Window.
4. **Run the Monte Carlo study** (about 45 CPU-minutes). This step needs no data; without the empirical results, `make_figures.m` in step 5 then draws Figures G.1 and G.2 only.
   - **One process:**
     ```matlab
     clear; run("scripts/run_monte_carlo.m")
     ```
   - **K processes in parallel:** first compute the population labels once:
     ```matlab
     clear; n_rep = 0; run("scripts/run_monte_carlo.m")
     ```
     Then start K processes from a terminal in the repository root, with k = 1, …, K:
     ```bash
     matlab -batch "worker = [k K]; run('scripts/run_monte_carlo.m')"
     ```
   - **Combine and summarize** (Table G.1 and the data for Figures G.1 and G.2):
     ```matlab
     clear; run("scripts/summarize_monte_carlo.m")
     ```
5. **Produce the figures and tables** (seconds; saved results only):
   ```matlab
   clear; run("scripts/make_figures.m")
   clear; run("scripts/make_tables.m")
   ```

| Exhibit | Script | Output |
|---|---|---|
| Figures 5.1–5.4, G.1–G.3 | `make_figures.m` | `figures/e1_curve.pdf`, `e2_profiles.pdf`, `e3_profiles.pdf`, `e4_profiles_surfaces.pdf`, `mc_pointwise_errors.pdf`, `mc_rejection_frequencies.pdf`, `e2_histograms.pdf` |
| Tables 5.1–5.4 | `make_tables.m` | `tables/e1_shape_tests` to `tables/e4_shape_tests`, in `.tex` and `.csv` |
| Table F.1 | `run_sensitivity.m` | Printed in the Command Window |
| Table G.1 | `summarize_monte_carlo.m` | `results/mc_estimation_errors.csv` |
| Monte Carlo population labels (Appendix G) | `run_monte_carlo.m` | `results/mc_population_labels.csv` |

The outputs can be compared with the files in `reference/`, which have the same names.

## Citation

Please cite the article and this package; citation metadata are in `CITATION.cff`.

## License

The code is released under the MIT License (`LICENSE`). The license does not cover the CRSP and JKP data, which remain subject to their providers' terms.
