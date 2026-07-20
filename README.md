# MNABC — Modified Novel Artificial Bee Colony

Implementation and experimental evaluation of **MNABC**, a variant of the *Artificial Bee Colony* (ABC) algorithm featuring:
- a hybrid utility function (**HUFS** — *High-Utility Food Sources*) combining solution quality and spatial diversity;
- a memory-driven restart mechanism (*Memory-Driven Restart*) for the Scout phase;
- an anchor-shift operation on a randomly selected dimension.

The algorithm is evaluated on the **CEC2013** and **CEC2022** benchmark suites (official functions via `opfunu`).

---

## 📁 Repository structure

```
MNABC/
├── results/
│   ├── Results_cec2013_D30.csv     # CEC2013 results, dimension 30
│   ├── Results_cec2013_D50.csv     # CEC2013 results, dimension 50
│   ├── Results_cec2022_D10.csv     # CEC2022 results, dimension 10
│   └── Results_cec2022_D20.csv     # CEC2022 results, dimension 20
└── src/
    ├── MNABC                       # Main algorithm implementation
    ├── MNABC_CEC2022.ipynb         # Experiment notebook — CEC2022 benchmark
    └── MNABC_MNABC_Neigh_on_CE...  # Neighborhood-based variant
```

---

## ⚙️ Environment setup

The project has been validated with **Python 3.10** (or 3.12) via `conda`.

```bash
# 1) Create a clean conda environment
conda create -n mnabc_env python=3.10 -y
conda activate mnabc_env
```

Alternative: create the environment in a specific directory instead of conda's default `envs` folder:

```bash
# Create the environment in a given directory
conda create -p C:\mnabc_env python=3.12 -y

# Activate it via its full path
conda activate "C:\mnabc_env"
```

Then install the required dependencies:

```bash
pip install numpy scipy matplotlib pandas tabulate notebook
pip install "setuptools<70.0.0" opfunu
```

> ℹ️ `opfunu` requires `setuptools < 70.0.0` to install correctly with current `pip` versions.

---

## ▶️ Usage

1. Activate the environment:
   ```bash
   conda activate mnabc_env
   ```
2. Launch Jupyter:
   ```bash
   jupyter notebook
   ```
3. Open `src/MNABC_CEC2022.ipynb` or 'MNABC_MNABC_Neigh_on_CEC2013_D30_or_D50.ipynb'.
4. Adjust parameters at the top of the notebook if needed (`DIM`, `N_RUNS`, `N_BEES`, `alpha`, `HUFS_ratio`, etc.).
5. Run the cells.

Results (CSV files, convergence plots, LaTeX table) are automatically generated in a `results_MNABC_CEC2022_D{DIM}_.../` folder.

---

## 📊 Results

| File | Benchmark | Dimension |
|---|---|---|
| `Results_cec2013_D30.csv` | CEC2013 | 30 |
| `Results_cec2013_D50.csv` | CEC2013 | 50 |
| `Results_cec2022_D10.csv` | CEC2022 | 10 |
| `Results_cec2022_D20.csv` | CEC2022 | 20 |

Each file contains, per function, aggregated statistics across all runs (mean, standard deviation, best, worst, median).

---

## 📦 Main dependencies

| Library | Role |
|---|---|
| `numpy`, `scipy` | Scientific computing / statistical tests (Wilcoxon, Friedman) |
| `pandas` | Tabular results handling |
| `matplotlib` | Convergence plots |
| `tabulate` | Table formatting |
| `opfunu` | Official CEC2013/CEC2022 benchmark functions |
| `notebook` | Running Jupyter notebooks |

---

## 📄 License

Apache 2.0

## 👤 Author

sgherbi2025
