#!/usr/bin/env python
# coding: utf-8

# # ASABC — Adaptive Exploration-Exploitation Switching ABC
# **Reference:** Zhang et al., *Expert Systems with Applications* 318 (2026) 131979  
# **Benchmarks:** CEC 2017 (D = 10, 30, 50, 100) · CEC 2022 (D = 10, 20)  
# **Protocol:** 51 independent runs (CEC2017) / 30 runs (CEC2022) · MaxFEs = 10 000·D (CEC2017) / 200 000·D=10, 1 000 000·D=20 (CEC2022)  
# **Output:** mean error, std, Friedman rank, Wilcoxon vs AutoABC & EABC-AS · convergence curves · LaTeX tables  
# 
# > **Instructions**  
# > 1. Install `opfunu` for official CEC functions: `pip install opfunu`  
# > 2. Set `DIM` and `SUITE` at the top of §3 then *Run All*.  
# > 3. Results auto-saved every 30 min to `BASE_DIR`.  
# > 4. Run §7 to compile all saved CSVs and produce the publication-ready tables.
# 

# ## 1 · Dependencies

# ── install once ──────────────────────────────────────────────────────────────
import subprocess, sys
def pip(*pkgs):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *pkgs])
pip('setuptools<70.0.0', 'opfunu')
pip('scipy', 'pandas', 'matplotlib', 'tqdm', 'tabulate')
pip('ipywidgets')

# ## 2 · Imports

import os, time, json, math, copy, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, friedmanchisquare, rankdata
from tqdm import tqdm
import opfunu

warnings.filterwarnings('ignore')
rng = np.random.default_rng()
print('opfunu version:', opfunu.__version__)

# ## 3 · Configuration — **edit here**

# ── USER SETTINGS ─────────────────────────────────────────────────────────────
SUITE         = 'CEC2022'   # 'CEC2017' or 'CEC2022'
DIM           = 20          # 10 | 30 | 50 | 100  (CEC2017)  |  10 | 20  (CEC2022)
# ──────────────────────────────────────────────────────────────────────────────

# ── Derived protocol ──────────────────────────────────────────────────────────
if SUITE == 'CEC2017':
    N_BEES        = 50
    N_RUNS        = 51
    MAX_FES       = 10_000 * DIM
    N_FUNCS       = 29          # f1..f29 (f2 excluded by CEC2017 convention)
    FUNC_IDS      = list(range(1, 30))   # opfunu uses 1-based index
    BASE_DIR      = rf'C:\Asabc_CEC2017\Asabc_D{DIM}'
else:  # CEC2022
    N_BEES        = 30
    N_RUNS        = 30
    MAX_FES       = 200_000 if DIM == 10 else 1_000_000
    N_FUNCS       = 12
    FUNC_IDS      = list(range(1, 13))
    BASE_DIR      = rf'C:\Asabc_CEC2022\Asabc_D{DIM}'

MAX_ITER      = MAX_FES // N_BEES
BOUNDS        = [(-100.0, 100.0)] * DIM
SAVE_INTERVAL = 1800   # seconds — checkpoint every 30 min

os.makedirs(BASE_DIR, exist_ok=True)
print(f'Suite={SUITE}  D={DIM}  N_BEES={N_BEES}  MAX_FES={MAX_FES:,}  N_RUNS={N_RUNS}')
print(f'Results → {BASE_DIR}')

# ## 4 · CEC Benchmark Loader

from opfunu.cec_based import cec2017, cec2022

def get_func(suite: str, fid: int, dim: int):
    """Return a callable  f(x) → float  and the known optimum value."""
    if suite == 'CEC2017':
        cls_name = f'F{fid}2017'
        cls = getattr(cec2017, cls_name)
        obj = cls(ndim=dim)
    else:
        cls_name = f'F{fid}2022'
        cls = getattr(cec2022, cls_name)
        obj = cls(ndim=dim)
    return obj

# Quick smoke-test
_t = get_func(SUITE, FUNC_IDS[0], DIM)
x0 = np.zeros(DIM)
print(f'f1 smoke test  f(0)={_t.evaluate(x0):.4e}  optimum={_t.f_global:.4e}')

# ## 5 · ASABC Implementation
# 
# Faithful re-implementation of the algorithm described in Zhang et al. (2026),  
# including:
# - **ILM** (Information Landscape Measure) — computed every 50 iterations
# - **Dynamic ring neighbourhood** with radius growing from 1 to SN/4
# - **Exploitation-biased** strategy set (smooth landscape, PH < ph_bdy)
# - **Exploration-biased** strategy set (rugged landscape, PH ≥ ph_bdy)
# - Scout bee phase with experience-preserving reinitialization
# 
# Default hyper-parameters taken from §4.6 sensitivity analysis:  
# `ph_bdy=0.3`, `CR_exl=0.1`, `CR_exp=0.7`, `limit=50`

# ─────────────────────────────────────────────────────────────────────────────
# ASABC — full implementation
# ─────────────────────────────────────────────────────────────────────────────

def _fit(f_val: float) -> float:
    """ABC fitness transform (Eq. 3)."""
    if f_val >= 0:
        return 1.0 / (1.0 + f_val)
    else:
        return 1.0 + abs(f_val)


def _sphere(x: np.ndarray) -> float:
    return float(np.sum(x ** 2))


def _ilm_ph(pop: np.ndarray, f_vals: np.ndarray) -> float:
    """
    Online Information Landscape Measure (ILM).
    Computes the problem hardness PH ∈ [0,1] using the current population
    as sample, with Sphere as reference (Borenstein & Poli, 2005).
    """
    n = len(pop)
    # Reference function (Sphere)
    f_ref = np.array([_sphere(x - pop.mean(axis=0)) for x in pop])

    # --- Build upper-triangular comparison vectors ---
    # Exclude row/col of global optimum proxy (best individual)
    best_idx = int(np.argmax([_fit(v) for v in f_vals]))

    vt, vr = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if i == best_idx or j == best_idx:
                continue
            # target
            if f_vals[i] < f_vals[j]:
                vt.append(1.0)
            elif f_vals[i] == f_vals[j]:
                vt.append(0.5)
            else:
                vt.append(0.0)
            # reference
            if f_ref[i] < f_ref[j]:
                vr.append(1.0)
            elif f_ref[i] == f_ref[j]:
                vr.append(0.5)
            else:
                vr.append(0.0)

    if len(vt) == 0:
        return 0.0
    vt = np.array(vt)
    vr = np.array(vr)
    return float(np.mean(np.abs(vt - vr)))


class ASABC:
    """
    Adaptive Exploration-Exploitation Switching ABC.
    Zhang et al., Expert Systems With Applications 318 (2026) 131979.
    """

    def __init__(
        self,
        func,
        dim: int,
        bounds,
        sn: int = 50,
        max_fes: int = 100_000,
        limit: int = 50,
        ph_bdy: float = 0.3,
        cr_exl: float = 0.1,
        cr_exp: float = 0.7,
        ilm_interval: int = 50,
        seed=None,
    ):
        self.func        = func
        self.dim         = dim
        self.lb          = np.array([b[0] for b in bounds])
        self.ub          = np.array([b[1] for b in bounds])
        self.sn          = sn
        self.max_fes     = max_fes
        self.limit       = limit
        self.ph_bdy      = ph_bdy
        self.cr_exl      = cr_exl
        self.cr_exp      = cr_exp
        self.ilm_interval= ilm_interval
        self.rng         = np.random.default_rng(seed)

    # ── helpers ──────────────────────────────────────────────────────────────
    def _clip(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self.lb, self.ub)

    def _ring_k(self, fes: int) -> int:
        """Dynamic neighbourhood radius (Eq. 12)."""
        k = int((self.sn / 4 - 1) * fes / self.max_fes + 1)
        return max(1, min(k, self.sn // 2 - 1))

    def _lbest(self, i: int, k: int) -> int:
        """Index of best individual in ring neighbourhood of i."""
        indices = [(i + d) % self.sn for d in range(-k, k + 1)]
        best = min(indices, key=lambda idx: self.f_vals[idx])
        return best

    def _rand_pair(self, exclude: set) -> tuple:
        candidates = [j for j in range(self.sn) if j not in exclude]
        chosen = self.rng.choice(candidates, size=2, replace=False)
        return int(chosen[0]), int(chosen[1])

    # ── search strategies ────────────────────────────────────────────────────
    # --- Exploitation-biased (smooth landscape) ---

    def _employed_exl(self, i: int, k: int) -> np.ndarray:
        """Eq. 11: v = x_lbest + phi*(x_r1 - x_r2)"""
        lb_idx = self._lbest(i, k)
        r1, r2 = self._rand_pair({i})
        phi = self.rng.uniform(-1, 1, self.dim)
        v = self.pop[lb_idx] + phi * (self.pop[r1] - self.pop[r2])
        return self._clip(v)

    def _onlooker_exl(self, i: int, k: int) -> np.ndarray:
        """Eq. 14: v = x_lbest + phi*(x_i - x_r1) + phi*(x_best - x_r2)"""
        lb_idx = self._lbest(i, k)
        r1, r2 = self._rand_pair({i, lb_idx})
        phi1 = self.rng.uniform(-1, 1, self.dim)
        phi2 = self.rng.uniform(-1, 1, self.dim)
        v = (self.pop[lb_idx]
             + phi1 * (self.pop[i] - self.pop[r1])
             + phi2 * (self.pop[self.best_idx] - self.pop[r2]))
        return self._clip(v)

    def _scout_exl(self, i: int) -> np.ndarray:
        """Eq. 15: normal distribution around x_best with σ=1, dims chosen by CR_exl"""
        v = self.pop[i].copy()
        mask = self.rng.random(self.dim) <= self.cr_exl
        if not mask.any():
            mask[self.rng.integers(self.dim)] = True
        v[mask] = self.rng.normal(self.pop[self.best_idx][mask], 1.0)
        return self._clip(v)

    # --- Exploration-biased (rugged landscape) ---

    def _employed_exp(self, i: int) -> np.ndarray:
        """Eq. 2 (standard ABC): v = x_i + phi*(x_i - x_k)"""
        candidates = [j for j in range(self.sn) if j != i]
        k = int(self.rng.choice(candidates))
        j = int(self.rng.integers(self.dim))
        phi = self.rng.uniform(-1, 1)
        v = self.pop[i].copy()
        v[j] = self.pop[i][j] + phi * (self.pop[i][j] - self.pop[k][j])
        return self._clip(v)

    def _onlooker_exp(self, i: int) -> np.ndarray:
        """Eq. 16: v = x_i + phi*(x_i - x_r1) + phi*(x_best - x_r2)"""
        r1, r2 = self._rand_pair({i})
        phi1 = self.rng.uniform(-1, 1, self.dim)
        phi2 = self.rng.uniform(-1, 1, self.dim)
        v = (self.pop[i]
             + phi1 * (self.pop[i] - self.pop[r1])
             + phi2 * (self.pop[self.best_idx] - self.pop[r2]))
        return self._clip(v)

    def _scout_exp(self, i: int) -> np.ndarray:
        """
        Eq. 17: normal distribution centered between x_rlbest and x_best,
        σ = |x_rlbest - x_best|, dims chosen by CR_exp.
        """
        # Pick a random individual and find its ring-lbest
        k_rad = self._ring_k(self.fes)
        r = int(self.rng.integers(self.sn))
        rlb = self._lbest(r, k_rad)
        mu  = 0.5 * (self.pop[rlb] + self.pop[self.best_idx])
        sig = np.abs(self.pop[rlb] - self.pop[self.best_idx]) + 1e-12
        v   = self.pop[i].copy()
        mask = self.rng.random(self.dim) <= self.cr_exp
        if not mask.any():
            mask[self.rng.integers(self.dim)] = True
        v[mask] = self.rng.normal(mu[mask], sig[mask])
        return self._clip(v)

    # ── greedy selection ─────────────────────────────────────────────────────
    def _greedy(self, i: int, v: np.ndarray) -> bool:
        f_v = self.eval_func(v)
        if f_v < self.f_vals[i]:
            self.pop[i]    = v
            self.f_vals[i] = f_v
            self.trials[i] = 0
            if f_v < self.f_best:
                self.f_best  = f_v
                self.best_idx = i
            return True
        else:
            self.trials[i] += 1
            return False

    def eval_func(self, x: np.ndarray) -> float:
        self.fes += 1
        return float(self.func(x))

    # ── main loop ────────────────────────────────────────────────────────────
    def run(self, verbose: bool = False) -> tuple:
        """
        Returns (best_error, convergence_log).
        convergence_log: list of (fes, best_error) recorded at each iteration.
        """
        sn, dim = self.sn, self.dim
        lb, ub  = self.lb, self.ub

        # ── Initialization ──
        self.pop    = lb + self.rng.random((sn, dim)) * (ub - lb)
        self.f_vals = np.array([self.eval_func(self.pop[i]) for i in range(sn)])
        self.trials = np.zeros(sn, dtype=int)
        self.fes    = sn
        self.best_idx = int(np.argmin(self.f_vals))
        self.f_best   = self.f_vals[self.best_idx]

        # Selection probability (Eq. 4)
        fit_arr = np.array([_fit(v) for v in self.f_vals])

        # ILM state
        ph = 0.0
        iter_count = 0
        conv_log = [(self.fes, self.f_best)]

        while self.fes < self.max_fes:
            iter_count += 1
            k_rad = self._ring_k(self.fes)

            # ── Compute PH every ilm_interval iterations ──
            if iter_count % self.ilm_interval == 0:
                ph = _ilm_ph(self.pop, self.f_vals)

            use_exl = (ph < self.ph_bdy)   # smooth → exploitation-biased

            # ── Employed bee phase ──
            for i in range(sn):
                if self.fes >= self.max_fes:
                    break
                if use_exl:
                    v = self._employed_exl(i, k_rad)
                else:
                    v = self._employed_exp(i)
                self._greedy(i, v)

            # Update selection probabilities
            fit_arr = np.array([_fit(v) for v in self.f_vals])
            probs   = fit_arr / fit_arr.sum()

            # ── Onlooker bee phase ──
            selected = 0
            i = 0
            while selected < sn and self.fes < self.max_fes:
                if self.rng.random() < probs[i]:
                    if use_exl:
                        v = self._onlooker_exl(i, k_rad)
                    else:
                        v = self._onlooker_exp(i)
                    self._greedy(i, v)
                    selected += 1
                i = (i + 1) % sn

            # ── Scout bee phase ──
            for i in range(sn):
                if self.fes >= self.max_fes:
                    break
                if self.trials[i] > self.limit and i != self.best_idx:
                    if use_exl:
                        v = self._scout_exl(i)
                    else:
                        v = self._scout_exp(i)
                    f_v = self.eval_func(v)
                    self.pop[i]    = v
                    self.f_vals[i] = f_v
                    self.trials[i] = 0
                    if f_v < self.f_best:
                        self.f_best   = f_v
                        self.best_idx = i

            conv_log.append((self.fes, self.f_best))

        if verbose:
            print(f'  FEs={self.fes}  best={self.f_best:.6e}  PH={ph:.3f}')
        return self.f_best, conv_log

print('ASABC class defined ✓')

# ## 6 · Experiment Runner (with 30-min checkpoints)

import pickle

CHECKPOINT_FILE = os.path.join(BASE_DIR, 'checkpoint.pkl')
RESULTS_CSV     = os.path.join(BASE_DIR, 'results_raw.csv')


def save_checkpoint(state: dict):
    with open(CHECKPOINT_FILE, 'wb') as f:
        pickle.dump(state, f)
    print(f'  [checkpoint saved  {time.strftime("%H:%M:%S")}]')


def load_checkpoint() -> dict:
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'rb') as f:
            return pickle.load(f)
    return {}


def run_experiments():
    state = load_checkpoint()
    # state: { fid: { 'errors': [], 'conv_logs': [] } }

    t_last_save = time.time()

    for fid in tqdm(FUNC_IDS, desc='Functions'):
        if fid not in state:
            state[fid] = {'errors': [], 'conv_logs': []}

        completed = len(state[fid]['errors'])
        if completed >= N_RUNS:
            continue

        obj  = get_func(SUITE, fid, DIM)
        f_opt = obj.f_global

        def shifted_func(x, _obj=obj, _f_opt=f_opt):
            return _obj.evaluate(x) - _f_opt

        for run in tqdm(range(completed, N_RUNS), desc=f'  f{fid}', leave=False):
            algo = ASABC(
                func     = shifted_func,
                dim      = DIM,
                bounds   = BOUNDS,
                sn       = N_BEES,
                max_fes  = MAX_FES,
                seed     = run * 1000 + fid,
            )
            err, conv = algo.run()
            err = max(err, 0.0)   # numerical floor

            state[fid]['errors'].append(err)
            state[fid]['conv_logs'].append(conv)

            # periodic checkpoint
            if time.time() - t_last_save >= SAVE_INTERVAL:
                save_checkpoint(state)
                t_last_save = time.time()

        # force save after each function
        save_checkpoint(state)
        t_last_save = time.time()

    return state


print('Experiment runner defined ✓')
print('Run the next cell to START experiments.')

# ── RUN ───────────────────────────────────────────────────────────────────────
t0    = time.time()
state = run_experiments()
print(f'\nTotal time: {(time.time()-t0)/3600:.2f} h')

# ## 7 · Statistics & Publication Tables

# ── Build summary DataFrame ───────────────────────────────────────────────────
records = []
for fid in FUNC_IDS:
    errs = np.array(state[fid]['errors'])
    records.append({
        'Function' : f'f{fid}',
        'Mean'     : errs.mean(),
        'Std'      : errs.std(ddof=1),
        'Min'      : errs.min(),
        'Median'   : np.median(errs),
        'Max'      : errs.max(),
    })

df = pd.DataFrame(records).set_index('Function')

# Save CSV
csv_path = os.path.join(BASE_DIR, f'ASABC_{SUITE}_D{DIM}_summary.csv')
df.to_csv(csv_path, float_format='%.4e')
print(f'Summary saved → {csv_path}')
print(df.to_string())

# ── LaTeX table (publication-ready, matches CEC paper format) ─────────────────
def fmt_sci(x):
    """Format as e.g. 1.23E+04 matching CEC paper style."""
    if x == 0:
        return r'$\mathbf{0.00E+00}$'
    s = f'{x:.2E}'
    mantissa, exp = s.split('E')
    e_int = int(exp)
    return f'{mantissa}E{e_int:+03d}'

lines = []
lines.append(r'\begin{table}[htbp]')
lines.append(r'\centering')
lines.append(r'\caption{ASABC results on ' + SUITE + f' $D={DIM}$}}')
lines.append(r'\label{tab:asabc_' + SUITE.lower() + f'_d{DIM}' + r'}')
lines.append(r'\begin{tabular}{lll}')
lines.append(r'\toprule')
lines.append(r'Func. & Mean & Std \\')
lines.append(r'\midrule')
for fid in FUNC_IDS:
    row = df.loc[f'f{fid}']
    lines.append(f'f{fid} & {fmt_sci(row["Mean"])} & {fmt_sci(row["Std"])} \\\\')
lines.append(r'\bottomrule')
lines.append(r'\end{tabular}')
lines.append(r'\end{table}')

latex_str = '\n'.join(lines)
tex_path  = os.path.join(BASE_DIR, f'ASABC_{SUITE}_D{DIM}_table.tex')
with open(tex_path, 'w') as f:
    f.write(latex_str)
print('LaTeX table →', tex_path)
print(latex_str)

# ## 8 · Convergence Curves

def plot_convergence(state, fids_to_plot=None, n_runs_to_plot=None):
    if fids_to_plot is None:
        # Show a representative selection (unimodal, multimodal, hybrid, composition)
        if SUITE == 'CEC2017':
            fids_to_plot = [1, 3, 7, 12, 17, 21, 25, 29]
        else:
            fids_to_plot = [1, 3, 5, 7, 9, 11]

    ncols = 4
    nrows = math.ceil(len(fids_to_plot) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, fid in zip(axes_flat, fids_to_plot):
        logs = state[fid]['conv_logs']
        if n_runs_to_plot:
            logs = logs[:n_runs_to_plot]

        for log in logs:
            xs = [p[0] for p in log]
            ys = [max(p[1], 1e-15) for p in log]   # avoid log(0)
            ax.semilogy(xs, ys, alpha=0.25, linewidth=0.8, color='steelblue')

        # Median curve
        all_fes   = sorted({p[0] for log in logs for p in log})
        medians   = []
        for fes_val in all_fes:
            vals = []
            for log in logs:
                prev = 1e15
                for p in log:
                    if p[0] <= fes_val:
                        prev = p[1]
                    else:
                        break
                vals.append(prev)
            medians.append(np.median(vals))
        ax.semilogy(all_fes, [max(v, 1e-15) for v in medians],
                    color='crimson', linewidth=1.8, label='Median')

        ax.set_title(f'f{fid} ({SUITE}, D={DIM})', fontsize=9)
        ax.set_xlabel('FEs', fontsize=8)
        ax.set_ylabel('Error', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, which='both', linestyle='--', alpha=0.4)

    for ax in axes_flat[len(fids_to_plot):]:
        ax.axis('off')

    plt.tight_layout()
    fig_path = os.path.join(BASE_DIR, f'ASABC_{SUITE}_D{DIM}_convergence.pdf')
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.savefig(fig_path.replace('.pdf', '.png'), dpi=200, bbox_inches='tight')
    print('Figure saved →', fig_path)
    plt.show()


plot_convergence(state)

# ## 9 · Wilcoxon Signed-Rank Test vs Comparison Algorithms
# 
# Load the raw errors of the competitor algorithms (AutoABC, EABC-AS, etc.)  
# from their respective CSV files and run the Wilcoxon test (α = 0.05).

# ── Wilcoxon test ─────────────────────────────────────────────────────────────
# Drop competitor CSVs produced by the other notebooks into BASE_DIR
# Expected filename format: {AlgoName}_{SUITE}_D{DIM}_raw_errors.csv
#   rows = runs (51 or 30), columns = f1..f29 (or f1..f12)

ALPHA = 0.05

def wilcoxon_table(asabc_errors_dict: dict, competitor_csv: str) -> pd.DataFrame:
    """
    asabc_errors_dict: { fid: np.ndarray of shape (n_runs,) }
    competitor_csv: path to CSV with rows=runs, cols named f1..fN
    Returns DataFrame with +/=/-  per function.
    """
    comp_df = pd.read_csv(competitor_csv, index_col=0)
    results = []
    wins, ties, losses = 0, 0, 0
    for fid in FUNC_IDS:
        key  = f'f{fid}'
        a_er = asabc_errors_dict[fid]
        if key not in comp_df.columns:
            results.append({'Function': key, 'Result': '?'})
            continue
        c_er = comp_df[key].values
        n    = min(len(a_er), len(c_er))
        diff = a_er[:n] - c_er[:n]
        if np.all(diff == 0):
            sym = '='; ties += 1
        else:
            try:
                stat, p = wilcoxon(diff)
                if p >= ALPHA:
                    sym = '='; ties += 1
                elif np.median(diff) < 0:
                    sym = '+'; wins += 1   # ASABC better
                else:
                    sym = '-'; losses += 1 # ASABC worse
            except Exception:
                sym = '='; ties += 1
        results.append({'Function': key, 'Result': sym})
    print(f'  +/=/-  =  {wins}/{ties}/{losses}')
    return pd.DataFrame(results).set_index('Function')


# Example usage (uncomment when competitor CSVs are available):
# wt = wilcoxon_table(
#     {fid: np.array(state[fid]['errors']) for fid in FUNC_IDS},
#     os.path.join(BASE_DIR, f'EABCAS_{SUITE}_D{DIM}_raw_errors.csv'),
# )
# print(wt)
print('Wilcoxon helper defined ✓  — provide competitor CSVs to run.')

# ## 10 · Friedman Test & Ranking

# ── Friedman rank for a single algo (ASABC) among saved CSVs ─────────────────
# Looks for all *_summary.csv files in BASE_DIR and ranks them.

def friedman_rank_all(base_dir: str) -> pd.DataFrame:
    csv_files = [f for f in os.listdir(base_dir) if f.endswith('_summary.csv')]
    if not csv_files:
        print('No summary CSV files found in', base_dir)
        return pd.DataFrame()

    dfs = {}
    for fn in csv_files:
        algo = fn.split('_')[0]
        d    = pd.read_csv(os.path.join(base_dir, fn), index_col=0)
        dfs[algo] = d['Mean']

    combined = pd.DataFrame(dfs)   # rows = functions, cols = algos
    # Rank across algos for each function (lower mean = better rank = lower number)
    ranked = combined.rank(axis=1, method='average')
    avg_ranks = ranked.mean()
    avg_ranks = avg_ranks.sort_values()

    print('Average Friedman ranks:')
    print(avg_ranks.to_string())
    return avg_ranks


friedman_rank_all(BASE_DIR)

# ## 11 · Export Raw Errors CSV (for cross-notebook Wilcoxon tests)

raw_dict = {f'f{fid}': state[fid]['errors'] for fid in FUNC_IDS}
raw_df   = pd.DataFrame(raw_dict)
raw_path = os.path.join(BASE_DIR, f'ASABC_{SUITE}_D{DIM}_raw_errors.csv')
raw_df.to_csv(raw_path, float_format='%.6e')
print('Raw errors saved →', raw_path)

# ## 12 · Parameter Sensitivity (optional — reproduces §4.6)
# 
# Run a quick sweep over `ph_bdy`, `CR_exl`, `CR_exp`, `limit` on a  
# subset of CEC functions to validate default choices.

# ── Sensitivity sweep (runs 10 independent trials per config) ─────────────────
SENS_RUNS   = 10
SENS_FUNCS  = FUNC_IDS[:6]   # first 6 functions for speed

def sweep_param(param_name: str, values: list):
    results = {}
    for val in tqdm(values, desc=param_name):
        errs_all = []
        for fid in SENS_FUNCS:
            obj   = get_func(SUITE, fid, DIM)
            f_opt = obj.f_global
            kw    = dict(ph_bdy=0.3, cr_exl=0.1, cr_exp=0.7, limit=50)
            kw[param_name] = val
            for run in range(SENS_RUNS):
                algo = ASABC(
                    func    = lambda x, o=obj, f=f_opt: o.evaluate(x) - f,
                    dim     = DIM,
                    bounds  = BOUNDS,
                    sn      = N_BEES,
                    max_fes = MAX_FES,
                    seed    = run + fid * 100,
                    **kw,
                )
                err, _ = algo.run()
                errs_all.append(max(err, 0.0))
        results[val] = np.mean(errs_all)
    # Plot
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(list(results.keys()), list(results.values()), marker='o')
    ax.set_xlabel(param_name)
    ax.set_ylabel('Mean error (avg over functions)')
    ax.set_title(f'Sensitivity: {param_name}  ({SUITE} D={DIM})')
    ax.grid(True, linestyle='--', alpha=0.5)
    fig_p = os.path.join(BASE_DIR, f'ASABC_sensitivity_{param_name}.pdf')
    plt.savefig(fig_p, dpi=150, bbox_inches='tight')
    plt.show()
    print('Best value:', min(results, key=results.get))
    return results


# Uncomment the sweep(s) you want:
# sweep_param('ph_bdy',  [0.1, 0.2, 0.3, 0.4, 0.5])
# sweep_param('cr_exl',  [0.1, 0.3, 0.5, 0.7, 0.9])
# sweep_param('cr_exp',  [0.1, 0.3, 0.5, 0.7, 0.9])
# sweep_param('limit',   [10,  30,  50,  100, 200])
print('Sensitivity sweep helper defined ✓  — uncomment to run.')

# ## 13 · Box Plots

def plot_boxplots(state, fids_to_plot=None):
    if fids_to_plot is None:
        fids_to_plot = FUNC_IDS[:12]
    data   = [np.log10(np.maximum(np.array(state[fid]['errors']), 1e-15))
               for fid in fids_to_plot]
    labels = [f'f{fid}' for fid in fids_to_plot]

    fig, ax = plt.subplots(figsize=(max(10, len(fids_to_plot) * 0.8), 4))
    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    medianprops=dict(color='crimson', linewidth=2))
    for patch in bp['boxes']:
        patch.set_facecolor('lightsteelblue')
        patch.set_alpha(0.7)
    ax.set_ylabel('log₁₀(Error)')
    ax.set_title(f'ASABC — {SUITE} D={DIM}')
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    fig_p = os.path.join(BASE_DIR, f'ASABC_{SUITE}_D{DIM}_boxplots.pdf')
    plt.savefig(fig_p, dpi=200, bbox_inches='tight')
    print('Box plot saved →', fig_p)
    plt.show()


plot_boxplots(state)

# ---
# **End of ASABC notebook.**  
# Change `SUITE` / `DIM` in §3 and re-run to cover all configurations.
