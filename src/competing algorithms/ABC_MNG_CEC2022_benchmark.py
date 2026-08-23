#!/usr/bin/env python
# coding: utf-8

# # ABC-MNG on CEC2022 (D = 10 / 20)
# 
# **Algorithm:** Artificial Bee Colony with Multi-Neighbor Guidance (ABC-MNG)  
# **Reference:** Zhou et al., *Expert Systems With Applications* 259 (2025) 125283  
# 
# ## Experimental Protocol (CEC2022-specific)
# | Parameter | Value |
# |-----------|-------|
# | Population (SN) | 30 |
# | Independent runs | 30 |
# | Max FEs | 200 000 (D=10) / 1 000 000 (D=20) |
# | Ring radius R | 0.1 × SN = 3 |
# | CRe (employed) | 0.2 |
# | CRo (onlooker) | 0.1 |
# | limit | 100 |
# | Bounds | [−100, 100]^D |
# | Functions | F1–F12 |
# | Zero threshold | 1e-8 |
# 
# **Outputs:** mean error, std, Wilcoxon rank-sum vs EABC-AS,  
# Friedman ranking, convergence curves, LaTeX tables — all publication-ready.
# 

# In[1]:


# --- 0. Dependencies --------------------------------------------------------
!pip install "setuptools<70.0.0" opfunu
import subprocess, sys

def pip(*pkgs):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *pkgs])

try:
    import opfunu
except ImportError:
    pip('opfunu>=1.0.3')

try:
    import scipy
except ImportError:
    pip('scipy')

import importlib, opfunu
importlib.reload(opfunu)
print('opfunu version:', opfunu.__version__)

# In[2]:


# --- 1. Imports -------------------------------------------------------------
import os, time, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, friedmanchisquare
from scipy.stats import rankdata
from opfunu.cec_based import cec2022
warnings.filterwarnings('ignore')

# --- 2. Global configuration (CEC2022-specific) ----------------------------
# Set DIM to 10 or 20
DIM  = 20          # <-- CHANGE TO 20 FOR SECOND DIMENSION 10 est fait

N_BEES        = 30
N_RUNS        = 30
MAX_FES       = 200_000 if DIM == 10 else 1_000_000
MAX_ITER      = MAX_FES // N_BEES
BOUNDS_LO     = -100.0
BOUNDS_HI     =  100.0
N_FUNCS       = 12          # F1 ... F12
LIMIT         = 100
R_RADIUS      = max(1, int(0.1 * N_BEES))   # = 3
CR_E          = 0.2
CR_O          = 0.1
ZERO_THRESH   = 1e-8
SAVE_INTERVAL = 1800        # seconds -- auto-save every 30 min

BASE_DIR = rf'C:\AbCMNG_CEC2022\ABCMNG_D{DIM}'
os.makedirs(BASE_DIR, exist_ok=True)

CHECKPOINT_FILE = os.path.join(BASE_DIR, f'checkpoint_D{DIM}.pkl')
RESULTS_FILE    = os.path.join(BASE_DIR, f'results_D{DIM}.pkl')

print(f'CEC2022  DIM={DIM}  MAX_FES={MAX_FES}  MAX_ITER={MAX_ITER}  R={R_RADIUS}')
print(f'Output directory: {BASE_DIR}')

# In[3]:


# --- 3. CEC2022 function loader ---------------------------------------------
# opfunu CEC2022 class names: F12022, F22022, ... F122022

def get_cec2022_func(fid, ndim):
    """Return an opfunu CEC2022 function instance (1-indexed fid)."""
    cls_name = f'F{fid}2022'
    cls = getattr(cec2022, cls_name)
    return cls(ndim=ndim)

# Quick sanity check
_f = get_cec2022_func(1, DIM)
x0  = np.random.uniform(BOUNDS_LO, BOUNDS_HI, DIM)
val = _f.evaluate(x0) - _f.f_global
print(f'F1 (CEC2022) test call OK, error = {val:.4e}')

# In[4]:


# --- 4. ABC-MNG core --------------------------------------------------------

def clip(x, lo, hi):
    return np.clip(x, lo, hi)

def random_init(n, d, lo, hi, rng):
    return rng.uniform(lo, hi, (n, d))

def ring_neighbors(i, sn, R):
    return [(i + k) % sn for k in range(-R, R+1) if k != 0]

def pick_three_neighbors(pop, fit, i, R):
    sn = len(pop); xi = pop[i]
    nbrs    = ring_neighbors(i, sn, R)
    nbr_fit = fit[nbrs]
    dists   = np.linalg.norm(pop[nbrs] - xi, axis=1)
    best_idx     = nbrs[int(np.argmax(nbr_fit))]
    nearest_idx  = nbrs[int(np.argmin(dists))]
    farthest_idx = nbrs[int(np.argmax(dists))]
    return best_idx, nearest_idx, farthest_idx

def modified_search_eq(xi, x_star, x_r, CR, D, rng):
    j_rand = rng.integers(0, D)
    phi    = rng.uniform(-1, 1, D)
    mask   = (rng.random(D) <= CR)
    mask[j_rand] = True
    return np.where(mask, x_star + phi * (x_star - x_r), xi)

def cosine_similarity(v, x):
    nv = np.linalg.norm(v); nx = np.linalg.norm(x)
    if nv < 1e-300 or nx < 1e-300: return 0.0
    return float(np.dot(v, x) / (nv * nx))

def fitness_from_obj(obj_val):
    if obj_val >= 0: return 1.0 / (1.0 + obj_val)
    return 1.0 + abs(obj_val)

def select_offspring(candidates, xi, rank_i, sn, rng):
    cos_vals   = np.array([c[1] for c in candidates])
    best_local = int(np.argmax(cos_vals))
    prob_best  = 1.0 - rank_i / sn
    if rng.random() < prob_best:
        return candidates[best_local][0]
    others = [j for j in range(len(candidates)) if j != best_local]
    return candidates[rng.choice(others)][0]

def modified_gns(xi, gbest, pop, i, R, sn, rng):
    nbrs  = ring_neighbors(i, sn, R)
    xi_   = pop[i]
    dists = np.linalg.norm(pop[nbrs] - xi_, axis=1)
    near_idx = nbrs[int(np.argmin(dists))]
    far_idx  = nbrs[int(np.argmax(dists))]
    r = rng.dirichlet([1, 1, 1])
    return r[0]*xi_ + r[1]*gbest + r[2]*(pop[near_idx] - pop[far_idx])


def abc_mng(func, ndim, rng, record_curve=False):
    lo, hi = BOUNDS_LO, BOUNDS_HI
    sn = N_BEES; D = ndim; R = R_RADIUS
    f_opt = func.f_global

    pop    = random_init(sn, D, lo, hi, rng)
    obj    = np.array([func.evaluate(pop[i]) for i in range(sn)])
    fit    = np.vectorize(fitness_from_obj)(obj - f_opt)
    trials = np.zeros(sn, dtype=int)
    fes    = sn

    gbest_idx = int(np.argmax(fit))
    gbest     = pop[gbest_idx].copy()
    gbest_obj = obj[gbest_idx]

    curve = []
    if record_curve:
        err = max(gbest_obj - f_opt, 0.0)
        if err < ZERO_THRESH: err = 0.0
        curve.append((fes, err))

    max_fes = MAX_FES

    while fes < max_fes:
        ranks = rankdata(-fit).astype(int)

        # Employed bee phase
        for i in range(sn):
            if fes >= max_fes: break
            best_n, near_n, far_n = pick_three_neighbors(pop, fit, i, R)
            r_pool = [j for j in range(sn) if j != i]
            x_r    = pop[rng.choice(r_pool)]
            candidates = []
            for x_star_idx in [best_n, near_n, far_n]:
                x_star = pop[x_star_idx]
                v      = clip(modified_search_eq(pop[i], x_star, x_r, CR_E, D, rng), lo, hi)
                candidates.append((v, cosine_similarity(v, pop[i])))
            v_sel = select_offspring(candidates, pop[i], int(ranks[i]), sn, rng)
            v_obj = func.evaluate(v_sel); fes += 1
            if v_obj <= obj[i]:
                pop[i] = v_sel; obj[i] = v_obj
                fit[i] = fitness_from_obj(v_obj - f_opt); trials[i] = 0
                if v_obj < gbest_obj: gbest_obj = v_obj; gbest = v_sel.copy()
            else:
                trials[i] += 1

        # Onlooker bee phase
        S = np.ones(3) * 1e-10
        for i in range(sn):
            best_n, near_n, far_n = pick_three_neighbors(pop, fit, i, R)
            S += np.maximum(np.array([fit[best_n], fit[near_n], fit[far_n]]) - fit[i], 0)
        p = S / S.sum()

        for i in range(sn):
            if fes >= max_fes: break
            t_choice  = rng.choice(3, p=p)
            best_n, near_n, far_n = pick_three_neighbors(pop, fit, i, R)
            x_hat_idx = [best_n, near_n, far_n][t_choice]
            x_b_idx   = best_n
            x_hat = pop[x_hat_idx]; x_b = pop[x_b_idx]
            r_pool = [j for j in range(sn) if j != i]
            x_r    = pop[rng.choice(r_pool)]
            j_rand = rng.integers(0, D)
            phi    = rng.uniform(-1, 1, D)
            mask   = (rng.random(D) <= CR_O); mask[j_rand] = True
            v      = clip(np.where(mask, x_hat + phi*(x_hat - x_r), x_b), lo, hi)
            v_obj  = func.evaluate(v); fes += 1
            if v_obj <= obj[x_b_idx]:
                pop[x_b_idx] = v; obj[x_b_idx] = v_obj
                fit[x_b_idx] = fitness_from_obj(v_obj - f_opt); trials[x_b_idx] = 0
                if v_obj < gbest_obj: gbest_obj = v_obj; gbest = v.copy()
            else:
                trials[x_b_idx] += 1

        # Scout bee phase -- modified GNS
        for i in range(sn):
            if fes >= max_fes: break
            if trials[i] > LIMIT:
                tx     = clip(modified_gns(pop[i], gbest, pop, i, R, sn, rng), lo, hi)
                tx_obj = func.evaluate(tx); fes += 1
                pop[i] = tx; obj[i] = tx_obj
                fit[i] = fitness_from_obj(tx_obj - f_opt); trials[i] = 0
                if tx_obj < gbest_obj: gbest_obj = tx_obj; gbest = tx.copy()

        best_cur = int(np.argmax(fit))
        if obj[best_cur] < gbest_obj:
            gbest_obj = obj[best_cur]; gbest = pop[best_cur].copy()

        if record_curve:
            err = max(gbest_obj - f_opt, 0.0)
            if err < ZERO_THRESH: err = 0.0
            curve.append((fes, err))

    final_err = max(gbest_obj - f_opt, 0.0)
    if final_err < ZERO_THRESH: final_err = 0.0
    return final_err, curve


print('ABC-MNG (CEC2022) functions defined OK')

# In[5]:


# --- 5. Checkpoint helpers --------------------------------------------------

def save_checkpoint(state, path):
    with open(path, 'wb') as f: pickle.dump(state, f)
    print(f'  [checkpoint saved -> {path}]')

def load_checkpoint(path):
    if os.path.exists(path):
        with open(path, 'rb') as f: state = pickle.load(f)
        print(f'  [checkpoint loaded <- {path}]')
        return state
    return None

def save_results(results, path):
    with open(path, 'wb') as f: pickle.dump(results, f)
    print(f'  [results saved -> {path}]')

print('Checkpoint helpers ready')

# In[ ]:


# --- 6. Main experimental loop ----------------------------------------------

FUNC_IDS = list(range(1, N_FUNCS + 1))   # F1 ... F12

ckpt = load_checkpoint(CHECKPOINT_FILE)
if ckpt is not None:
    all_errors = ckpt['all_errors']
    all_curves = ckpt['all_curves']
    start_fid  = ckpt['last_fid'] + 1
    print(f'Resuming from F{start_fid}')
else:
    all_errors = {fid: [] for fid in FUNC_IDS}
    all_curves = {fid: [] for fid in FUNC_IDS}
    start_fid  = 1

last_save_time = time.time()

for fid in FUNC_IDS:
    if fid < start_fid: continue

    func = get_cec2022_func(fid, DIM)
    print(f'\nF{fid:02d}  (fopt={func.f_global:.4e})', end='  runs: ', flush=True)

    for run in range(N_RUNS):
        rng    = np.random.default_rng(seed=run * 100 + fid)
        record = (run == 0)
        err, curve = abc_mng(func, DIM, rng, record_curve=record)
        all_errors[fid].append(err)
        if record: all_curves[fid] = curve
        print('.', end='', flush=True)

        if (time.time() - last_save_time) >= SAVE_INTERVAL:
            save_checkpoint({'all_errors': all_errors, 'all_curves': all_curves,
                             'last_fid': fid}, CHECKPOINT_FILE)
            last_save_time = time.time()

    m = np.mean(all_errors[fid]); s = np.std(all_errors[fid])
    print(f'  mean={m:.4e}  std={s:.4e}')
    save_checkpoint({'all_errors': all_errors, 'all_curves': all_curves,
                     'last_fid': fid}, CHECKPOINT_FILE)
    last_save_time = time.time()

save_results({'errors': all_errors, 'curves': all_curves,
              'dim': DIM, 'n_runs': N_RUNS}, RESULTS_FILE)
print('\n=== All CEC2022 functions completed ===')

# In[ ]:


# --- 7. Summary statistics --------------------------------------------------

rows = []
for fid in FUNC_IDS:
    errs = np.array(all_errors[fid])
    errs[errs < ZERO_THRESH] = 0.0
    rows.append({'Func': f'F{fid:02d}',
                 'Mean':   np.mean(errs), 'Std':    np.std(errs),
                 'Min':    np.min(errs),  'Max':    np.max(errs),
                 'Median': np.median(errs)})

df_stats = pd.DataFrame(rows).set_index('Func')

# CORRECTION ICI : Remplacement de applymap par map
df_display = df_stats[['Mean','Std']].map(lambda x: f'{x:.2E}')

print(f'\nABC-MNG on CEC2022, D={DIM}')
print(df_display.to_string())

csv_path = os.path.join(BASE_DIR, f'stats_D{DIM}.csv')
df_stats.to_csv(csv_path, float_format='%.6E')
print(f'\nCSV saved: {csv_path}')

# In[ ]:


# --- 8. Wilcoxon rank-sum test vs EABC-AS -----------------------------------

EABC_AS_FILE = os.path.join(BASE_DIR, f'eabc_as_errors_D{DIM}.pkl')

def wilcoxon_test(a, b, alpha=0.05):
    a = np.array(a); b = np.array(b)
    if np.all(a - b == 0): return '='
    try:
        _, p = wilcoxon(a, b, alternative='two-sided', zero_method='wilcox')
    except Exception:
        return '='
    if p >= alpha: return '='
    return '+' if np.median(a) < np.median(b) else '-'

if os.path.exists(EABC_AS_FILE):
    with open(EABC_AS_FILE, 'rb') as f:
        eabc_as_errors = pickle.load(f)
    wx_rows = []
    for fid in FUNC_IDS:
        a_err = all_errors[fid]
        b_err = eabc_as_errors.get(fid, [np.nan]*N_RUNS)
        sign  = wilcoxon_test(a_err, b_err)
        wx_rows.append({'Func': f'F{fid:02d}',
                        'ABC-MNG Mean': f'{np.mean(a_err):.2E}',
                        'ABC-MNG Std':  f'{np.std(a_err):.2E}',
                        'EABC-AS Mean': f'{np.mean(b_err):.2E}',
                        'EABC-AS Std':  f'{np.std(b_err):.2E}',
                        'W-test': sign})
    df_wx = pd.DataFrame(wx_rows).set_index('Func')
    counts = df_wx['W-test'].value_counts()
    p, e, m = counts.get('+',0), counts.get('=',0), counts.get('-',0)
    print(df_wx.to_string())
    print(f'\n+/=/-  {p}/{e}/{m}')
    df_wx.to_csv(os.path.join(BASE_DIR, f'wilcoxon_vs_EABCAS_D{DIM}.csv'))
else:
    print(f'EABC-AS file not found: {EABC_AS_FILE}')
    print('Place dict pkl there then re-run.')

# In[ ]:


# --- 9. Friedman ranking ----------------------------------------------------

algo_means = {'ABC-MNG': [np.mean(all_errors[f]) for f in FUNC_IDS]}
# Add competitors: algo_means['EABC-AS'] = [...]

if len(algo_means) >= 2:
    data   = np.array(list(algo_means.values())).T
    ranked = np.apply_along_axis(rankdata, 1, data)
    avg_ranks = dict(zip(algo_means.keys(), ranked.mean(axis=0)))
    print('Friedman average ranks:')
    for k, v in sorted(avg_ranks.items(), key=lambda x: x[1]):
        print(f'  {k:20s}: {v:.3f}')
    stat, p = friedmanchisquare(*[ranked[:,j] for j in range(ranked.shape[1])])
    print(f'chi2={stat:.4f}, p={p:.4e}')

    fig, ax = plt.subplots(figsize=(max(4, len(algo_means)*1.5), 4))
    names = list(avg_ranks.keys()); vals = [avg_ranks[n] for n in names]
    colors = ['#d62728' if n == 'ABC-MNG' else '#1f77b4' for n in names]
    bars = ax.bar(names, vals, color=colors, edgecolor='black', width=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.03,
                f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Average Ranking', fontsize=11)
    ax.set_title(f'Friedman Ranking -- CEC2022 D={DIM}', fontsize=12)
    ax.set_ylim(0, max(vals)+0.5); plt.tight_layout()
    fp = os.path.join(BASE_DIR, f'friedman_D{DIM}.pdf')
    plt.savefig(fp, dpi=300, bbox_inches='tight')
    plt.savefig(fp.replace('.pdf','.png'), dpi=300, bbox_inches='tight')
    plt.show(); print(f'Saved: {fp}')
else:
    print('Add competitor means to compute Friedman ranking.')

# In[ ]:


# --- 10. Convergence curves (all 12 functions) ------------------------------

ncols = 4; nrows = 3
fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
axes = axes.flatten()

for ax, fid in zip(axes, FUNC_IDS):
    curve = all_curves.get(fid, [])
    if not curve: ax.set_visible(False); continue
    fes_v = [c[0] for c in curve]
    err_v = [max(c[1], 1e-50) for c in curve]
    ax.semilogy(fes_v, err_v, color='red', linewidth=1.5, label='ABC-MNG')
    ax.set_xlabel('FEs', fontsize=10); ax.set_ylabel('Error (lg)', fontsize=10)
    ax.set_title(f'F{fid:02d}', fontsize=11); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle(f'ABC-MNG Convergence -- CEC2022 D={DIM}', fontsize=13, y=1.01)
plt.tight_layout()
cp = os.path.join(BASE_DIR, f'convergence_D{DIM}.pdf')
plt.savefig(cp, dpi=300, bbox_inches='tight')
plt.savefig(cp.replace('.pdf','.png'), dpi=300, bbox_inches='tight')
plt.show(); print(f'Saved: {cp}')

# In[ ]:


# --- 11. LaTeX table (publication-ready) ------------------------------------

def fmt_sci(x):
    s = f'{x:.2E}'
    m, e = s.split('E')
    return f'{m}E{int(e):+03d}'

lines = [
    r'\begin{table}[htbp]', r'\centering',
    r'\caption{ABC-MNG on CEC2022 ($D=' + str(DIM) + r'$, 30 runs).}',
    r'\label{tab:abcmng_cec2022_D' + str(DIM) + r'}',
    r'\begin{tabular}{lrr}', r'\hline',
    r'Func & Mean & Std \\', r'\hline'
]
for fid in FUNC_IDS:
    errs = np.array(all_errors[fid]); errs[errs < ZERO_THRESH] = 0.0
    m = np.mean(errs); s = np.std(errs)
    lines.append(f'F{fid:02d} & ${fmt_sci(m)}$ & ${fmt_sci(s)}$ \\\\')
lines += [r'\hline', r'\end{tabular}', r'\end{table}']

latex_str = '\n'.join(lines)
tp = os.path.join(BASE_DIR, f'table_abcmng_cec2022_D{DIM}.tex')
with open(tp, 'w') as f: f.write(latex_str)
print(latex_str)
print(f'\nSaved: {tp}')

# In[ ]:


# --- 12. LaTeX comparison table (ABC-MNG vs EABC-AS) -----------------------

if 'df_wx' in dir():
    body = []
    for _, row in df_wx.iterrows():
        body.append(f"{row.name} & ${row['ABC-MNG Mean']}$ & ${row['ABC-MNG Std']}$ "
                    f"& ${row['EABC-AS Mean']}$ & ${row['EABC-AS Std']}$ "
                    f"& {row['W-test']} \\\\")
    counts = df_wx['W-test'].value_counts()
    p2, e2, m2 = counts.get('+',0), counts.get('=',0), counts.get('-',0)
    tex = ('\n'.join([
        r'\begin{table}[htbp]', r'\centering',
        r'\caption{ABC-MNG vs EABC-AS on CEC2022 ($D=' + str(DIM) + r'$)}',
        r'\begin{tabular}{lccccc}', r'\hline',
        r'Func & \multicolumn{2}{c}{ABC-MNG} & \multicolumn{2}{c}{EABC-AS} & W \\',
        r' & Mean & Std & Mean & Std & \\', r'\hline',
    ]) + '\n' + '\n'.join(body) + '\n' +
    r'\hline' + '\n' +
    f'+/=/-  & \\multicolumn{{5}}{{c}}{{{p2}/{e2}/{m2}}} \\\\' + '\n' +
    r'\hline' + '\n' + r'\end{tabular}' + '\n' + r'\end{table}')
    tp2 = os.path.join(BASE_DIR, f'comparison_vs_EABCAS_cec2022_D{DIM}.tex')
    with open(tp2, 'w') as f: f.write(tex)
    print(tex); print(f'\nSaved: {tp2}')
else:
    print('Run cell 8 with EABC-AS data first.')

# In[ ]:


# --- 13. Optimum-reached analysis -------------------------------------------
# Key finding per study: ABC-MNG achieves exact optimum on F10, F12 (CEC2022)
# where EABC-AS fails.

zero_funcs = [fid for fid in FUNC_IDS if np.mean(all_errors[fid]) == 0.0]
print(f'ABC-MNG functions with mean error = 0 (D={DIM}):')
if zero_funcs:
    for fid in zero_funcs: print(f'  F{fid:02d}')
else:
    print('  None')

if os.path.exists(EABC_AS_FILE):
    adv = [fid for fid in FUNC_IDS
           if np.mean(all_errors[fid]) == 0.0
           and np.mean(eabc_as_errors.get(fid, [1.0])) > 0.0]
    print(f'\nABC-MNG=0 but EABC-AS fails (D={DIM}):')
    for fid in adv:
        print(f'  F{fid:02d}  EABC-AS mean={np.mean(eabc_as_errors[fid]):.2E}')

# In[ ]:


# --- 14. Final summary ------------------------------------------------------

print('=' * 60)
print(f'  ABC-MNG CEC2022  D={DIM}  summary')
print('=' * 60)
n_zero = sum(np.mean(all_errors[fid]) == 0.0 for fid in FUNC_IDS)
print(f'  Functions solved to optimum (mean=0): {n_zero}/{N_FUNCS}')
print(f'  Output directory : {BASE_DIR}')
print('  Files generated:')
for fn in sorted(os.listdir(BASE_DIR)):
    print(f'    {fn}')
