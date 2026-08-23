#!/usr/bin/env python
# coding: utf-8

# # ABC-ANT — CEC2022 Benchmark (Publication-grade)
# **Paper:** Zhou et al., *Information Sciences* 610 (2022) 1078-1101
# **Protocol:** CEC2022 · D=10/20 · 30 runs · MaxFEs=200K/1M
# **Tests:** Wilcoxon · Friedman · Nemenyi · CD-diagram

# ## 1 · Install dependencies

!pip install "setuptools<70.0.0" opfunu
import importlib, subprocess, sys

def ensure(pkg, import_as=None):
    name = import_as or pkg.split('[')[0]
    if importlib.util.find_spec(name) is None:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])

ensure('opfunu')
ensure('scikit-posthocs', 'scikit_posthocs')
ensure('scipy')
ensure('pandas')
ensure('matplotlib')
ensure('seaborn')
ensure('tqdm')
print('All dependencies ready.')


# ## 2 · Imports & style

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy import stats
from scipy.stats import friedmanchisquare, mannwhitneyu
import scikit_posthocs as sp
from tqdm.auto import tqdm
import time, warnings, copy, pickle
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Callable, Dict
warnings.filterwarnings('ignore')
np.set_printoptions(precision=4, suppress=True)
GLOBAL_SEED = 2024
plt.rcParams.update({
    'font.family':'DejaVu Serif','font.size':11,
    'axes.titlesize':12,'axes.labelsize':11,
    'xtick.labelsize':9,'ytick.labelsize':9,
    'legend.fontsize':9,'figure.dpi':150,
    'savefig.dpi':300,'savefig.bbox':'tight',
    'axes.grid':True,'grid.alpha':0.3,'lines.linewidth':1.5,
})
RESULTS_DIR = Path(f'results_ANC_ant_D20_cec2022')
RESULTS_DIR.mkdir(exist_ok=True)
print('Setup done. Saving to:', RESULTS_DIR.resolve())


# ## 3 · CEC2022 protocol constants

DIM        = 20          # set to 20 for full publication run
N_BEES     = 30
N_RUNS     = 30
MAX_FES    = 200_000 if DIM == 10 else 1_000_000
MAX_ITER   = MAX_FES // N_BEES
BOUNDS     = [(-100.0, 100.0)] * DIM
N_FUNCS    = 12
ALPHA      = 0.05
LOG_POINTS = [0.01,0.02,0.03,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
LOG_FES    = [int(p * MAX_FES) for p in LOG_POINTS]
print(f'DIM={DIM}  N_BEES={N_BEES}  N_RUNS={N_RUNS}  MAX_FES={MAX_FES:,}')


# ## 4 · CEC2022 benchmark functions

try:
    from opfunu.cec_based.cec2022 import (
        F12022, F22022, F32022, F42022, F52022, F62022,
        F72022, F82022, F92022, F102022, F112022, F122022,
    )
    _CLS = [F12022,F22022,F32022,F42022,F52022,F62022,
            F72022,F82022,F92022,F102022,F112022,F122022]
    USE_OPFUNU = True
    print('opfunu CEC2022 loaded.')
except ImportError:
    USE_OPFUNU = False
    print('opfunu not found — using surrogate functions.')

def make_func(idx):
    if USE_OPFUNU:
        obj   = _CLS[idx-1](ndim=DIM)
        f_opt = obj.f_global
        def evaluate(x): return float(obj.evaluate(np.asarray(x)))
    else:
        rng   = np.random.default_rng(idx * 999)
        shift = rng.uniform(-50, 50, DIM)
        scale = float(idx)
        f_opt = 100.0 * idx
        def evaluate(x): return float(np.sum(scale*(np.asarray(x)-shift)**2)) + f_opt
    return evaluate, f_opt

f, fopt = make_func(1)
print(f'F1 sanity: f(0)={f(np.zeros(DIM)):.4f}  f_opt={fopt:.4f}')


# ## 5 · Fitness-Distance Correlation (Eq. 6-7)

def compute_fdc(population, fitness):
    best_idx  = int(np.argmax(fitness))
    best      = population[best_idx]
    rest_pop  = np.delete(population, best_idx, axis=0)
    rest_fit  = np.delete(fitness,    best_idx)
    if len(rest_pop) < 2: return 0.0
    distances = np.linalg.norm(rest_pop - best, axis=1)
    if distances.std() < 1e-15 or rest_fit.std() < 1e-15: return 0.0
    r, _ = stats.pearsonr(rest_fit, distances)
    return float(r) if np.isfinite(r) else 0.0

def select_topology(r):
    if   r > 0.75:  return 'random'
    elif r >= 0.15: return 'ring'
    else:           return 'cellular'

print('FDC ready.')


# ## 6 · Neighborhood topology builders

def get_neighbors_random(i, sn, rng, k=None):
    k = k or max(2, int(0.3 * sn))
    cnd = [j for j in range(sn) if j != i]
    return rng.choice(cnd, size=min(k, len(cnd)), replace=False)

def get_neighbors_ring(i, sn, radius=None):
    radius = radius or max(1, int(0.1 * sn))
    return np.array([(i+d) % sn for d in range(-radius, radius+1) if d != 0])

def get_neighbors_cellular(i, sn):
    rows = max(2, int(np.sqrt(sn)))
    cols = (sn + rows - 1) // rows
    r, c = divmod(i, cols)
    nbrs = [(r-1)*cols+c, (r+1)*cols+c, r*cols+(c-1), r*cols+(c+1)]
    return np.array([n % sn for n in nbrs if n != i])

def best_neighbor(indices, fitness):
    return int(indices[np.argmax(fitness[indices])])

print('Topology builders ready.')


# ## 7 · ABC-ANT (all equations from the paper)

@dataclass
class ABCANTConfig:
    sn         : int   = 30
    limit      : int   = 100
    cr         : float = 0.5
    elite_frac : float = 0.10
    w_min      : float = 0.0
    w_max      : float = 1.5


def abc_ant(evaluate, bounds, f_opt, config, max_fes, log_fes, seed):
    rng = np.random.default_rng(seed)
    dim = len(bounds)
    lo  = np.array([b[0] for b in bounds])
    hi  = np.array([b[1] for b in bounds])
    sn  = config.sn
    pop    = lo + rng.random((sn, dim)) * (hi - lo)  # Eq.1
    obj    = np.array([evaluate(pop[i]) for i in range(sn)])
    fes    = sn
    def tof(o): return 1.0/(1.0+o) if o>=0 else 1.0+abs(o)  # Eq.3
    fit    = np.array([tof(o) for o in obj])
    trials = np.zeros(sn, dtype=int)
    gb     = int(np.argmax(fit))
    topology   = 'ring'
    gb_updated = True
    curve = []; ptr = 0

    def rec():
        nonlocal ptr
        while ptr < len(log_fes) and fes >= log_fes[ptr]:
            curve.append(max(0.0, obj[gb] - f_opt)); ptr += 1

    def elites():
        k = max(2, int(config.elite_frac * sn))
        return np.argsort(fit)[-k:]

    def clip_elite(v, j):
        if v < lo[j] or v > hi[j]: return float(pop[rng.choice(elites())][j])
        return v

    def get_nbrs(i):
        if topology == 'random':   return get_neighbors_random(i, sn, rng)
        elif topology == 'ring':   return get_neighbors_ring(i, sn)
        else:                      return get_neighbors_cellular(i, sn)

    rec()
    while fes < max_fes:
        if not gb_updated:
            topology = select_topology(compute_fdc(pop, fit))
        prev_gb = gb

        # Employed bee phase (Eq.9)
        for i in range(sn):
            nbrs  = get_nbrs(i)
            nb    = best_neighbor(nbrs, fit)
            xnb   = pop[nb]
            excl  = {i, nb}
            cands = [j for j in range(sn) if j not in excl]
            xr    = pop[rng.choice(cands)]
            phi   = rng.uniform(-1, 1, dim)
            w     = rng.uniform(config.w_min, config.w_max, dim)
            v     = xnb + phi*(xnb - xr) + w*(pop[gb] - xnb)
            v     = np.array([clip_elite(v[j], j) for j in range(dim)])
            ov    = evaluate(v); fes += 1; fv = tof(ov)
            if fv > fit[i]: pop[i],obj[i],fit[i],trials[i] = v,ov,fv,0
            else:           trials[i] += 1
            if fes >= max_fes: break
        gb = int(np.argmax(fit)); rec()
        if fes >= max_fes: break

        # Onlooker bee phase (Eq.10)
        for i in range(sn):
            nbrs  = get_nbrs(i)
            nb    = best_neighbor(nbrs, fit)
            xnb   = pop[nb]
            excl  = {i, nb}
            cands = [j for j in range(sn) if j not in excl]
            xr    = pop[rng.choice(cands)]
            phi   = rng.uniform(-1, 1, dim)
            v     = xnb + phi*(xnb - xr)
            v     = np.array([clip_elite(v[j], j) for j in range(dim)])
            ov    = evaluate(v); fes += 1; fv = tof(ov)
            if fv > fit[i]: pop[i],obj[i],fit[i],trials[i] = v,ov,fv,0
            else:           trials[i] += 1
            if fes >= max_fes: break
        gb         = int(np.argmax(fit))
        gb_updated = (gb != prev_gb)
        rec()
        if fes >= max_fes: break

        # Scout bee phase — dual-elite strategy (Eq.13)
        el = elites()
        for i in range(sn):
            if trials[i] > config.limit:
                xe1, xe2 = pop[rng.choice(el, 2, replace=False)]
                a, b, c_ = rng.dirichlet([1, 1, 1])
                jrand    = rng.integers(0, dim)
                mask     = (rng.random(dim) <= config.cr) | (np.arange(dim) == jrand)
                v        = np.where(mask, a*pop[i]+b*xe1+c_*xe2, pop[i])
                v        = np.clip(v, lo, hi)
                ov       = evaluate(v); fes += 1; fv = tof(ov)
                pop[i],obj[i],fit[i],trials[i] = v,ov,fv,0
                if fes >= max_fes: break
        gb = int(np.argmax(fit)); rec()

    final = max(0.0, obj[gb] - f_opt)
    while len(curve) < len(log_fes): curve.append(final)
    return {'best_error':final,'best_obj':float(obj[gb]),'curve':curve,'fes_used':fes}

print('ABC-ANT ready.')

ALGORITHMS = {'ABC-ANT': abc_ant}
print('Algorithm: ABC-ANT')


# ## 8 · Experiment runner (sequential + checkpoint)

def run_single(algo_name, func_idx, run_idx, config):
    seed = GLOBAL_SEED + func_idx*10_000 + run_idx
    evaluate, f_opt = make_func(func_idx)
    t0  = time.perf_counter()
    res = ALGORITHMS[algo_name](evaluate, BOUNDS, f_opt, config, MAX_FES, LOG_FES, seed)
    res['wall_time'] = time.perf_counter() - t0
    res['algo'] = algo_name
    res['func'] = func_idx
    res['run']  = run_idx
    return res


def run_all(config):
    ckpt  = RESULTS_DIR / 'checkpoint.pkl'
    all_r = []
    done  = set()
    if ckpt.exists():
        with open(ckpt,'rb') as fh: all_r = pickle.load(fh)
        done = {(r['algo'],r['func'],r['run']) for r in all_r}
        print(f'Resuming: {len(done)} tasks already done.')
    tasks = [
        (a,fi,ri)
        for a  in ALGORITHMS
        for fi in range(1, N_FUNCS+1)
        for ri in range(N_RUNS)
        if (a,fi,ri) not in done
    ]
    print(f'Tasks remaining: {len(tasks)}')
    if not tasks: return all_r
    for a, fi, ri in tqdm(tasks, desc='Running'):
        res = run_single(a, fi, ri, config)
        all_r.append(res)
    with open(ckpt,'wb') as fh: pickle.dump(all_r, fh)
    print(f'Done. {len(all_r)} results saved.')
    return all_r


# ## 9 · Run experiments

config      = ABCANTConfig(sn=N_BEES, limit=100, cr=0.5)
raw_results = run_all(config)
print(f'Total results: {len(raw_results)}')


# ## 11 · Aggregate results

curve_keys = [f'curve_{i}' for i in range(len(LOG_FES))]

records = []
for r in raw_results:
    row = {'Algorithm':r['algo'],'Function':r['func'],
           'Run':r['run'],'Error':r['best_error'],
           'WallTime':r.get('wall_time', float('nan'))}
    for i,k in enumerate(curve_keys):
        row[k] = r['curve'][i] if i < len(r['curve']) else r['best_error']
    records.append(row)

df = pd.DataFrame(records)
df.to_csv(RESULTS_DIR/'raw_results.csv', index=False)

summary = (df.groupby(['Algorithm','Function'])['Error']
             .agg(['mean','std','min','median','max'])
             .reset_index())
summary.columns = ['Algorithm','Function','Mean','Std','Min','Median','Max']
summary.to_csv(RESULTS_DIR/'summary_table.csv', index=False)

algos = sorted(df['Algorithm'].unique())
funcs = sorted(df['Function'].unique())
print('Algorithms:', algos)
print('Functions: ', funcs)
print(summary.head(8).to_string(index=False))


# ## 12 · Statistical tests

REF = 'ABC-ANT'

# Wilcoxon rank-sum (Mann-Whitney U)
wrows = []
for fi in funcs:
    ref_e = df[(df.Algorithm==REF)&(df.Function==fi)]['Error'].values
    for algo in algos:
        if algo == REF: continue
        cmp_e = df[(df.Algorithm==algo)&(df.Function==fi)]['Error'].values
        n = min(len(ref_e), len(cmp_e))
        try:
            _, p = mannwhitneyu(ref_e[:n], cmp_e[:n], alternative='two-sided')
        except Exception:
            p = 1.0
        if p < ALPHA:
            dec = '+' if ref_e.mean() < cmp_e.mean() else '-'
        else:
            dec = '='
        wrows.append({'Function':fi,'Competitor':algo,'p':p,'Decision':dec})

wdf = pd.DataFrame(wrows)
wdf.to_csv(RESULTS_DIR/'wilcoxon.csv', index=False)

# AJOUT DE LA VÉRIFICATION : On s'assure que le DataFrame n'est pas vide
if not wdf.empty:
    wdl = wdf.groupby('Competitor')['Decision'].value_counts().unstack(fill_value=0)
    print('W/D/L  (+ = ABC-ANT better):')
    print(wdl.to_string())
else:
    print('W/D/L  (+ = ABC-ANT better):')
    print("Aucun résultat à afficher (aucune comparaison effectuée).")


# Friedman test + average ranks
rank_mat = np.column_stack([
    df[df.Algorithm==a].groupby('Function')['Error'].median().values
    for a in algos
])

# CORRECTION : On vérifie qu'il y a au moins 3 algorithmes avant de lancer le test
if len(algos) >= 3:
    f_stat, f_p = friedmanchisquare(*[rank_mat[:,i] for i in range(len(algos))])
    print(f'Friedman chi2={f_stat:.3f}  p={f_p:.2e}')
else:
    print(f"Friedman test impossible : au moins 3 algorithmes requis, récupéré {len(algos)}.")

avg_ranks = np.mean(
    np.argsort(np.argsort(rank_mat, axis=1), axis=1)+1, axis=0
)
rank_df = (pd.DataFrame({'Algorithm':algos,'Avg_Rank':avg_ranks})
             .sort_values('Avg_Rank').reset_index(drop=True))
rank_df.to_csv(RESULTS_DIR/'friedman_ranks.csv', index=False)
print(rank_df.to_string(index=False))


# Nemenyi post-hoc
try:
    ph = sp.posthoc_nemenyi_friedman(rank_mat)
    ph.columns = algos; ph.index = algos
    ph.to_csv(RESULTS_DIR/'posthoc_nemenyi.csv')
    print('Nemenyi p-values for ABC-ANT row:')
    print(ph.loc[REF].round(4).to_string())
except Exception as e:
    print('Post-hoc skipped:', e)


# ## 13 · Result table + LaTeX export

def sci(v):
    if v == 0 or (isinstance(v,float) and (v!=v)): return '0.00E+00'
    import math
    e = int(math.floor(math.log10(abs(v))))
    return f'{v/10**e:.2f}E{e:+03d}'

rows = []
for fi in funcs:
    row = {'F': f'F{fi:02d}'}
    for algo in algos:
        sub = df[(df.Algorithm==algo)&(df.Function==fi)]['Error']
        m, s = sub.mean(), sub.std()
        if algo != REF:
            dec = wdf[(wdf.Function==fi)&(wdf.Competitor==algo)]['Decision'].values
            mk  = dec[0] if len(dec) else '?'
        else: mk = ''
        row[algo] = f'{sci(m)}+-{sci(s)}{mk}'
    rows.append(row)

rtab = pd.DataFrame(rows).set_index('F')
rtab.to_csv(RESULTS_DIR/'result_table.csv')
print(rtab.to_string())


# Export to LaTeX
lines = [
    r'\begin{table}[htbp]',
    r'\centering',
    r'\caption{CEC2022 D=' + str(DIM) + r' --- Mean$\pm$Std. +/=/- vs ABC-ANT ($\alpha$=0.05)}',
    r'\label{tab:cec2022}',
    r'\resizebox{\textwidth}{!}{%',
    r'\begin{tabular}{l' + 'c'*len(rtab.columns) + '}',
    r'\toprule',
    'Func. & ' + ' & '.join(rtab.columns) + r' \\',
    r'\midrule',
]
for idx, row in rtab.iterrows():
    lines.append(str(idx) + ' & ' + ' & '.join(str(v) for v in row) + r' \\')
lines += [r'\bottomrule', r'\end{tabular}', r'}', r'\end{table}']
(RESULTS_DIR/'result_table.tex').write_text('\n'.join(lines))
print('LaTeX table saved.')


# ## 14 · Convergence curves

palette = sns.color_palette('tab10', n_colors=len(algos))
COL = dict(zip(algos, palette))
LS  = dict(zip(algos, ['-','--','-.',':']))
FES_VALS = np.array(LOG_FES)
cc = [c for c in df.columns if c.startswith('curve_')]

ncols = 4; nrows = (N_FUNCS + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.2*nrows))
axes = axes.flatten()

for idx, fi in enumerate(funcs):
    ax = axes[idx]
    for algo in algos:
        sub = df[(df.Algorithm==algo)&(df.Function==fi)]
        mat = sub[cc].values.astype(float)
        mu  = mat.mean(0); sd = mat.std(0)
        ax.semilogy(FES_VALS, mu+1e-300, color=COL[algo],
                    linestyle=LS.get(algo,'-'), label=algo)
        ax.fill_between(FES_VALS,
            np.maximum(mu-sd, 1e-300), mu+sd+1e-300,
            alpha=0.12, color=COL[algo])
    ax.set_title(f'F{fi}')
    ax.set_xlabel('FEs'); ax.set_ylabel('Error')
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f'{x/1e3:.0f}K'))
    if idx == 0: ax.legend(fontsize=7)

for ax in axes[N_FUNCS:]: ax.set_visible(False)
plt.suptitle(f'Convergence Curves - CEC2022 D={DIM}', fontsize=13, y=1.01)
plt.tight_layout()
fig.savefig(RESULTS_DIR/'convergence_curves.pdf')
fig.savefig(RESULTS_DIR/'convergence_curves.png')
plt.show()
print('Convergence curves saved.')


# ## 15 · Box plots

fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.2*nrows))
axes = axes.flatten()
for idx, fi in enumerate(funcs):
    ax  = axes[idx]
    dat = [df[(df.Algorithm==a)&(df.Function==fi)]['Error'].values+1e-300
           for a in algos]
    bp  = ax.boxplot(dat, patch_artist=True, labels=algos, notch=False,
                     flierprops=dict(marker='.', markersize=3, alpha=0.5))
    for patch, a in zip(bp['boxes'], algos):
        patch.set_facecolor(COL[a]); patch.set_alpha(0.7)
    ax.set_yscale('log'); ax.set_title(f'F{fi}')
    ax.tick_params(axis='x', labelrotation=30, labelsize=7)
for ax in axes[N_FUNCS:]: ax.set_visible(False)
plt.suptitle(f'Error Distribution - CEC2022 D={DIM}', fontsize=13, y=1.01)
plt.tight_layout()
fig.savefig(RESULTS_DIR/'boxplots.pdf')
fig.savefig(RESULTS_DIR/'boxplots.png')
plt.show(); print('Box plots saved.')


# ## 16 · Critical-Difference diagram (Demsar 2006)

Q_TAB = {2:1.960,3:2.344,4:2.569,5:2.728,6:2.850,7:2.949,8:3.031,9:3.102,10:3.164}

def cd_diagram(rank_df_in, p_val, n_ds, alpha=0.05):
    k   = len(rank_df_in)
    CD  = Q_TAB.get(k, 2.569) * (k*(k+1)/(6*n_ds))**0.5
    df2 = rank_df_in.sort_values('Avg_Rank').reset_index(drop=True)
    rks = df2['Avg_Rank'].values; nms = df2['Algorithm'].values
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.set_xlim(rks.min()-0.5, rks.max()+0.5)
    ax.set_ylim(-1.5, k+1); ax.axis('off')
    ax.annotate('', xy=(rks.max()+0.3, k+0.2), xytext=(rks.min()-0.3, k+0.2),
                arrowprops=dict(arrowstyle='-', lw=1.5))
    for rv in range(int(rks.min())+1, int(rks.max())+1):
        ax.plot([rv,rv],[k+0.05,k+0.35],'k-',lw=1)
        ax.text(rv,k+0.55,str(rv),ha='center',fontsize=9)
    for i,(nm,rk) in enumerate(zip(nms,rks)):
        ax.plot([rk,rk],[k+0.2,i],'k-',lw=0.7,alpha=0.4)
        ax.plot(rk, i, 'o', color=COL.get(nm,'gray'), ms=9)
        ax.text(rk+0.05, i, f' {nm} ({rk:.2f})', va='center', fontsize=9)
    ax.annotate('',xy=(rks.min()+CD,-0.8),xytext=(rks.min(),-0.8),
                arrowprops=dict(arrowstyle='<->',lw=1.5))
    ax.text(rks.min()+CD/2,-1.2,f'CD={CD:.2f}',ha='center',fontsize=9)
    ax.set_title(f'CD Diagram (Nemenyi alpha={alpha})  |  Friedman p={p_val:.2e}',
                 fontsize=10)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR/'cd_diagram.pdf')
    fig.savefig(RESULTS_DIR/'cd_diagram.png')
    plt.show(); print('CD diagram saved.')

# CORRECTION : On s'assure que f_p existe même si le test n'a pas pu être exécuté
if 'f_p' not in locals() and 'f_p' not in globals():
    f_p = 1.0  

cd_diagram(rank_df, f_p, N_FUNCS)

# ## 17 · FDC landscape case study (Fig.1/2 of the paper)

def fdc_case_study(func_idx, n_iter=100, sn=60):
    rng = np.random.default_rng(GLOBAL_SEED + func_idx)
    lo  = np.full(DIM, -100.); hi = np.full(DIM, 100.)
    evaluate, _ = make_func(func_idx)
    pop = lo + rng.random((sn, DIM)) * (hi - lo)
    obj = np.array([evaluate(pop[i]) for i in range(sn)])
    def tof(o): return 1/(1+o) if o>=0 else 1+abs(o)
    fit = np.array([tof(o) for o in obj])
    hist = [(0, compute_fdc(pop, fit))]
    for it in range(1, n_iter+1):
        for i in range(sn):
            r_ = rng.choice([j for j in range(sn) if j!=i])
            j_ = rng.integers(0, DIM)
            v  = pop[i].copy()
            v[j_] = pop[i,j_]+rng.uniform(-1,1)*(pop[i,j_]-pop[r_,j_])
            v = np.clip(v, lo, hi); ov=evaluate(v); fv=tof(ov)
            if fv>fit[i]: pop[i],obj[i],fit[i]=v,ov,fv
        hist.append((it, compute_fdc(pop, fit)))
    return hist

STUDY = [1, 3]
fig, axes = plt.subplots(1, len(STUDY), figsize=(10, 3.8))
for ax, fi in zip(axes, STUDY):
    hist = fdc_case_study(fi)
    iters, rvals = zip(*hist)
    ax.plot(iters, rvals, 'o-', color='steelblue', ms=4, label='FDC r')
    ax.axhline(0.75,  color='red',   ls='--', lw=1, label='r=0.75 (Random)')
    ax.axhline(0.15,  color='orange',ls='--', lw=1, label='r=0.15 (Ring)')
    ax.fill_between(list(iters), 0.75, 1.05, alpha=0.1, color='red')
    ax.fill_between(list(iters), 0.15, 0.75, alpha=0.1, color='orange')
    ax.fill_between(list(iters),-1.0,  0.15, alpha=0.1, color='green',
                    label='r<0.15 (Cellular)')
    ax.set_xlabel('Iteration'); ax.set_ylabel('r')
    ax.set_ylim(-1, 1.1); ax.set_title(f'F{fi} FDC analysis')
    ax.legend(fontsize=8)
plt.tight_layout()
fig.savefig(RESULTS_DIR/'fdc_analysis.pdf')
fig.savefig(RESULTS_DIR/'fdc_analysis.png')
plt.show(); print('FDC analysis saved.')


# ## 18 · Output file inventory

import math
print('=== Output files ===')
for p in sorted(RESULTS_DIR.iterdir()):
    sz = p.stat().st_size
    unit = 'B'
    for u in ['KB','MB']:
        if sz > 1024: sz /= 1024; unit = u
        else: break
    print(f'  {p.name:<40s}  {sz:6.1f} {unit}')

