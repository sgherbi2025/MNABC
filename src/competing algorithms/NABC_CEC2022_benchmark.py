# 
# # NABC (Best Neighbor-guided ABC) — Benchmark CEC2022 (D=10, D=20)
# 
# **Référence algorithme :** Peng, Deng & Wu (2018), *"Best neighbor-guided artificial bee colony algorithm for continuous optimization problems"*, Soft Computing. https://doi.org/10.1007/s00500-018-3473-6
# 
# **Objectif de ce notebook :**
# - Implémenter NABC fidèlement à l'Algorithme 1 du papier (Eq. 1–5).
# - Implémenter l'ABC de base (baseline) pour reproduire l'ablation NABC vs ABC du papier (Table 2/4 originales), ce qui sert de point de comparaison direct dans votre pipeline AutoABC.
# - Tester sur **CEC2022** (12 fonctions) à **D=10** et **D=20**, avec le protocole standardisé déjà utilisé dans vos autres notebooks (30 abeilles, 30 runs).
# - Produire toutes les sorties nécessaires pour publication Class A : tableaux LaTeX (booktabs), tests de Wilcoxon (signed-rank, par fonction et multi-problème), test de Friedman + rangs moyens, courbes de convergence (PDF), et fichiers `.csv`/`.tex` exportables.
# 
# **Protocole (CEC2022, conforme à votre configuration standard) :**
# ```
# N_BEES        = 30
# N_RUNS        = 30
# MAX_FES       = 200_000 if DIM == 10 else 1_000_000
# N_FUNCS       = 12
# FUNC_IDS      = list(range(1, 13))
# BOUNDS        = [(-100.0, 100.0)] * DIM
# ```
# 
# > Note méthodologique : NABC dans le papier original utilise SN = D, limit = 50, N (taille de voisinage) = 5. Pour que la comparaison avec vos autres algorithmes (MNABC-MAB, APSM-jSO, EABC-AS, etc.) soit strictement équitable, on fixe **SN = N_BEES = 30** (conforme à votre protocole), et on garde **limit = 50, N = 5** (valeurs validées empiriquement dans le papier, Sect. 5.5).
# 


# ============================================================
# 0. Installation / imports
# ============================================================
!pip install "setuptools<70.0.0" opfunu
import sys, subprocess
for pkg in ["opfunu", "scikit-posthocs"]:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", pkg,
                         "--break-system-packages", "-q"])

import os, time, json, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)

print("Imports OK.")



# ============================================================
# 1. Configuration globale
# ============================================================
N_BEES        = 30
N_RUNS        = 30
N_FUNCS       = 12
FUNC_IDS      = list(range(1, 13))

# Paramètres NABC (Peng et al. 2018, Sect. 5.1 / 5.5)
NABC_LIMIT    = 50
NABC_NEIGH    = 5      # taille du voisinage N (meilleure valeur trouvée Table 7)

# Dossier racine de sortie (adapter si besoin)
BASE_ROOT = Path("./NABC_CEC2022_results")
BASE_ROOT.mkdir(parents=True, exist_ok=True)

def get_config(DIM):
    cfg = dict(
        DIM      = DIM,
        N_BEES   = N_BEES,
        N_RUNS   = N_RUNS,
        MAX_FES  = 200_000 if DIM == 10 else 1_000_000,
        N_FUNCS  = N_FUNCS,
        FUNC_IDS = FUNC_IDS,
        BOUNDS   = [(-100.0, 100.0)] * DIM,
        BASE_DIR = BASE_ROOT / f"D{DIM}",
    )
    cfg["MAX_ITER"] = cfg["MAX_FES"] // N_BEES
    cfg["BASE_DIR"].mkdir(parents=True, exist_ok=True)
    return cfg

print("Configuration prête. Exemple D=10 :")
print(get_config(10))



# ============================================================
# 2. Wrapper CEC2022 (opfunu)
# ============================================================
import opfunu

def get_cec2022_problem(func_id, dim):
    """Retourne l'objet problème opfunu CEC2022 pour func_id (1..12) et dim (10/20/30/40)."""
    class_name = f"F{func_id}2022"
    problem_class = getattr(opfunu.cec_based.cec2022, class_name)
    problem = problem_class(ndim=dim)
    return problem

def make_objective(func_id, dim):
    problem = get_cec2022_problem(func_id, dim)
    f_global = problem.f_global  # valeur optimale connue (biais inclus)
    def objective(x):
        return problem.evaluate(x) - f_global
    return objective, f_global, problem.lb, problem.ub

# Sanity check
obj, fg, lb, ub = make_objective(1, 10)
x_test = np.random.uniform(lb, ub)
print("F1 D=10 -> f_global =", fg, "| erreur test (point aléatoire) =", obj(x_test))


# ## 3. Implémentation des algorithmes


# ============================================================
# 3.1 NABC — Algorithme 1 du papier (Eq. 1-5)
# ============================================================
def clip_bounds(x, lb, ub):
    return np.clip(x, lb, ub)

def calc_fitness(errors):
   #"Eq. (2): transforme l'erreur (à minimiser) en fitness (proportionnel, >0).
    fit = np.where(errors >= 0, 1.0 / (1.0 + errors), 1.0 + np.abs(errors))
    return fit

def best_neighbor_guided_search(X, i, fitness, neigh_size, lb, ub, D, rng):
    #Eq. (4): vi = x_nbest + phi * (x_nbest - xi)"""
    SN = X.shape[0]
    candidates = [k for k in range(SN) if k != i]
    neigh_idx = rng.choice(candidates, size=min(neigh_size, len(candidates)), replace=False)
    nbest = neigh_idx[np.argmin([-fitness[k] for k in neigh_idx])]  # fitness max == meilleur (min erreur)
    nbest = neigh_idx[np.argmax(fitness[neigh_idx])]
    phi = rng.uniform(-1, 1, size=D)
    v = X[nbest] + phi * (X[nbest] - X[i])
    return clip_bounds(v, lb, ub)

def global_neighbor_search(X, i, gbest, D, rng):
    #Eq. (5): scout bee phase, GNS au lieu du tirage aléatoire."""
    SN = X.shape[0]
    candidates = [k for k in range(SN) if k != i]
    a, b = rng.choice(candidates, size=2, replace=False)
    r = rng.uniform(0, 1, size=3)
    r = r / r.sum()  # r1+r2+r3 = 1
    v = r[0] * X[i] + r[1] * gbest + r[2] * (X[a] - X[b])
    return v

def run_NABC(objective, D, lb, ub, SN, max_fes, limit=NABC_LIMIT,
             neigh_size=NABC_NEIGH, seed=None, record_curve=False):
    rng = np.random.default_rng(seed)
    lb_arr = np.full(D, lb) if np.isscalar(lb) else np.asarray(lb)
    ub_arr = np.full(D, ub) if np.isscalar(ub) else np.asarray(ub)

    X = rng.uniform(lb_arr, ub_arr, size=(SN, D))
    fX = np.array([objective(x) for x in X])
    nfe = SN
    trial = np.zeros(SN, dtype=int)

    best_idx = np.argmin(fX)
    best_x, best_f = X[best_idx].copy(), fX[best_idx]

    curve_fe, curve_f = [nfe], [best_f]

    while nfe < max_fes:
        fitness = calc_fitness(fX)

        # ---- Employed bee phase (Eq. 4) ----
        for i in range(SN):
            if nfe >= max_fes:
                break
            v = best_neighbor_guided_search(X, i, fitness, neigh_size, lb_arr, ub_arr, D, rng)
            fv = objective(v)
            nfe += 1
            if fv < fX[i]:
                X[i], fX[i] = v, fv
                trial[i] = 0
            else:
                trial[i] += 1

        # ---- Onlooker bee phase (Eq. 4, sélection proba.) ----
        fitness = calc_fitness(fX)
        p = fitness / fitness.sum()
        for _ in range(SN):
            if nfe >= max_fes:
                break
            i = rng.choice(SN, p=p)
            v = best_neighbor_guided_search(X, i, fitness, neigh_size, lb_arr, ub_arr, D, rng)
            fv = objective(v)
            nfe += 1
            if fv < fX[i]:
                X[i], fX[i] = v, fv
                trial[i] = 0
            else:
                trial[i] += 1

        # ---- Scout bee phase (Eq. 5, GNS) ----
        gbest_idx = np.argmin(fX)
        gbest = X[gbest_idx]
        for i in range(SN):
            if nfe >= max_fes:
                break
            if trial[i] > limit:
                v = global_neighbor_search(X, i, gbest, D, rng)
                v = clip_bounds(v, lb_arr, ub_arr)
                fv = objective(v)
                nfe += 1
                X[i], fX[i] = v, fv
                trial[i] = 0

        cur_best_idx = np.argmin(fX)
        if fX[cur_best_idx] < best_f:
            best_f = fX[cur_best_idx]
            best_x = X[cur_best_idx].copy()

        if record_curve:
            curve_fe.append(nfe)
            curve_f.append(best_f)

    result = {"best_x": best_x, "best_f": max(best_f, 0.0), "nfe": nfe}
    if record_curve:
        result["curve_fe"] = np.array(curve_fe)
        result["curve_f"] = np.maximum(np.array(curve_f), 1e-300)
    return result



# ============================================================
# 3.2 ABC de base (baseline, Eq. 1 + scout aléatoire Eq. 3)
#     -> reproduit l'ablation NABC vs ABC du papier original
# ============================================================
def basic_search_eq1(X, i, lb, ub, D, rng):
    SN = X.shape[0]
    k = rng.choice([j for j in range(SN) if j != i])
    j = rng.integers(0, D)
    phi = rng.uniform(-1, 1)
    v = X[i].copy()
    v[j] = X[i, j] + phi * (X[i, j] - X[k, j])
    return clip_bounds(v, lb, ub)

def run_ABC_basic(objective, D, lb, ub, SN, max_fes, limit=NABC_LIMIT, seed=None, record_curve=False):
    rng = np.random.default_rng(seed)
    lb_arr = np.full(D, lb) if np.isscalar(lb) else np.asarray(lb)
    ub_arr = np.full(D, ub) if np.isscalar(ub) else np.asarray(ub)

    X = rng.uniform(lb_arr, ub_arr, size=(SN, D))
    fX = np.array([objective(x) for x in X])
    nfe = SN
    trial = np.zeros(SN, dtype=int)
    best_idx = np.argmin(fX)
    best_f = fX[best_idx]

    curve_fe, curve_f = [nfe], [best_f]

    while nfe < max_fes:
        for i in range(SN):
            if nfe >= max_fes:
                break
            v = basic_search_eq1(X, i, lb_arr, ub_arr, D, rng)
            fv = objective(v)
            nfe += 1
            if fv < fX[i]:
                X[i], fX[i] = v, fv
                trial[i] = 0
            else:
                trial[i] += 1

        fitness = calc_fitness(fX)
        p = fitness / fitness.sum()
        for _ in range(SN):
            if nfe >= max_fes:
                break
            i = rng.choice(SN, p=p)
            v = basic_search_eq1(X, i, lb_arr, ub_arr, D, rng)
            fv = objective(v)
            nfe += 1
            if fv < fX[i]:
                X[i], fX[i] = v, fv
                trial[i] = 0
            else:
                trial[i] += 1

        for i in range(SN):
            if nfe >= max_fes:
                break
            if trial[i] > limit:
                v = rng.uniform(lb_arr, ub_arr)
                fv = objective(v)
                nfe += 1
                X[i], fX[i] = v, fv
                trial[i] = 0

        cur_best = fX.min()
        if cur_best < best_f:
            best_f = cur_best
        if record_curve:
            curve_fe.append(nfe)
            curve_f.append(best_f)

    result = {"best_f": max(best_f, 0.0), "nfe": nfe}
    if record_curve:
        result["curve_fe"] = np.array(curve_fe)
        result["curve_f"] = np.maximum(np.array(curve_f), 1e-300)
    return result


# ## 4. Boucle d'expérimentation (sans checkpoint/resume)


# ============================================================
# 4. Runner simple (exécute tout d'un coup, pas de reprise)
# ============================================================
ALGOS = {
    "ABC":  run_ABC_basic,
    "NABC": run_NABC,
}

def results_path(cfg, algo_name):
    return cfg["BASE_DIR"] / f"results_{algo_name}_D{cfg['DIM']}.csv"

def run_experiment(cfg, algo_name, verbose=True):
    """Exécute algo_name sur les 12 fonctions CEC2022, N_RUNS runs,
    sans checkpoint ni reprise : tout est relancé depuis zéro à chaque appel."""
    run_fn = ALGOS[algo_name]
    D = cfg["DIM"]
    state = {}  # {func_id: {run_idx: best_f, ...}}

    for func_id in cfg["FUNC_IDS"]:
        state.setdefault(func_id, {})
        objective, f_global, lb, ub = make_objective(func_id, D)

        for run_idx in range(cfg["N_RUNS"]):
            seed = 1000 * func_id + run_idx
            res = run_fn(objective, D, lb, ub, cfg["N_BEES"], cfg["MAX_FES"], seed=seed)
            state[func_id][run_idx] = res["best_f"]

            if verbose:
                print(f"[{algo_name} D={D}] F{func_id} run {run_idx+1}/{cfg['N_RUNS']} "
                      f"-> err={res['best_f']:.4e}", end="\r")

        if verbose:
            print(f"\n[{algo_name} D={D}] F{func_id} terminé "
                  f"({len(state[func_id])}/{cfg['N_RUNS']} runs).")

    # Export final en CSV (lignes = func, colonnes = runs)
    df = pd.DataFrame.from_dict(state, orient="index").sort_index()
    df.index.name = "func_id"
    df.to_csv(results_path(cfg, algo_name))
    return df


# ## 5. Exécution — D = 10


cfg10 = get_config(10)
print("Lancement NABC, D=10 ...")
df_nabc_10 = run_experiment(cfg10, "NABC")



print("Lancement ABC (baseline), D=10 ...")
df_abc_10 = run_experiment(cfg10, "ABC")


# ## 6. Exécution — D = 20


cfg20 = get_config(20)
print("Lancement NABC, D=20 ...")
df_nabc_20 = run_experiment(cfg20, "NABC")



print("Lancement ABC (baseline), D=20 ...")
df_abc_20 = run_experiment(cfg20, "ABC")


# ## 7. Statistiques descriptives (Mean ± Std) et tests de Wilcoxon par fonction


def wilcoxon_symbol(nabc_runs, comp_runs, alpha=0.05):
   #"Retourne '-', '+', ou '≈' du point de vue de NABC, comme dans le papier
    #(- : le comparateur est moins bon que NABC ; + : meilleur ; ≈ : pas de diff. significative)."""
    nabc_runs = np.asarray(nabc_runs)
    comp_runs = np.asarray(comp_runs)
    if np.allclose(nabc_runs, comp_runs):
        return "≈"
    try:
        stat, p = stats.wilcoxon(comp_runs, nabc_runs, alternative="two-sided")
    except ValueError:
        return "≈"
    if p >= alpha:
        return "≈"
    return "-" if comp_runs.mean() > nabc_runs.mean() else "+"

def build_summary_table(df_nabc, df_comp, comp_name="ABC"):
    rows = []
    for func_id in df_nabc.index:
        nabc_runs = df_nabc.loc[func_id].dropna().values
        comp_runs = df_comp.loc[func_id].dropna().values
        sym = wilcoxon_symbol(nabc_runs, comp_runs)
        rows.append({
            "func_id": func_id,
            f"{comp_name}_mean": comp_runs.mean(), f"{comp_name}_std": comp_runs.std(),
            "NABC_mean": nabc_runs.mean(), "NABC_std": nabc_runs.std(),
            "wilcoxon_vs_NABC": sym,
        })
    return pd.DataFrame(rows).set_index("func_id")

summary_10 = build_summary_table(df_nabc_10, df_abc_10, "ABC")
summary_20 = build_summary_table(df_nabc_20, df_abc_20, "ABC")

print("=== D=10 ===")
display(summary_10)
print("=== D=20 ===")
display(summary_20)


# ## 8. Test de Friedman et rangs moyens (NABC vs ABC, par dimension)


def friedman_ranking(df_nabc, df_abc):
    """Calcule les rangs moyens (Friedman) entre NABC et ABC, par fonction
    (rang basé sur la moyenne d'erreur de chaque algo sur chaque fonction)."""
    means = pd.DataFrame({
        "ABC":  df_abc.mean(axis=1),
        "NABC": df_nabc.mean(axis=1),
    })
    ranks = means.rank(axis=1, method="average")
    avg_ranks = ranks.mean(axis=0).sort_values()
    stat, p = stats.friedmanchisquare(*[means[c].values for c in means.columns])
    return avg_ranks, stat, p

ranks_10, fstat_10, fp_10 = friedman_ranking(df_nabc_10, df_abc_10)
ranks_20, fstat_20, fp_20 = friedman_ranking(df_nabc_20, df_abc_20)

print("Rangs moyens Friedman — D=10 :"); print(ranks_10)
print(f"Friedman stat={fstat_10:.4f}, p={fp_10:.4g}\n")
print("Rangs moyens Friedman — D=20 :"); print(ranks_20)
print(f"Friedman stat={fstat_20:.4f}, p={fp_20:.4g}")


# ## 9. Test de Wilcoxon multi-problème (global NABC vs ABC)


def multi_problem_wilcoxon(df_nabc, df_abc):
    nabc_means = df_nabc.mean(axis=1)
    abc_means  = df_abc.mean(axis=1)
    stat, p = stats.wilcoxon(abc_means, nabc_means, alternative="two-sided")
    r_plus  = np.sum(np.where(nabc_means.values < abc_means.values,
                               np.abs(nabc_means.values - abc_means.values), 0))
    r_minus = np.sum(np.where(nabc_means.values >= abc_means.values,
                               np.abs(nabc_means.values - abc_means.values), 0))
    return {"R+": r_plus, "R-": r_minus, "p_value": p, "significant_0.05": p < 0.05}

mw_10 = multi_problem_wilcoxon(df_nabc_10, df_abc_10)
mw_20 = multi_problem_wilcoxon(df_nabc_20, df_abc_20)
print("Wilcoxon multi-problème D=10 :", mw_10)
print("Wilcoxon multi-problème D=20 :", mw_20)


# ## 10. Courbes de convergence (NABC vs ABC) — export PDF


def convergence_curves(cfg, func_ids_to_plot, n_seeds=5):
    #Génère et sauvegarde les courbes de convergence moyennes NABC vs ABC
    #pour un sous-ensemble représentatif de fonctions."""
    D = cfg["DIM"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for ax, func_id in zip(axes, func_ids_to_plot):
        objective, f_global, lb, ub = make_objective(func_id, D)
        curves = {"ABC": [], "NABC": []}
        for algo_name, run_fn in ALGOS.items():
            for seed in range(n_seeds):
                res = run_fn(objective, D, lb, ub, cfg["N_BEES"], cfg["MAX_FES"],
                              seed=2000 + seed, record_curve=True)
                curves[algo_name].append((res["curve_fe"], res["curve_f"]))

        for algo_name, runs in curves.items():
            fe_ref = runs[0][0]
            f_stack = np.array([np.interp(fe_ref, fe, f) for fe, f in runs])
            ax.plot(fe_ref, f_stack.mean(axis=0), label=algo_name)

        ax.set_yscale("log")
        ax.set_title(f"F{func_id} (D={D})")
        ax.set_xlabel("FEs")
        ax.set_ylabel("Erreur moyenne")
        ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = cfg["BASE_DIR"] / f"convergence_NABC_vs_ABC_D{D}.pdf"
    plt.savefig(out_path)
    plt.show()
    print("Sauvegardé :", out_path)

convergence_curves(cfg10, [1, 2, 5, 7, 10, 12])
convergence_curves(cfg20, [1, 2, 5, 7, 10, 12])


# ## 11. Tableaux LaTeX (booktabs) — prêts pour l'article


def to_latex_booktabs(summary_df, dim, comp_name="ABC", caption_suffix=""):
    """Génère un tableau LaTeX style booktabs (Mean±Std + symbole Wilcoxon),
    cohérent avec le format déjà utilisé dans autoabc_main.tex."""
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{Résultats NABC vs {comp_name} sur CEC2022 (D={dim}){caption_suffix}}}")
    lines.append(rf"\label{{tab:nabc_vs_{comp_name.lower()}_D{dim}}}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{c c c c}")
    lines.append(r"\toprule")
    lines.append(rf"F & \makecell{{{comp_name}\\Mean $\pm$ Std}} & \makecell{{NABC\\Mean $\pm$ Std}} & Wilcoxon \\")
    lines.append(r"\midrule")
    for func_id, row in summary_df.iterrows():
        lines.append(
            f"F{func_id} & "
            f"{row[f'{comp_name}_mean']:.2E} $\\pm$ {row[f'{comp_name}_std']:.2E} & "
            f"{row['NABC_mean']:.2E} $\\pm$ {row['NABC_std']:.2E} & "
            f"{row['wilcoxon_vs_NABC']} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")
    return "\n".join(lines)

tex_10 = to_latex_booktabs(summary_10, 10)
tex_20 = to_latex_booktabs(summary_20, 20)

(cfg10["BASE_DIR"] / "table_NABC_vs_ABC_D10.tex").write_text(tex_10, encoding="utf-8")
(cfg20["BASE_DIR"] / "table_NABC_vs_ABC_D20.tex").write_text(tex_20, encoding="utf-8")

print(tex_10)


# ## 12. Export complet (CSV + résumé JSON) pour intégration dans le pipeline AutoABC


summary_10.to_csv(cfg10["BASE_DIR"] / "summary_NABC_vs_ABC_D10.csv")
summary_20.to_csv(cfg20["BASE_DIR"] / "summary_NABC_vs_ABC_D20.csv")

final_report = {
    "D10": {
        "friedman_avg_ranks": ranks_10.to_dict(),
        "friedman_stat": fstat_10, "friedman_p": fp_10,
        "multi_problem_wilcoxon": mw_10,
    },
    "D20": {
        "friedman_avg_ranks": ranks_20.to_dict(),
        "friedman_stat": fstat_20, "friedman_p": fp_20,
        "multi_problem_wilcoxon": mw_20,
    },
    "generated_at": datetime.now().isoformat(),
    "protocol": {
        "N_BEES": N_BEES, "N_RUNS": N_RUNS, "N_FUNCS": N_FUNCS,
        "MAX_FES_D10": 200_000, "MAX_FES_D20": 1_000_000,
        "NABC_limit": NABC_LIMIT, "NABC_neighbor_size": NABC_NEIGH,
    },
}

with open(BASE_ROOT / "NABC_CEC2022_final_report.json", "w", encoding="utf-8") as f:
    json.dump(final_report, f, indent=2, default=str)

print("Rapport final sauvegardé :", BASE_ROOT / "NABC_CEC2022_final_report.json")
print(json.dumps(final_report, indent=2, default=str))


# 
# ## 13. Notes pour la rédaction de l'article
# 
# - **Protocole conforme CEC2022** : 30 abeilles, 30 runs indépendants, MaxFES = 200 000 (D=10) / 1 000 000 (D=20), bornes [-100, 100]^D, 12 fonctions.
# - Le symbole Wilcoxon (`-`, `+`, `≈`) suit exactement la convention du papier Peng et al. (2018) : du point de vue de **NABC**.
# - Pour intégrer NABC comme **comparateur additionnel** dans votre tableau AutoABC (CEC2022 D=10/D=20), utilisez les colonnes `NABC_mean`/`NABC_std` de `summary_10`/`summary_20`, au même format que vos autres algorithmes (`results_AutoABC_D10.tex`, etc.).
# - Si vous voulez insérer une colonne NABC dans `autoabc_main.tex`, le format `\resizebox` + `\makecell` + `booktabs` généré ici est directement compatible avec vos tableaux existants.
# - ⚠️ Ce script ne gère pas de checkpoint/reprise : une interruption oblige à relancer `run_experiment(...)` depuis le début.
# 

