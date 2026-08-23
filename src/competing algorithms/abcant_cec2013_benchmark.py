import numpy as np

import os, time
import pandas as pd
from opfunu.cec_based import cec2013
import matplotlib.pyplot as plt

RESULTS_DIR = "abcant_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

class ABCANT:
    def __init__(self, func, ndim, lb, ub, SN=50, limit=100, CR=0.5, max_nfe=None, rng=None):
        self.func = func
        self.D = ndim
        self.lb = np.asarray(lb, dtype=float)
        self.ub = np.asarray(ub, dtype=float)
        self.SN = SN
        self.limit = limit
        self.CR = CR
        self.max_nfe = max_nfe if max_nfe is not None else 10000 * ndim
        self.rng = rng if rng is not None else np.random.default_rng()

        self.n_random = max(2, int(round(0.3 * SN)))
        self.n_ring_radius = max(1, int(round(0.1 * SN)))
        self.n_cellular = 4
        self.elite_size = max(2, int(round(0.1 * SN)))

        self.nfe = 0
        self.history = []

    def _evaluate(self, x):
        x = np.clip(x, self.lb, self.ub)
        val = self.func.evaluate(x)
        self.nfe += 1
        return val

    def _init_population(self):
        X = self.lb + self.rng.random((self.SN, self.D)) * (self.ub - self.lb)
        f = np.empty(self.SN)
        for i in range(self.SN):
            f[i] = self._evaluate(X[i])
        return X, f

    def _elite_indices(self, f):
        order = np.argsort(f)
        return order[:self.elite_size]

    def _compute_fdc(self, X, f, gb_idx):
        dists = np.linalg.norm(X - X[gb_idx], axis=1)
        fits = f
        if np.std(dists) < 1e-300 or np.std(fits) < 1e-300:
            return 1.0
        r = np.corrcoef(fits, dists)[0, 1]
        if np.isnan(r):
            r = 1.0
        return r

    def _select_topology(self, r):
        if r > 0.75:
            return "random"
        elif r < 0.15:
            return "cellular"
        else:
            return "ring"

    def _neighbors(self, i, topology):
        SN = self.SN
        if topology == "random":
            pool = [k for k in range(SN) if k != i]
            size = min(self.n_random, len(pool))
            return self.rng.choice(pool, size=size, replace=False)
        elif topology == "ring":
            r = self.n_ring_radius
            idx = [(i + off) % SN for off in range(-r, r + 1) if off != 0]
            return np.array(idx)
        else:
            idx = [(i - 2) % SN, (i - 1) % SN, (i + 1) % SN, (i + 2) % SN]
            idx = list(dict.fromkeys(idx))
            idx = [k for k in idx if k != i]
            return np.array(idx)

    def _repair(self, v, elite_idx, X):
        out = (v < self.lb) | (v > self.ub)
        if np.any(out):
            e = X[self.rng.choice(elite_idx)]
            v = v.copy()
            v[out] = e[out]
        return np.clip(v, self.lb, self.ub)

    def optimize(self, history_every=None):
        X, f = self._init_population()
        trial = np.zeros(self.SN, dtype=int)
        gb_idx = int(np.argmin(f))
        gb_val = f[gb_idx]
        elite_idx = self._elite_indices(f)

        trigger_fdc = True
        topology = "random"

        self.history.append((self.nfe, gb_val))
        last_rec = self.nfe

        while self.nfe < self.max_nfe:
            gb_updated = False

            if trigger_fdc:
                r = self._compute_fdc(X, f, gb_idx)
                topology = self._select_topology(r)

            for i in range(self.SN):
                if self.nfe >= self.max_nfe:
                    break
                nb = self._neighbors(i, topology)
                nb_best = nb[np.argmin(f[nb])]
                cand_pool = [k for k in range(self.SN) if k not in (i, nb_best)]
                r_idx = self.rng.choice(cand_pool)

                j = self.rng.integers(0, self.D)
                phi = self.rng.uniform(-1, 1)
                w = self.rng.uniform(0, 1.5)

                V = X[i].copy()
                V[j] = (X[nb_best, j] + phi * (X[nb_best, j] - X[r_idx, j])
                        + w * (X[gb_idx, j] - X[nb_best, j]))
                V = self._repair(V, elite_idx, X)

                fv = self._evaluate(V)
                if fv < f[i]:
                    X[i] = V
                    f[i] = fv
                    trial[i] = 0
                else:
                    trial[i] += 1

                if f[i] < gb_val:
                    gb_val = f[i]
                    gb_idx = i
                    gb_updated = True

            for i in range(self.SN):
                if self.nfe >= self.max_nfe:
                    break
                nb = self._neighbors(i, topology)
                nb_best = nb[np.argmin(f[nb])]
                cand_pool = [k for k in range(self.SN) if k != nb_best]
                r_idx = self.rng.choice(cand_pool)

                j = self.rng.integers(0, self.D)
                phi = self.rng.uniform(-1, 1)

                V = X[nb_best].copy()
                V[j] = X[nb_best, j] + phi * (X[nb_best, j] - X[r_idx, j])
                V = self._repair(V, elite_idx, X)

                fv = self._evaluate(V)
                if fv < f[nb_best]:
                    X[nb_best] = V
                    f[nb_best] = fv
                    trial[nb_best] = 0
                else:
                    trial[nb_best] += 1

                if f[nb_best] < gb_val:
                    gb_val = f[nb_best]
                    gb_idx = nb_best
                    gb_updated = True

            elite_idx = self._elite_indices(f)
            for i in range(self.SN):
                if self.nfe >= self.max_nfe:
                    break
                if trial[i] > self.limit:
                    e_pool = [k for k in elite_idx if k != i]
                    if len(e_pool) < 2:
                        e_pool = list(elite_idx)
                    e1, e2 = self.rng.choice(e_pool, size=2, replace=False)

                    abc_w = self.rng.random(3)
                    abc_w = abc_w / abc_w.sum()
                    a, b, c = abc_w

                    jrand = self.rng.integers(0, self.D)
                    mask = (self.rng.random(self.D) <= self.CR)
                    mask[jrand] = True

                    V = X[i].copy()
                    V[mask] = a * X[i, mask] + b * X[e1, mask] + c * X[e2, mask]
                    V = np.clip(V, self.lb, self.ub)

                    fv = self._evaluate(V)
                    X[i] = V
                    f[i] = fv
                    trial[i] = 0

                    if f[i] < gb_val:
                        gb_val = f[i]
                        gb_idx = i
                        gb_updated = True

            elite_idx = self._elite_indices(f)
            trigger_fdc = not gb_updated

            if history_every is not None and self.nfe - last_rec >= history_every:
                self.history.append((self.nfe, gb_val))
                last_rec = self.nfe

        self.history.append((self.nfe, gb_val))
        return gb_val, X[gb_idx].copy()


def build_cec2013_suite(ndim):
    suite = {}
    for fid in range(1, 29):
        cls = getattr(cec2013, f"F{fid}2013")
        suite[fid] = cls(ndim=ndim)
    return suite


SN = 50
LIMIT = 100
CR = 0.5
N_RUNS = 30
DIMENSIONS = [30, 50]
NFE_PER_D = 5000

def run_experiment(D):
    store = {fid: {"errors": [], "histories": []} for fid in range(1, 29)}
    suite = build_cec2013_suite(D)
    max_nfe = NFE_PER_D * D

    for fid in range(1, 29):
        func = suite[fid]
        print(f"[D={D}] F{fid:02d}")
        for run_id in range(N_RUNS):
            seed = 100000 * D + 1000 * fid + run_id
            rng = np.random.default_rng(seed)
            opt = ABCANT(func, ndim=D, lb=func.lb, ub=func.ub,
                         SN=SN, limit=LIMIT, CR=CR, max_nfe=max_nfe, rng=rng)
            t0 = time.time()
            best_f, best_x = opt.optimize(history_every=max(1, max_nfe // 50))
            elapsed = time.time() - t0
            error = best_f - func.f_global
            store[fid]["errors"].append(error)
            store[fid]["histories"].append(opt.history)
            print(f"    run {run_id+1:2d}/{N_RUNS}  error={error!r}  nfe={opt.nfe}  time={elapsed:.1f}s")

    return store


store_D30 = run_experiment(D=30)


store_D50 = run_experiment(D=50)


def summarize(store, D):
    rows = []
    for fid in range(1, 29):
        errs = np.array(store[fid]["errors"], dtype=float)
        rows.append({
            "D": D,
            "Func": f"F{fid:02d}",
            "Best": np.min(errs),
            "Worst": np.max(errs),
            "Median": np.median(errs),
            "Mean": np.mean(errs),
            "Std": np.std(errs, ddof=1) if len(errs) > 1 else 0.0,
            "N_runs": len(errs),
        })
    return pd.DataFrame(rows)

summary_D30 = summarize(store_D30, 30)
summary_D50 = summarize(store_D50, 50)
summary_all = pd.concat([summary_D30, summary_D50], ignore_index=True)

summary_D30.to_csv(os.path.join(RESULTS_DIR, "abcant_cec2013_D30_summary.csv"), index=False)
summary_D50.to_csv(os.path.join(RESULTS_DIR, "abcant_cec2013_D50_summary.csv"), index=False)
summary_all.to_csv(os.path.join(RESULTS_DIR, "abcant_cec2013_summary_all.csv"), index=False)

print(summary_D30)
print(summary_D50)


def to_latex_booktabs(df, D, caption, label):
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + caption + "}")
    lines.append(r"\label{" + label + "}")
    lines.append(r"\begin{tabular}{lrrrrr}")
    lines.append(r"\toprule")
    lines.append("Func. & Best & Worst & Median & Mean & Std \\\\")
    lines.append(r"\midrule")
    sub = df[df["D"] == D]
    for _, row in sub.iterrows():
        lines.append(
            f"{row['Func']} & {row['Best']!r} & {row['Worst']!r} & "
            f"{row['Median']!r} & {row['Mean']!r} & {row['Std']!r} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)

latex_D30 = to_latex_booktabs(
    summary_all, 30,
    "ABC-ANT results on the CEC2013 test suite, $D=30$ (SN=50, limit=100, $MAX\\_NFE=5000\\times D$, 30 runs).",
    "tab:abcant_cec2013_D30",
)
latex_D50 = to_latex_booktabs(
    summary_all, 50,
    "ABC-ANT results on the CEC2013 test suite, $D=50$ (SN=50, limit=100, $MAX\\_NFE=5000\\times D$, 30 runs).",
    "tab:abcant_cec2013_D50",
)

with open(os.path.join(RESULTS_DIR, "abcant_cec2013_D30_table.tex"), "w") as fh:
    fh.write(latex_D30)
with open(os.path.join(RESULTS_DIR, "abcant_cec2013_D50_table.tex"), "w") as fh:
    fh.write(latex_D50)

print(latex_D30)


def plot_convergence(store, D, fids, fname):
    fig, ax = plt.subplots(figsize=(7, 5))
    for fid in fids:
        histories = store[fid]["histories"]
        min_len = min(len(h) for h in histories)
        nfe_grid = [histories[0][k][0] for k in range(min_len)]
        vals = np.array([[h[k][1] for k in range(min_len)] for h in histories])
        func = build_cec2013_suite(D)[fid]
        errs = vals - func.f_global
        errs = np.clip(errs, 1e-300, None)
        mean_curve = errs.mean(axis=0)
        ax.plot(nfe_grid, mean_curve, label=f"F{fid:02d}")
    ax.set_yscale("log")
    ax.set_xlabel("Function evaluations (NFE)")
    ax.set_ylabel("Mean best error (log scale)")
    ax.set_title(f"ABC-ANT convergence - CEC2013, D={D}")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, fname), dpi=200)
    plt.show()

representative = [1, 6, 13, 24]
plot_convergence(store_D30, 30, representative, "convergence_D30.pdf")
plot_convergence(store_D50, 50, representative, "convergence_D50.pdf")

