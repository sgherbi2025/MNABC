import os
import numpy as np
import pandas as pd
from opfunu.cec_based import cec2013

SN = 50
LIMIT = 100
N_RUNS = 30
DIMENSIONS = [30, 50]
FES_FACTOR = 5000

R_RATIO = 0.1
CR_E = 0.2
CR_O = 0.1

OUTPUT_DIR = "abc_mng_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CEC2013_CLASSES = {
    1: cec2013.F12013, 2: cec2013.F22013, 3: cec2013.F32013, 4: cec2013.F42013,
    5: cec2013.F52013, 6: cec2013.F62013, 7: cec2013.F72013, 8: cec2013.F82013,
    9: cec2013.F92013, 10: cec2013.F102013, 11: cec2013.F112013, 12: cec2013.F122013,
    13: cec2013.F132013, 14: cec2013.F142013, 15: cec2013.F152013, 16: cec2013.F162013,
    17: cec2013.F172013, 18: cec2013.F182013, 19: cec2013.F192013, 20: cec2013.F202013,
    21: cec2013.F212013, 22: cec2013.F222013, 23: cec2013.F232013, 24: cec2013.F242013,
    25: cec2013.F252013, 26: cec2013.F262013, 27: cec2013.F272013, 28: cec2013.F282013,
}


class ABCMNG:

    def __init__(self, obj_func, lb, ub, dim, optimum,
                 sn=SN, limit=LIMIT, max_fes=None,
                 cr_e=CR_E, cr_o=CR_O, r_ratio=R_RATIO, rng=None):
        self.f = obj_func
        self.lb = np.asarray(lb, dtype=float)
        self.ub = np.asarray(ub, dtype=float)
        self.D = dim
        self.optimum = optimum
        self.SN = sn
        self.limit = limit
        self.max_fes = max_fes if max_fes is not None else FES_FACTOR * dim
        self.CR_e = cr_e
        self.CR_o = cr_o
        self.R = max(1, int(round(r_ratio * sn)))
        self.rng = rng if rng is not None else np.random.default_rng()

        self.fes = 0
        self.trial = np.zeros(sn, dtype=int)

    def _init_population(self):
        X = self.lb + self.rng.random((self.SN, self.D)) * (self.ub - self.lb)
        fit = np.array([self._evaluate(X[i]) for i in range(self.SN)])
        return X, fit

    def _evaluate(self, x):
        x = np.clip(x, self.lb, self.ub)
        val = self.f(x)
        self.fes += 1
        return val

    def _bound_repair(self, v):
        below = v < self.lb
        above = v > self.ub
        if np.any(below) or np.any(above):
            rnd = self.lb + self.rng.random(self.D) * (self.ub - self.lb)
            v = np.where(below | above, rnd, v)
        return v

    def _ring_neighbors_idx(self, i):
        idx = [(i + k) % self.SN for k in range(-self.R, self.R + 1) if k != 0]
        return idx

    def _identify_three_neighbors(self, i, X, fit):
        neigh_idx = self._ring_neighbors_idx(i)
        neigh_idx = np.array(neigh_idx)

        best_local = neigh_idx[np.argmin(fit[neigh_idx])]

        dists = np.linalg.norm(X[neigh_idx] - X[i], axis=1)
        nearest_local = neigh_idx[np.argmin(dists)]
        farthest_local = neigh_idx[np.argmax(dists)]

        return best_local, nearest_local, farthest_local

    def _solution_search_eq(self, base_idx, X, CR, exclude_idx):
        candidates = [k for k in range(self.SN) if k != base_idx]
        r = candidates[self.rng.integers(len(candidates))]

        x_star = X[base_idx].copy()
        x_i = X[exclude_idx]

        phi = self.rng.uniform(-1, 1, self.D)
        j_rand = self.rng.integers(self.D)

        v = x_i.copy()
        mask = (self.rng.random(self.D) <= CR)
        mask[j_rand] = True
        v[mask] = x_star[mask] + phi[mask] * (x_star[mask] - X[r][mask])

        v = self._bound_repair(v)
        return v

    @staticmethod
    def _cosine_similarity(v, x):
        nv = np.linalg.norm(v)
        nx = np.linalg.norm(x)
        if nv == 0 or nx == 0:
            return -np.inf
        return float(np.dot(v, x) / (nv * nx))

    def _employed_bee_phase(self, X, fit, gbest_val_history=None):
        for i in range(self.SN):
            if self.fes >= self.max_fes:
                break

            b_idx, n_idx, f_idx = self._identify_three_neighbors(i, X, fit)

            v_b = self._solution_search_eq(b_idx, X, self.CR_e, i)
            v_n = self._solution_search_eq(n_idx, X, self.CR_e, i)
            v_f = self._solution_search_eq(f_idx, X, self.CR_e, i)

            offsprings = [v_b, v_n, v_f]
            cosines = [self._cosine_similarity(v, X[i]) for v in offsprings]

            r_i = np.sum(fit < fit[i])

            if self.rng.random() < (1.0 - r_i / self.SN):
                best_off_idx = int(np.argmax(cosines))
            else:
                remaining = [k for k in range(3)]
                argmax_idx = int(np.argmax(cosines))
                remaining.remove(argmax_idx)
                best_off_idx = remaining[self.rng.integers(len(remaining))]

            v_i = offsprings[best_off_idx]
            f_v = self._evaluate(v_i)

            if f_v <= fit[i]:
                X[i] = v_i
                fit[i] = f_v
                self.trial[i] = 0
            else:
                self.trial[i] += 1

        return X, fit

    def _onlooker_bee_phase(self, X, fit):
        S = np.zeros(3)

        for i in range(self.SN):
            b_idx, n_idx, f_idx = self._identify_three_neighbors(i, X, fit)
            for t, idx in enumerate([b_idx, n_idx, f_idx]):
                improvement = max(0.0, fit[i] - fit[idx])
                S[t] += improvement

        if S.sum() <= 0:
            p = np.array([1 / 3, 1 / 3, 1 / 3])
        else:
            p = S / S.sum()

        for i in range(self.SN):
            if self.fes >= self.max_fes:
                break

            b_idx, n_idx, f_idx = self._identify_three_neighbors(i, X, fit)
            choice_type = self.rng.choice(3, p=p)
            chosen_idx = [b_idx, n_idx, f_idx][choice_type]

            Xb_i = X[b_idx]
            X_hat = X[chosen_idx]

            phi = self.rng.uniform(-1, 1, self.D)
            j_rand = self.rng.integers(self.D)
            v = Xb_i.copy()
            mask = (self.rng.random(self.D) <= self.CR_o)
            mask[j_rand] = True
            v[mask] = X_hat[mask] + phi[mask] * (X_hat[mask] - X[self._random_other(b_idx)][mask])
            v = self._bound_repair(v)

            f_v = self._evaluate(v)
            if f_v <= fit[b_idx]:
                X[b_idx] = v
                fit[b_idx] = f_v
                self.trial[b_idx] = 0
            else:
                self.trial[b_idx] += 1

        return X, fit

    def _random_other(self, exclude_idx):
        candidates = [k for k in range(self.SN) if k != exclude_idx]
        return candidates[self.rng.integers(len(candidates))]

    def _scout_bee_phase(self, X, fit, gbest):
        for i in range(self.SN):
            if self.fes >= self.max_fes:
                break
            if self.trial[i] > self.limit:
                _, n_idx, f_idx = self._identify_three_neighbors(i, X, fit)

                r = self.rng.random(3)
                r = r / r.sum()

                TX = r[0] * X[i] + r[1] * gbest + r[2] * (X[n_idx] - X[f_idx])
                TX = self._bound_repair(TX)
                f_TX = self._evaluate(TX)

                X[i] = TX
                fit[i] = f_TX
                self.trial[i] = 0
        return X, fit

    def run(self):
        X, fit = self._init_population()
        self.fes = self.SN

        best_idx = int(np.argmin(fit))
        gbest = X[best_idx].copy()
        gbest_val = fit[best_idx]

        while self.fes < self.max_fes:
            X, fit = self._employed_bee_phase(X, fit)
            if self.fes >= self.max_fes:
                break
            X, fit = self._onlooker_bee_phase(X, fit)
            if self.fes >= self.max_fes:
                break

            cur_best_idx = int(np.argmin(fit))
            if fit[cur_best_idx] < gbest_val:
                gbest_val = fit[cur_best_idx]
                gbest = X[cur_best_idx].copy()

            X, fit = self._scout_bee_phase(X, fit, gbest)

            cur_best_idx = int(np.argmin(fit))
            if fit[cur_best_idx] < gbest_val:
                gbest_val = fit[cur_best_idx]
                gbest = X[cur_best_idx].copy()

        error = gbest_val - self.optimum
        return error


def run_experiment(dim, n_runs=N_RUNS, seed=0, verbose=True):
    results = {}

    for func_id in range(1, 29):
        cls = CEC2013_CLASSES[func_id]
        problem = cls(ndim=dim)
        lb, ub = problem.lb, problem.ub
        optimum = problem.f_global

        errors = []

        for run_idx in range(n_runs):
            rng = np.random.default_rng(seed + 1000 * func_id + run_idx)
            solver = ABCMNG(
                obj_func=problem.evaluate,
                lb=lb, ub=ub, dim=dim, optimum=optimum,
                sn=SN, limit=LIMIT, max_fes=FES_FACTOR * dim,
                cr_e=CR_E, cr_o=CR_O, r_ratio=R_RATIO, rng=rng,
            )
            err = solver.run()
            errors.append(err)
            results[func_id] = errors

            if verbose:
                print(f"[D={dim}] F{func_id:02d} run {run_idx + 1:02d}/{n_runs} "
                      f"-> error = {err:.6e}")

        if verbose:
            print(f"[D={dim}] F{func_id:02d} completed.")

    return results


def results_to_dataframe(results, n_runs=N_RUNS):
    rows = []
    for func_id in range(1, 29):
        errs = np.array(results.get(func_id, []))
        if len(errs) == 0:
            mean, std = np.nan, np.nan
        else:
            mean, std = errs.mean(), errs.std()
        rows.append({
            "Function": f"F{func_id:02d}",
            "Mean": mean,
            "Std": std,
            "N_runs": len(errs),
        })
    return pd.DataFrame(rows)


def main():
    for dim in DIMENSIONS:
        print(f"\n===== ABC-MNG on CEC2013, D = {dim} "
              f"(SN={SN}, limit={LIMIT}, Max_FEs={FES_FACTOR * dim}, "
              f"nruns={N_RUNS}) =====\n")
        results = run_experiment(dim, n_runs=N_RUNS, seed=42, verbose=True)
        df = results_to_dataframe(results, n_runs=N_RUNS)
        csv_path = os.path.join(OUTPUT_DIR, f"ABC-MNG_CEC2013_D{dim}.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nResults for D={dim} saved to {csv_path}")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
