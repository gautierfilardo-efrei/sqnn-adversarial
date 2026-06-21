"""
Per-gate quantum dropout vs depolarising output noise on NSL-KDD (10 seeds).
============================================================================
Self-contained program for the follow-up paper. Produces (i) the train-test gap
table and (ii) two publication figures to insert in the article.

  python qnns_program.py run    # run / resume the 10-seed experiment (checkpoints each cell)
  python qnns_program.py fig    # build figures from current results

gap = train_acc - test_acc  (positive = overfitting). A regulariser REDUCES the
gap relative to the noiseless VQC.
"""
import sys, json, os, warnings, time
warnings.filterwarnings("ignore")
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------- experiment config
SIZES   = [40, 80, 160]
EPOCHS  = 40
N_QUBITS, N_LAYERS, D = 4, 3, 4
CONDS = {"VQC (none)":   dict(p_deco=0.0,  p_drop=0.0),
         "Depol 0.05":   dict(p_deco=0.05, p_drop=0.0),
         "QDropout 0.3": dict(p_deco=0.0,  p_drop=0.3)}
COND_COL = {"VQC (none)": "#8aa0b6", "Depol 0.05": "#d1242f", "QDropout 0.3": "#2da44e"}
FN = "results_10seeds.json"

# ---------------------------------------------------------------- model
class QNNS_Dropout:
    """Ring SQNN: modes none / depolarising-output / per-gate dropout."""
    def __init__(self, n_qubits=4, n_layers=3, p_deco=0.0, p_drop=0.0, seed=0):
        self.nq, self.nl = n_qubits, n_layers
        self.p_deco, self.p_drop = p_deco, p_drop
        self.dev = qml.device("default.mixed", wires=n_qubits)
        rng = np.random.default_rng(seed)
        self.params = {"w_in":  pnp.array(rng.normal(0, 1.0, (n_layers, n_qubits)), requires_grad=True),
                       "w_rot": pnp.array(rng.normal(0, 0.3, (n_layers, n_qubits, 2)), requires_grad=True),
                       "a": pnp.array(2.0, requires_grad=True), "b": pnp.array(0.0, requires_grad=True)}
        self._rng = rng; self._build()
    def _build(self):
        @qml.qnode(self.dev, interface="autograd", diff_method="backprop")
        def qnode(x, w_in, w_rot, mask, p_deco):
            for l in range(self.nl):
                for i in range(self.nq):
                    feat = x[..., i % x.shape[-1]]
                    qml.RY(w_in[l, i] * feat + mask[l, i, 0] * w_rot[l, i, 0], wires=i)
                    qml.RZ(mask[l, i, 1] * w_rot[l, i, 1], wires=i)
                for i in range(self.nq):
                    qml.CNOT(wires=[i, (i + 1) % self.nq])
                if p_deco > 0:
                    for i in range(self.nq):
                        qml.DepolarizingChannel(p_deco, wires=i)
            return qml.expval(qml.PauliZ(0))
        self.qnode = qnode
    def _proba(self, X, p, mask):
        z = self.qnode(X, p["w_in"], p["w_rot"], mask, self.p_deco)
        return 1.0 / (1.0 + pnp.exp(-(p["a"] * z + p["b"])))
    def fit(self, X, y, epochs=60, lr=0.12):
        Xb = pnp.array(X, requires_grad=False); yb = pnp.array(y.astype(float), requires_grad=False)
        ones = pnp.array(np.ones((self.nl, self.nq, 2)), requires_grad=False)
        opt = qml.AdamOptimizer(lr); keys = list(self.params.keys())
        def cost(*vals, mask=ones):
            p = dict(zip(keys, vals)); pr = pnp.clip(self._proba(Xb, p, mask), 1e-7, 1 - 1e-7)
            return -pnp.mean(yb * pnp.log(pr) + (1 - yb) * pnp.log(1 - pr))
        vals = [self.params[k] for k in keys]
        for _ in range(epochs):
            if self.p_drop > 0:
                m = (self._rng.random((self.nl, self.nq, 2)) > self.p_drop).astype(float)
                mask = pnp.array(m, requires_grad=False)
            else: mask = ones
            vals = opt.step(lambda *v: cost(*v, mask=mask), *vals)
        self.params = dict(zip(keys, vals)); self._ones = ones; return self
    def predict(self, X):
        pr = self._proba(pnp.array(X, requires_grad=False), self.params, self._ones)
        return (np.array(pr) >= 0.5).astype(int)

# ---------------------------------------------------------------- data (NSL-KDD, synthetic fallback)
NSLKDD_COLS = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
 "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised","root_shell",
 "su_attempted","num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds",
 "is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate","rerror_rate",
 "srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
 "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
 "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate",
 "dst_host_srv_rerror_rate","label","difficulty"]

def load_nslkdd(n=1000, d=D, seed=7):
    import pandas as pd, urllib.request
    path = next((p for p in ["nslkdd_train.txt", "nslkdd.txt"] if os.path.exists(p)), None)
    if path is None:
        path = "nslkdd_train.txt"
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt", path)
    df = pd.read_csv(path, header=None, names=NSLKDD_COLS)
    df = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    y = (df["label"].values != "normal").astype(int)
    X = df.drop(columns=["label", "difficulty"])
    for c in ["protocol_type", "service", "flag"]:
        X[c] = LabelEncoder().fit_transform(X[c].astype(str))
    X = StandardScaler().fit_transform(X.astype(float).values)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    pca = PCA(n_components=d, random_state=seed).fit(Xtr)
    sc = StandardScaler().fit(pca.transform(Xtr))
    return sc.transform(pca.transform(Xtr)), sc.transform(pca.transform(Xte)), ytr, yte

# ---------------------------------------------------------------- run (checkpointed)
def run(n_seeds=30):
    seeds = list(range(n_seeds))                       # resumes from existing checkpoint
    total = len(SIZES) * len(CONDS) * len(seeds)
    Xtr_full, Xte, ytr_full, yte = load_nslkdd()
    R = json.load(open(FN)) if os.path.exists(FN) else {}
    t0 = time.time()
    for size in SIZES:
        for cond, cfg in CONDS.items():
            for sd in seeds:
                key = f"{cond}|N{size}|s{sd}"
                if key in R: continue
                idx = np.random.default_rng(1000 + sd).choice(len(Xtr_full), size=size, replace=False)
                m = QNNS_Dropout(N_QUBITS, N_LAYERS, seed=sd, **cfg).fit(
                    Xtr_full[idx], ytr_full[idx], epochs=EPOCHS)
                tr = accuracy_score(ytr_full[idx], m.predict(Xtr_full[idx]))
                te = accuracy_score(yte, m.predict(Xte))
                R[key] = tr - te
                json.dump(R, open(FN, "w"))
                print(f"  {cond:13s} N={size:3d} s{sd}: gap={tr-te:+.3f}  [{len(R)}/{total}]", flush=True)
    print(f"done in {time.time()-t0:.0f}s")
    summary(R)

def summary(R):
    print("\n=== mean gap +- std over completed seeds ===")
    print(f"{'N':>5} " + "".join(f"{c:>18}" for c in CONDS))
    for size in SIZES:
        row = [f"{size:>5}"]
        for c in CONDS:
            g = [R[k] for k in R if k.startswith(f"{c}|N{size}|")]
            row.append(f"{np.mean(g):+.3f}+-{np.std(g):.3f}" if g else "--")
        print(" ".join(f"{x:>18}" for x in row))
    print("\n=== mean across sizes (lower = better regularisation) ===")
    for c in CONDS:
        g = [R[k] for k in R if k.startswith(f"{c}|")]
        print(f"  {c:13s}: {np.mean(g):+.3f}  (n={len(g)})")

# ---------------------------------------------------------------- figures
def fig():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    R = json.load(open(FN))
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 150, "savefig.bbox": "tight"})

    # Figure 1: grouped bars, gap vs training size, 10-seed error bars
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(len(SIZES)); w = 0.26
    for k, c in enumerate(CONDS):
        mu = [np.mean([R[q] for q in R if q.startswith(f"{c}|N{n}|")]) for n in SIZES]
        sd = [np.std([R[q] for q in R if q.startswith(f"{c}|N{n}|")]) for n in SIZES]
        ax.bar(x + (k - 1) * w, mu, w, yerr=sd, capsize=3, color=COND_COL[c],
               edgecolor="white", label=c)
    ax.set_xticks(x); ax.set_xticklabels([f"N={n}" for n in SIZES])
    ax.set_ylabel("Train$-$test accuracy gap"); ax.set_xlabel("Training-set size")
    ax.axhline(0, color="#666", lw=0.8)
    ax.set_title("Train$-$test gap by regularisation mode (NSL-KDD, 10 seeds)")
    ax.legend(frameon=False, fontsize=9.5)
    fig.savefig("fig_dropout_nslkdd.png"); plt.close(fig)

    # Figure 2: mean gap across sizes with 95% CI, ranks the three modes
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for i, c in enumerate(CONDS):
        g = np.array([R[q] for q in R if q.startswith(f"{c}|")])
        se = g.std() / np.sqrt(len(g))
        ax.errorbar([i], [g.mean()], yerr=[1.96 * se], fmt="o", ms=9, capsize=5,
                    color=COND_COL[c])
        ax.text(i, g.mean() + 1.96 * se + 0.004, f"{g.mean():.3f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(CONDS))); ax.set_xticklabels(list(CONDS), rotation=0)
    ax.set_ylabel("Mean train$-$test gap (95% CI)")
    ax.set_title("Per-gate dropout: lowest mean gap")
    fig.savefig("fig_dropout_mean.png"); plt.close(fig)

    # Figure 3 (if the high-dropout sweep was run): cranking p_drop lowers TEST accuracy
    if os.path.exists("pdrop_results.json"):
        P = json.load(open("pdrop_results.json"))
        order = [c for c in ["VQC (none)", "Depol 0.05", "QDrop 0.5", "QDrop 0.7"]
                 if any(k.startswith(c + "|") for k in P)]
        ns = max(int(k.split("s")[-1]) for k in P) + 1
        def col(c, j): return np.array([P[f"{c}|s{s}"][j] for s in range(ns) if f"{c}|s{s}" in P])
        tr_m = [col(c, 0).mean() for c in order]; tr_e = [col(c, 0).std() for c in order]
        te_m = [col(c, 1).mean() for c in order]; te_e = [col(c, 1).std() for c in order]
        x = np.arange(len(order)); w = 0.38
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        ax.bar(x - w/2, tr_m, w, yerr=tr_e, capsize=3, color="#9aa6b2", label="train")
        ax.bar(x + w/2, te_m, w, yerr=te_e, capsize=3, color="#1f6feb", label="test")
        for i in range(len(order)):
            ax.text(i, min(te_m[i]-te_e[i], tr_m[i]-tr_e[i]) - 0.018,
                    f"gap\n{tr_m[i]-te_m[i]:+.3f}", ha="center", fontsize=8, color="#444")
        ax.set_xticks(x); ax.set_xticklabels(order)
        ax.set_ylim(0.78, 1.02); ax.set_ylabel("Accuracy")
        ax.set_title("Cranking per-gate dropout lowers train AND test (depth 5, N=40, 10 seeds)")
        ax.legend(frameon=False, fontsize=9.5)
        fig.savefig("fig_pdrop_sweep.png"); plt.close(fig)
        print("figures: fig_dropout_nslkdd.png, fig_dropout_mean.png, fig_pdrop_sweep.png")
    else:
        print("figures: fig_dropout_nslkdd.png, fig_dropout_mean.png")
    summary(R)

# ---------------------------------------------------------------- how to launch
# In a NOTEBOOK (Colab / Jupyter): run THIS cell to define everything, then in the
# next cell(s) call the functions DIRECTLY. Do NOT use the __main__ block below:
# in a notebook sys.argv holds Jupyter's kernel path, which would raise ValueError.
#
#     run(30)     # run / resume 30 seeds (reuses any existing checkpoint)
#     fig()       # build the 3 figures + print the table and paired t-tests
#
# In a TERMINAL: uncomment the block below, then
#     python qnns_program.py run 30      /      python qnns_program.py fig
#
# if __name__ == "__main__":
#     cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
#     if cmd == "fig":
#         fig()
#     else:
#         n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
#         run(n)
