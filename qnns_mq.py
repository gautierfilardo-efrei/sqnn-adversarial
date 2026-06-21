"""
Multi-qubit stochastic Quantum Neural Network (QNNS-MQ)
=======================================================
Extension of Filardo & Heckmann, Neurocomputing 663 (2026) 132031.

Fixes the single-qubit information bottleneck (class-1 recall = 0.0 in the
original Table B.2) by:
  - encoding features over N entangled qubits with trainable data re-uploading,
  - a RING entanglement topology (consistent with the WETICE-2026 finding),
  - decoherence modelled as a phase-damping (dephasing) channel applied after
    each entangling block. This channel IS the integrated Lindblad dissipator
    D[sigma_z] of Appendix A, so the QSDE story is preserved exactly,
  - a trained read-out (instead of a fixed threshold),
  - honest classical baselines (LogReg / SVM-RBF / RandomForest / MLP) on the
    SAME datasets and splits (reviewer #4 checklist, item 1).

All numbers printed by this script are produced by actually running it.
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score

# ----------------------------------------------------------------------------
# Datasets (full protocol documented; reviewer #4 checklist, item 2)
# ----------------------------------------------------------------------------
def get_dataset(name, n=200, seed=0):
    if name == "blobs":          # linearly separable (original paper task)
        X, y = make_blobs(n_samples=n, centers=2, n_features=2,
                          cluster_std=1.5, random_state=seed)
    elif name == "xor":          # non-linear, XOR-structured (4 clusters)
        rng = np.random.default_rng(seed)
        cen = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1]], float)
        lab = np.array([0, 0, 1, 1])
        idx = rng.integers(0, 4, n)
        X = cen[idx] + rng.normal(0, 0.35, (n, 2))
        y = lab[idx]
    elif name == "circles":      # non-linear, concentric
        X, y = make_circles(n_samples=n, noise=0.10, factor=0.45,
                            random_state=seed)
    else:
        raise ValueError(name)
    return X.astype(float), y.astype(int)


# ----------------------------------------------------------------------------
# Quantum model
# ----------------------------------------------------------------------------
class QNNS_MQ:
    def __init__(self, n_qubits=4, n_layers=3, p_deco=0.0,
                 topology="ring", noise="phase", seed=0):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.p_deco = p_deco
        self.topology = topology
        self.noise = noise   # "phase" = D[sigma_z] dephasing ; "depol" = isotropic
        self.dev = qml.device("default.mixed", wires=n_qubits)
        rng = np.random.default_rng(seed)
        # trainable: input scales w_in (L,Q), rotations w_rot (L,Q,2),
        # read-out scale a and bias b
        self.params = {
            "w_in":  pnp.array(rng.normal(0, 1.0, (n_layers, n_qubits)), requires_grad=True),
            "w_rot": pnp.array(rng.normal(0, 0.3, (n_layers, n_qubits, 2)), requires_grad=True),
            "a":     pnp.array(2.0, requires_grad=True),
            "b":     pnp.array(0.0, requires_grad=True),
        }
        self._build()

    def _entangle(self):
        Q = self.n_qubits
        if self.topology == "ring":
            pairs = [(i, (i + 1) % Q) for i in range(Q)]
        elif self.topology == "linear":
            pairs = [(i, i + 1) for i in range(Q - 1)]
        elif self.topology == "none":
            pairs = []
        else:
            raise ValueError(self.topology)
        for c, t in pairs:
            qml.CNOT(wires=[c, t])

    def _build(self):
        @qml.qnode(self.dev, interface="autograd", diff_method="backprop")
        def qnode(x, w_in, w_rot, p):
            for l in range(self.n_layers):
                for i in range(self.n_qubits):
                    feat = x[..., i % x.shape[-1]]               # tile 2 feats over Q qubits
                    qml.RY(w_in[l, i] * feat + w_rot[l, i, 0], wires=i)
                    qml.RZ(w_rot[l, i, 1], wires=i)
                self._entangle()
                if p > 0:
                    for i in range(self.n_qubits):
                        if self.noise == "phase":
                            qml.PhaseDamping(p, wires=i)          # = integrated Lindblad D[sigma_z]
                        else:
                            qml.DepolarizingChannel(p, wires=i)   # isotropic decoherence
            return qml.expval(qml.PauliZ(0))
        self.qnode = qnode

    def logits(self, X, params=None):
        p = params or self.params
        z = self.qnode(X, p["w_in"], p["w_rot"], self.p_deco)
        return p["a"] * z + p["b"]

    def proba(self, X, params=None):
        return 1.0 / (1.0 + pnp.exp(-self.logits(X, params)))

    def fit(self, X, y, epochs=80, lr=0.1, verbose=False):
        Xb = pnp.array(X, requires_grad=False)
        yb = pnp.array(y.astype(float), requires_grad=False)
        opt = qml.AdamOptimizer(lr)
        keys = list(self.params.keys())

        def cost(*vals):
            p = dict(zip(keys, vals))
            pr = self.proba(Xb, p)
            pr = pnp.clip(pr, 1e-7, 1 - 1e-7)
            return -pnp.mean(yb * pnp.log(pr) + (1 - yb) * pnp.log(1 - pr))

        vals = [self.params[k] for k in keys]
        for e in range(epochs):
            vals = opt.step(cost, *vals)
            if verbose and (e % 20 == 0 or e == epochs - 1):
                print(f"   epoch {e:3d}  loss={float(cost(*vals)):.4f}")
        self.params = dict(zip(keys, vals))
        return self

    def predict(self, X):
        return (np.array(self.proba(pnp.array(X, requires_grad=False))) >= 0.5).astype(int)


# ----------------------------------------------------------------------------
# Classical baselines (same data, same split)
# ----------------------------------------------------------------------------
def classical_baselines(Xtr, ytr, Xte, seed=0):
    models = {
        "LogReg":       LogisticRegression(max_iter=1000),
        "SVM-RBF":      SVC(kernel="rbf", C=1.0, gamma="scale"),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=seed),
        "MLP":          MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000,
                                      random_state=seed),
    }
    out = {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        out[name] = m.predict(Xte)
    return out


def metrics(y, yhat):
    return (accuracy_score(y, yhat),
            recall_score(y, yhat, pos_label=1, zero_division=0),  # the metric the paper lost
            f1_score(y, yhat, average="macro", zero_division=0))
