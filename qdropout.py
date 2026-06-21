"""
True quantum dropout (per-gate Bernoulli deactivation) vs depolarising output
noise -- the experiment WETICE [P2] proposed but did not run.

[P2] showed the depolarising channel ACTS AS OUTPUT NOISE and *increases* the
train-test gap (does not regularise). It proposed replacing each variational
rotation R_Y(theta) by identity with probability p_drop during training only.
Here we implement that and measure whether it CLOSES the overfitting gap that
depolarising noise could not.

gap = train_accuracy - test_accuracy   (positive = overfitting).
A regulariser should REDUCE the gap relative to the no-noise VQC.
"""
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.metrics import accuracy_score


class QNNS_Dropout:
    """Ring SQNN with three regularisation modes: none / depolarising / per-gate dropout."""
    def __init__(self, n_qubits=4, n_layers=3, p_deco=0.0, p_drop=0.0, seed=0):
        self.nq, self.nl = n_qubits, n_layers
        self.p_deco, self.p_drop = p_deco, p_drop
        self.dev = qml.device("default.mixed", wires=n_qubits)
        rng = np.random.default_rng(seed)
        self.params = {
            "w_in":  pnp.array(rng.normal(0, 1.0, (n_layers, n_qubits)), requires_grad=True),
            "w_rot": pnp.array(rng.normal(0, 0.3, (n_layers, n_qubits, 2)), requires_grad=True),
            "a":     pnp.array(2.0, requires_grad=True),
            "b":     pnp.array(0.0, requires_grad=True),
        }
        self._rng = rng
        self._build()

    def _build(self):
        @qml.qnode(self.dev, interface="autograd", diff_method="backprop")
        def qnode(x, w_in, w_rot, mask, p_deco):
            for l in range(self.nl):
                for i in range(self.nq):
                    feat = x[..., i % x.shape[-1]]
                    # data encoding kept; VARIATIONAL rotations masked by `mask`
                    qml.RY(w_in[l, i] * feat + mask[l, i, 0] * w_rot[l, i, 0], wires=i)
                    qml.RZ(mask[l, i, 1] * w_rot[l, i, 1], wires=i)
                for i in range(self.nq):
                    qml.CNOT(wires=[i, (i + 1) % self.nq])      # ring
                if p_deco > 0:
                    for i in range(self.nq):
                        qml.DepolarizingChannel(p_deco, wires=i)
            return qml.expval(qml.PauliZ(0))
        self.qnode = qnode

    def _proba(self, X, p, mask):
        z = self.qnode(X, p["w_in"], p["w_rot"], mask, self.p_deco)
        return 1.0 / (1.0 + pnp.exp(-(p["a"] * z + p["b"])))

    def fit(self, X, y, epochs=60, lr=0.12):
        Xb = pnp.array(X, requires_grad=False)
        yb = pnp.array(y.astype(float), requires_grad=False)
        ones = pnp.array(np.ones((self.nl, self.nq, 2)), requires_grad=False)
        opt = qml.AdamOptimizer(lr)
        keys = list(self.params.keys())

        def cost(*vals, mask=ones):
            p = dict(zip(keys, vals))
            pr = pnp.clip(self._proba(Xb, p, mask), 1e-7, 1 - 1e-7)
            return -pnp.mean(yb * pnp.log(pr) + (1 - yb) * pnp.log(1 - pr))

        vals = [self.params[k] for k in keys]
        for _ in range(epochs):
            if self.p_drop > 0:    # fresh per-gate Bernoulli mask each step (train only)
                m = (self._rng.random((self.nl, self.nq, 2)) > self.p_drop).astype(float)
                mask = pnp.array(m, requires_grad=False)
            else:
                mask = ones
            vals = opt.step(lambda *v: cost(*v, mask=mask), *vals)
        self.params = dict(zip(keys, vals))
        self._ones = ones
        return self

    def predict(self, X):    # inference: FULL circuit (mask = 1), no dropout
        Xb = pnp.array(X, requires_grad=False)
        pr = self._proba(Xb, self.params, self._ones)
        return (np.array(pr) >= 0.5).astype(int)


def gap(model, Xtr, ytr, Xte, yte):
    return (accuracy_score(ytr, model.predict(Xtr)),
            accuracy_score(yte, model.predict(Xte)))
