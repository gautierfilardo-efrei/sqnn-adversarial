"""
Adversarial robustness extension of the collaborative-anomaly-detection SQNN
===========================================================================
Drop-in continuation of the parent notebook (Qadence / PyQTorch). It keeps the
parent's ring-entangled circuit and sigma_z read-out, and ADDS:
  (1) white-box FGSM and PGD attacks in feature space (torch autograd),
  (2) a real-data loader (NSL-KDD) replacing the synthetic cyber generator,
  (3) an *inference-time* noise option.

IMPORTANT methodological note. The parent injects gamma as additive Gaussian
noise on the read-out, applied ONLY during training (`if self.training`), so the
parent SQNN is *noiseless at inference* and gamma acts purely as a regulariser.
Adversarial hardening requires the stochasticity to be present at INFERENCE.
This module therefore exposes `inference_noise`; the paper's headline robustness
numbers, however, were produced with a genuine depolarising *quantum channel*
present at inference (density-matrix simulator, see adv_qnns.py) — the cleaner
realisation of the decoherence-contraction mechanism.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

import qadence as qd
from qadence import (QuantumCircuit, QNN, FeatureParameter, RY, CNOT, Z,
                     chain, add, hea)
from qadence.types import BackendName, DiffMode

DTYPE = torch.float64


# --- parent circuit: feature map + RING entanglement + HEA ansatz ---
def build_circuit(n_qubits, n_features, depth):
    fm, names = [], []
    for i in range(n_qubits):
        nm = f"x{i % n_features}"
        if nm not in names:
            names.append(nm)
        fm.append(RY(i, FeatureParameter(nm)))
    ent = [CNOT(i, i + 1) for i in range(n_qubits - 1)] + [CNOT(n_qubits - 1, 0)]
    circ = QuantumCircuit(n_qubits, chain(chain(*fm), chain(*ent),
                                          hea(n_qubits=n_qubits, depth=depth)))
    return circ, names


class RobustQNNS(nn.Module):
    """Parent QNNSModel + an inference_noise switch so gamma can harden inference."""
    def __init__(self, n_qubits, n_features, depth, noise_gamma=0.05,
                 inference_noise=True):
        super().__init__()
        self.n_qubits = n_qubits
        self.noise_gamma = noise_gamma
        self.inference_noise = inference_noise
        circ, self.input_names = build_circuit(n_qubits, n_features, depth)
        self.qnn = QNN(circuit=circ, observable=add(Z(i) for i in range(n_qubits)),
                       inputs=self.input_names, backend=BackendName.PYQTORCH,
                       diff_mode=DiffMode.AD)
        self.threshold = nn.Parameter(torch.tensor(0.0, dtype=DTYPE))
        self.steepness = nn.Parameter(torch.tensor(5.0, dtype=DTYPE))

    def forward(self, x):
        vals = {f"x{i}": x[:, i] for i in range(len(self.input_names))}
        e = self.qnn(vals).squeeze(-1)
        if self.noise_gamma > 0 and (self.training or self.inference_noise):
            e = e + torch.randn_like(e) * self.noise_gamma
        prob = (e + self.n_qubits) / (2.0 * self.n_qubits)
        return torch.sigmoid(self.steepness * (prob - self.threshold))


def train_model(model, Xtr, ytr, epochs=50, lr=0.01, batch=32):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.BCELoss()
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            crit(model(Xtr[idx]), ytr[idx]).backward()
            opt.step()
    return model


# --- white-box attacks in feature space (torch autograd) ---
def fgsm(model, X, y, eps):
    model.eval()
    Xa = X.clone().detach().requires_grad_(True)
    loss = nn.BCELoss()(model(Xa), y)
    loss.backward()
    return (X + eps * Xa.grad.sign()).detach()

def pgd(model, X, y, eps, alpha=None, steps=7):
    model.eval()
    alpha = alpha or eps / 4
    Xa = (X + torch.empty_like(X).uniform_(-eps, eps)).detach()
    for _ in range(steps):
        Xa.requires_grad_(True)
        nn.BCELoss()(model(Xa), y).backward()
        Xa = (Xa + alpha * Xa.grad.sign()).detach()
        Xa = torch.max(torch.min(Xa, X + eps), X - eps).detach()
    return Xa


# --- NSL-KDD loader (returns torch tensors in a shared PCA feature space) ---
def load_nslkdd_torch(path="nslkdd_train.txt", n_samples=2000, d=8, seed=42):
    from adv_qnns import load_nslkdd          # reuse the numpy loader
    Xtr, Xte, ytr, yte = load_nslkdd(path, n_samples, d, seed)
    t = lambda a, f: torch.tensor(a, dtype=f)
    return (t(Xtr, DTYPE), t(Xte, DTYPE),
            t(ytr, DTYPE), t(yte, DTYPE), yte)


def evaluate(model, X, y_np):
    model.eval()
    with torch.no_grad():
        return accuracy_score(y_np, (model(X) > 0.5).float().numpy())


def run_robustness(n_qubits=8, depth=3, gammas=(0.0, 0.05, 0.15, 0.30),
                   eps_list=(0.1, 0.3), n_samples=2000, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, Xte, ytr, yte, yte_np = load_nslkdd_torch(n_samples=n_samples, d=n_qubits, seed=seed)
    res = {}
    for g in gammas:
        m = RobustQNNS(n_qubits, n_qubits, depth, noise_gamma=g, inference_noise=True).to(DTYPE)
        m = train_model(m, Xtr, ytr)
        rec = {"clean": evaluate(m, Xte, yte_np)}
        for e in eps_list:
            rec[f"fgsm@{e}"] = accuracy_score(yte_np, (m(fgsm(m, Xte, yte, e)) > 0.5).float().detach().numpy())
            rec[f"pgd@{e}"]  = accuracy_score(yte_np, (m(pgd(m, Xte, yte, e)) > 0.5).float().detach().numpy())
        res[g] = rec
        print(f"gamma={g:.2f}  {rec}")
    return res


if __name__ == "__main__":
    # requires:  pip install qadence
    run_robustness(n_qubits=8, n_samples=2000)
