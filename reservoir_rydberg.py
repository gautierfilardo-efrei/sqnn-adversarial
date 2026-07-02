"""
reservoir_rydberg.py  --  Neutral-atom (Rydberg) quantum reservoir on the XOR task.

A ring of N atoms (with per-seed geometric jitter) evolves under the Rydberg
Hamiltonian H = (Omega/2) sum_i X_i  -  sum_i delta_i n_i  +  sum_{i<j} V_ij n_i n_j,
where each input (a,b) is injected through heterogeneous *local* detunings
delta_i = s * (W[i,0] a + W[i,1] b + c_i), with W, c fixed random "input weights"
per reservoir instance. The reservoir read-out is the vector of single-atom
excitations <n_i> and blockade-induced pair correlations <n_i n_j>; a LINEAR
(logistic) head is trained on top. Closed-system statevector simulation.

Methodology follows neutral-atom quantum reservoir computing
(Bravo et al., PRX Quantum 3, 030325, 2022); tooling-compatible with Pulser.

Usage:
    python reservoir_rydberg.py --n 8  --seeds 100
    python reservoir_rydberg.py --n 10 --seeds 57
"""
import argparse, itertools, json, warnings
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

def make_xor(n_samples, noise, seed):
    rng = np.random.default_rng(seed)
    C = np.array([[0,0],[0,1],[1,0],[1,1]], float); L = np.array([0,1,1,0])
    idx = rng.integers(0, 4, n_samples)
    return C[idx] + noise*rng.standard_normal((n_samples,2)), L[idx]

def ring_positions(n, jitter, rng):
    ang = 2*np.pi*np.arange(n)/n; R = 1.0/(2*np.sin(np.pi/n))
    xy = np.c_[R*np.cos(ang), R*np.sin(ang)]
    return xy + jitter*rng.standard_normal((n,2))

def static_parts(n, Omega, xy, V0):
    dim = 2**n
    B = ((np.arange(dim)[:,None] >> np.arange(n)[::-1]) & 1).astype(float)
    I2 = sp.identity(2, format='csr'); X = sp.csr_matrix(np.array([[0.,1.],[1.,0.]]))
    Xsum = sp.csr_matrix((dim,dim))
    for i in range(n):
        op = sp.identity(1, format='csr')
        for k in range(n): op = sp.kron(op, X if k==i else I2, format='csr')
        Xsum = Xsum + op
    Vdiag = np.zeros(dim)
    for i in range(n):
        for j in range(i+1,n):
            Vij = V0/np.linalg.norm(xy[i]-xy[j])**6
            Vdiag += Vij*B[:,i]*B[:,j]
    return B, (Omega/2.0)*Xsum, Vdiag

def reservoir_features(Xin, n, seed, Omega, V0, dscale, T, jitter):
    rng = np.random.default_rng(seed)
    xy = ring_positions(n, jitter, rng)
    B, Hx, Vdiag = static_parts(n, Omega, xy, V0)
    W = rng.standard_normal((n,2)); c = 0.5*rng.standard_normal(n)
    psi0 = np.zeros(2**n, complex); psi0[0] = 1.0
    pairs = list(itertools.combinations(range(n),2))
    feats = []
    for x in Xin:
        D = Vdiag - B @ (dscale*(W@x + c))
        psi = expm_multiply(-1j*T*(Hx + sp.diags(D)), psi0)
        p = np.abs(psi)**2
        feats.append(np.concatenate([p@B, [p@(B[:,i]*B[:,j]) for (i,j) in pairs]]))
    return np.array(feats)

def run_one(n, seed, n_samples, noise, Omega, V0, dscale, T, jitter):
    X, y = make_xor(n_samples, noise, seed)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    sc = StandardScaler().fit(Xtr)
    Ftr = reservoir_features(sc.transform(Xtr), n, seed, Omega, V0, dscale, T, jitter)
    Fte = reservoir_features(sc.transform(Xte), n, seed, Omega, V0, dscale, T, jitter)
    head = LogisticRegression(max_iter=5000, C=10.0).fit(Ftr, ytr)
    # classical linear baseline on RAW inputs (XOR -> ~chance)
    base = LogisticRegression(max_iter=5000).fit(sc.transform(Xtr), ytr).score(sc.transform(Xte), yte)
    return head.score(Fte, yte), base

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--noise", type=float, default=0.15)
    ap.add_argument("--Omega", type=float, default=1.0)
    ap.add_argument("--V0", type=float, default=6.0)
    ap.add_argument("--dscale", type=float, default=4.0)
    ap.add_argument("--T", type=float, default=4.0)
    ap.add_argument("--jitter", type=float, default=0.05)
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()
    accs, bases = [], []
    for s in range(a.seeds):
        acc, base = run_one(a.n, s, a.samples, a.noise, a.Omega, a.V0, a.dscale, a.T, a.jitter)
        accs.append(acc); bases.append(base)
    accs = np.array(accs); bases = np.array(bases)
    print(f"N={a.n}  seeds={a.seeds}  reservoir test acc = {100*accs.mean():.1f} +/- {100*accs.std():.1f} %"
          f"   (linear-on-raw baseline = {100*bases.mean():.1f} +/- {100*bases.std():.1f} %)")
    if a.out:
        json.dump({"n":a.n,"acc":accs.tolist(),"baseline":bases.tolist()}, open(a.out,"w"))
    return accs

if __name__ == "__main__":
    main()
