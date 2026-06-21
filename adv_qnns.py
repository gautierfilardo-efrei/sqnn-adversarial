"""
Adversarial robustness of stochastic quantum neural networks (follow-up study)
==============================================================================
Tests the central hypothesis of the follow-up paper: the intrinsic decoherence
gamma of an SQNN, which *reduces* clean accuracy, *increases* robustness to
gradient-based adversarial perturbations (FGSM / PGD). The mechanism is the
decoherence-contraction theorem of the N-qubit theory: a depolarising layer
contracts every weight-w Pauli expectation by eta^{wL} with eta=1-4gamma/3,
shrinking the input-gradient that the attacker exploits.

Continuity with the parent (collaborative-anomaly-detection) paper:
  - same ring entanglement topology and sigma_z read-out (via qnns_mq),
  - same role of gamma as the stochastic/decoherence knob,
  - REAL data (NSL-KDD) instead of the synthetic cyber generator.
"""
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from qnns_mq import QNNS_MQ

# NSL-KDD column names (41 features + label + difficulty)
NSLKDD_COLS = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
    "num_shells","num_access_files","num_outbound_cmds","is_host_login",
    "is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
    "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
    "srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty",
]
CATEGORICAL = ["protocol_type", "service", "flag"]


def load_nslkdd(path="nslkdd_train.txt", n_samples=None, n_components=4, seed=42):
    """Parse NSL-KDD, binary-encode (normal=0 / attack=1), reduce to PCA features.
    Returns standardized Xtr,Xte,ytr,yte in the reduced space shared by all models."""
    import pandas as pd
    df = pd.read_csv(path, header=None, names=NSLKDD_COLS)
    if n_samples is not None:
        df = df.sample(n=min(n_samples, len(df)), random_state=seed).reset_index(drop=True)
    y = (df["label"].values != "normal").astype(int)                 # binary IDS label
    X = df.drop(columns=["label", "difficulty"])
    for c in CATEGORICAL:                                            # encode categoricals
        X[c] = LabelEncoder().fit_transform(X[c].astype(str))
    X = X.astype(float).values
    X = StandardScaler().fit_transform(X)
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                          random_state=seed, stratify=y)
    pca = PCA(n_components=n_components, random_state=seed).fit(Xtr)  # shared d-dim space
    Xtr, Xte = pca.transform(Xtr), pca.transform(Xte)
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xte), ytr, yte


# ----------------------------------------------------------------------------
# Gradient-based attacks in the (shared) feature space
# ----------------------------------------------------------------------------
def _bce(model, Xv, y):
    pr = pnp.clip(model.proba(Xv), 1e-7, 1 - 1e-7)
    yb = pnp.array(y.astype(float), requires_grad=False)
    return -pnp.mean(yb * pnp.log(pr) + (1 - yb) * pnp.log(1 - pr))

def fgsm_q(model, X, y, eps):
    Xv = pnp.array(X, requires_grad=True)
    g = qml.grad(lambda Xv: _bce(model, Xv, y))(Xv)
    return np.array(X) + eps * np.sign(np.array(g))

def pgd_q(model, X, y, eps, alpha=None, steps=7):
    alpha = alpha or eps / 4
    X0 = np.array(X)
    Xadv = X0 + np.random.uniform(-eps, eps, X0.shape)
    for _ in range(steps):
        Xv = pnp.array(Xadv, requires_grad=True)
        g = qml.grad(lambda Xv: _bce(model, Xv, y))(Xv)
        Xadv = Xadv + alpha * np.sign(np.array(g))
        Xadv = np.clip(Xadv, X0 - eps, X0 + eps)        # project to eps-ball
    return Xadv

# classical attacks (finite-difference-free: use sklearn's decision_function gradient
# via a smooth surrogate is overkill; for tree/MLP we use a transfer attack from a
# differentiable logistic surrogate fit on the same features -> standard practice)
def fgsm_classical(clf, surrogate, X, y, eps):
    w = surrogate.coef_.ravel()
    margin = surrogate.decision_function(X)
    sign = np.sign((2 * y - 1))[:, None]      # push toward the wrong side
    return X - eps * sign * np.sign(w)[None, :]


# ----------------------------------------------------------------------------
# Experiment: clean + adversarial accuracy across gamma and across models
# ----------------------------------------------------------------------------
def run_robustness(Xtr, ytr, Xte, yte, gammas=(0.0, 0.05, 0.15, 0.30),
                   eps_list=(0.1, 0.3), seed=0, epochs=60):
    out = {"qnns": {}, "classical": {}}

    # quantum models at each gamma (gamma=0 is the noiseless VQC baseline)
    for g in gammas:
        q = QNNS_MQ(n_qubits=Xtr.shape[1], n_layers=3, p_deco=g,
                    topology="ring", noise="depol", seed=seed)
        q.fit(Xtr, ytr, epochs=epochs, lr=0.12)
        rec = {"clean": accuracy_score(yte, q.predict(Xte))}
        for eps in eps_list:
            rec[f"fgsm@{eps}"] = accuracy_score(yte, q.predict(fgsm_q(q, Xte, yte, eps)))
            rec[f"pgd@{eps}"]  = accuracy_score(yte, q.predict(pgd_q(q, Xte, yte, eps)))
        out["qnns"][g] = rec

    # classical baselines + transfer FGSM via logistic surrogate
    surrogate = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    for name, clf in {"RF": RandomForestClassifier(n_estimators=200, random_state=seed),
                      "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800,
                                           random_state=seed),
                      "SVM-RBF": SVC(kernel="rbf")}.items():
        clf.fit(Xtr, ytr)
        rec = {"clean": accuracy_score(yte, clf.predict(Xte))}
        for eps in eps_list:
            Xa = fgsm_classical(clf, surrogate, Xte, yte, eps)
            rec[f"fgsm@{eps}"] = accuracy_score(yte, clf.predict(Xa))
        out["classical"][name] = rec
    return out
