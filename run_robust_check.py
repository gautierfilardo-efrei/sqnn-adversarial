"""Reviewer stress-test: SQNN robustness real or artefact? gamma=0 vs gamma=0.30.
Exact gradients (density matrix) => no shot-noise gradient masking by construction;
strong PGD-20 is the adaptive test. L2 added (magnitude matters, unlike sign-based Linf).
"""
import sys, json, os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.metrics import accuracy_score
from adv_qnns import load_nslkdd, QNNS_MQ, _bce

def grad_x(model, X, y):
    Xv = pnp.array(X, requires_grad=True)
    return np.array(qml.grad(lambda Xv: _bce(model, Xv, y))(Xv))

def pgd_linf(model, X, y, eps, steps, restarts=1):
    X0=np.array(X); correct=np.ones(len(X0),bool)
    for _ in range(restarts):
        Xa=X0+np.random.uniform(-eps,eps,X0.shape)
        for _ in range(steps):
            Xa=np.clip(Xa+(eps/4)*np.sign(grad_x(model,Xa,y)),X0-eps,X0+eps)
        correct&=(model.predict(Xa)==y)
    return correct.mean()

def pgd_l2(model, X, y, eps2, steps, restarts=1):
    X0=np.array(X); correct=np.ones(len(X0),bool); alpha=eps2/4
    for _ in range(restarts):
        d=np.random.normal(size=X0.shape); d/=np.linalg.norm(d,axis=1,keepdims=True)+1e-12
        Xa=X0+np.random.uniform(0,eps2,(len(X0),1))*d
        for _ in range(steps):
            g=grad_x(model,Xa,y); g/=np.linalg.norm(g,axis=1,keepdims=True)+1e-12
            Xa=Xa+alpha*g
            delta=Xa-X0; norm=np.linalg.norm(delta,axis=1,keepdims=True)
            Xa=X0+delta*np.minimum(1.0,eps2/(norm+1e-12))
        correct&=(model.predict(Xa)==y)
    return correct.mean()

GAMMAS=[0.0,0.30]; EPS=0.3; EPS2=0.5
fn="robust_check.json"; R=json.load(open(fn)) if os.path.exists(fn) else {}
seeds=[int(s) for s in sys.argv[1:]] if len(sys.argv)>1 else [0]
for sd in seeds:
    Xtr,Xte,ytr,yte=load_nslkdd(n_samples=700,n_components=4,seed=200+sd)
    for g in GAMMAS:
        key=f"g{g}|s{sd}"
        if key in R: continue
        t0=time.time()
        m=QNNS_MQ(n_qubits=4,n_layers=3,p_deco=g,topology="ring",noise="depol",seed=sd)
        m.fit(Xtr,ytr,epochs=45,lr=0.12)
        rec={"clean":float(accuracy_score(yte,m.predict(Xte))),
             "pgd5_linf":float(pgd_linf(m,Xte,yte,EPS,5)),
             "pgd20_linf":float(pgd_linf(m,Xte,yte,EPS,20)),
             "pgd20_l2":float(pgd_l2(m,Xte,yte,EPS2,20))}
        R[key]=rec; json.dump(R,open(fn,"w"))
        print(f"  s{sd} g={g:.2f} clean={rec['clean']:.3f} pgd5L8={rec['pgd5_linf']:.3f} "
              f"pgd20L8={rec['pgd20_linf']:.3f} pgd20L2={rec['pgd20_l2']:.3f} ({time.time()-t0:.0f}s)",flush=True)
