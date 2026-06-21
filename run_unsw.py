"""2nd dataset (UNSW-NB15) + defended classical baselines vs SQNN.
Addresses reviewer concerns 5 (second dataset) and 6 (defended baselines, white-box, frontier)."""
import sys, json, os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pennylane as qml
from pennylane import numpy as pnp
from sklearn.metrics import accuracy_score
from adv_qnns import QNNS_MQ, _bce
import unsw_lib as U

def grad_x(m,X,y):
    Xv=pnp.array(X,requires_grad=True); return np.array(qml.grad(lambda Xv:_bce(m,Xv,y))(Xv))
def q_pgd_linf(m,X,y,eps,steps):
    X0=np.array(X); Xa=X0+np.random.uniform(-eps,eps,X0.shape)
    for _ in range(steps): Xa=np.clip(Xa+(eps/4)*np.sign(grad_x(m,Xa,y)),X0-eps,X0+eps)
    return (m.predict(Xa)==y).mean()
def q_pgd_l2(m,X,y,eps2,steps):
    X0=np.array(X); a=eps2/4; d=np.random.normal(size=X0.shape); d/=np.linalg.norm(d,axis=1,keepdims=True)+1e-12
    Xa=X0+np.random.uniform(0,eps2,(len(X0),1))*d
    for _ in range(steps):
        g=grad_x(m,Xa,y); g/=np.linalg.norm(g,axis=1,keepdims=True)+1e-12; Xa=Xa+a*g
        de=Xa-X0; nm=np.linalg.norm(de,axis=1,keepdims=True); Xa=X0+de*np.minimum(1.0,eps2/(nm+1e-12))
    return (m.predict(Xa)==y).mean()

EPS=0.3; EPS2=0.5
fn="unsw_results.json"; R=json.load(open(fn)) if os.path.exists(fn) else {}
seeds=[int(s) for s in sys.argv[1:]] if len(sys.argv)>1 else [0]
for sd in seeds:
    Xtr,Xte,ytr,yte=U.load_unsw(n_samples=700,n_components=4,seed=sd); yf=yte.astype(float)
    # ---- classical defended baselines (white-box, fast) ----
    for name,kw,ns in [("MLP",dict(mode="vanilla"),0.0),
                       ("MLP-AdvTrain",dict(mode="adv",eps=EPS),0.0),
                       ("MLP-Gauss",dict(mode="gauss",noise=0.3),0.3)]:
        key=f"{name}|s{sd}"
        if key in R: continue
        p=U.train_mlp(Xtr,ytr,epochs=300,lr=0.1,seed=sd,**kw); K=16 if ns>0 else 1
        R[key]={"clean":float((U.predict_mlp(p,Xte,ns,K)==yte).mean()),
                "pgd20_linf":float(U.pgd_linf_mlp(p,Xte,yf,EPS,20,ns)),
                "pgd20_l2":float(U.pgd_l2_mlp(p,Xte,yf,EPS2,20,ns))}
        json.dump(R,open(fn,"w"))
    # ---- SQNN gamma=0 and 0.30 (slow) ----
    for g in [0.0,0.30]:
        key=f"SQNN-g{g}|s{sd}"
        if key in R: continue
        t0=time.time(); m=QNNS_MQ(n_qubits=4,n_layers=3,p_deco=g,topology="ring",noise="depol",seed=sd)
        m.fit(Xtr,ytr,epochs=45,lr=0.12)
        R[key]={"clean":float(accuracy_score(yte,m.predict(Xte))),
                "pgd20_linf":float(q_pgd_linf(m,Xte,yte,EPS,20)),
                "pgd20_l2":float(q_pgd_l2(m,Xte,yte,EPS2,20))}
        json.dump(R,open(fn,"w")); print(f"  s{sd} SQNN g={g:.2f} clean={R[key]['clean']:.3f} L8={R[key]['pgd20_linf']:.3f} L2={R[key]['pgd20_l2']:.3f} ({time.time()-t0:.0f}s)",flush=True)
    # print classical for this seed
    for name in ["MLP","MLP-AdvTrain","MLP-Gauss"]:
        r=R[f"{name}|s{sd}"]; print(f"  s{sd} {name:13s} clean={r['clean']:.3f} L8={r['pgd20_linf']:.3f} L2={r['pgd20_l2']:.3f}",flush=True)
