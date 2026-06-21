"""UNSW-NB15 loader (d=4 PCA) + numpy MLP with white-box PGD, adversarial training,
Gaussian-noise injection / smoothing. Exact input gradients (no transfer attack)."""
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load_unsw(n_samples=700, n_components=4, seed=0, path="unsw_train.csv"):
    df = pd.read_csv(path)
    y = df["label"].values.astype(int)
    X = df.drop(columns=[c for c in ["id","attack_cat","label"] if c in df.columns]).copy()
    for c in ["proto","service","state"]:
        if c in X.columns: X[c] = pd.factorize(X[c])[0]
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(float)
    rng = np.random.RandomState(seed)
    # balanced stratified subsample
    idx0 = rng.permutation(np.where(y==0)[0])[:n_samples//2]
    idx1 = rng.permutation(np.where(y==1)[0])[:n_samples//2]
    idx = rng.permutation(np.concatenate([idx0,idx1])); X,y = X[idx],y[idx]
    Xs = StandardScaler().fit_transform(X)
    Xp = PCA(n_components=n_components, random_state=seed).fit_transform(Xs)
    Xp = StandardScaler().fit_transform(Xp)  # unit-scale PCA space so eps is comparable
    ntr = int(0.7*len(Xp))
    return Xp[:ntr], Xp[ntr:], y[:ntr], y[ntr:]

# ---------- numpy MLP ----------
def init_mlp(d, h, seed=0):
    r = np.random.RandomState(seed)
    return {"W1":r.randn(d,h)*np.sqrt(2/d),"b1":np.zeros(h),
            "W2":r.randn(h,1)*np.sqrt(2/h),"b2":np.zeros(1)}
def _fwd(p,X):
    z1=X@p["W1"]+p["b1"]; a1=np.maximum(z1,0); z2=a1@p["W2"]+p["b2"]
    return 1/(1+np.exp(-z2.ravel())), (z1,a1,z2)
def predict_mlp(p,X,noise=0.0,K=1):
    if noise<=0: return (_fwd(p,X)[0]>=0.5).astype(int)
    pr=np.zeros(len(X))
    for _ in range(K):
        pr+=_fwd(p,X+np.random.normal(0,noise,X.shape))[0]
    return (pr/K>=0.5).astype(int)
def input_grad(p,X,y):
    pr,(z1,a1,z2)=_fwd(p,X); dz2=(pr-y).reshape(-1,1)        # dL/dz2 for BCE+sigmoid
    da1=dz2@p["W2"].T; dz1=da1*(z1>0); dX=dz1@p["W1"].T
    return dX/len(X)
def _grads(p,X,y):
    pr,(z1,a1,z2)=_fwd(p,X); n=len(X); dz2=(pr-y).reshape(-1,1)/n
    gW2=a1.T@dz2; gb2=dz2.sum(0); da1=dz2@p["W2"].T; dz1=da1*(z1>0)
    return {"W1":X.T@dz1,"b1":dz1.sum(0),"W2":gW2,"b2":gb2}
def pgd_linf_mlp(p,X,y,eps,steps,noise=0.0):
    X0=np.array(X); Xa=X0+np.random.uniform(-eps,eps,X0.shape)
    for _ in range(steps): Xa=np.clip(Xa+(eps/4)*np.sign(input_grad(p,Xa,y)),X0-eps,X0+eps)
    return (predict_mlp(p,Xa,noise,16 if noise>0 else 1)==y).mean()
def pgd_l2_mlp(p,X,y,eps2,steps,noise=0.0):
    X0=np.array(X); a=eps2/4; d=np.random.normal(size=X0.shape); d/=np.linalg.norm(d,axis=1,keepdims=True)+1e-12
    Xa=X0+np.random.uniform(0,eps2,(len(X0),1))*d
    for _ in range(steps):
        g=input_grad(p,Xa,y); g/=np.linalg.norm(g,axis=1,keepdims=True)+1e-12; Xa=Xa+a*g
        de=Xa-X0; nm=np.linalg.norm(de,axis=1,keepdims=True); Xa=X0+de*np.minimum(1.0,eps2/(nm+1e-12))
    return (predict_mlp(p,Xa,noise,16 if noise>0 else 1)==y).mean()
def train_mlp(X,y,h=32,epochs=300,lr=0.1,mode="vanilla",eps=0.3,noise=0.0,seed=0):
    p=init_mlp(X.shape[1],h,seed)
    for _ in range(epochs):
        if mode=="adv":   # Madry PGD adversarial training (linf)
            Xt=np.array(X); Xt=Xt+np.random.uniform(-eps,eps,Xt.shape)
            for _ in range(7): Xt=np.clip(Xt+(eps/4)*np.sign(input_grad(p,Xt,y)),X-eps,X+eps)
        elif mode=="gauss": Xt=X+np.random.normal(0,noise,X.shape)
        else: Xt=X
        g=_grads(p,Xt,y)
        for k in p: p[k]-=lr*g[k]
    return p
