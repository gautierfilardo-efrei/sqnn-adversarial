import warnings, json, os, time, sys
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.metrics import accuracy_score
from adv_qnns import (load_nslkdd, QNNS_MQ, fgsm_q, pgd_q,
                      LogisticRegression, RandomForestClassifier, MLPClassifier, SVC,
                      fgsm_classical)

d=4; seed=0; epochs=55; eps_list=[0.1,0.3]
Xtr,Xte,ytr,yte = load_nslkdd(n_samples=700, n_components=d, seed=42)
fn="adv_results.json"
R = json.load(open(fn)) if os.path.exists(fn) else {"qnns":{}, "classical":{}, "meta":{}}
R["meta"]={"ntr":len(Xtr),"nte":len(Xte),"d":d,"attack_rate":float(yte.mean())}

gammas = [float(x) for x in sys.argv[1:]] if len(sys.argv)>1 else [0.0,0.05,0.15,0.30]
for g in gammas:
    if str(g) in R["qnns"]: continue
    t0=time.time()
    q=QNNS_MQ(n_qubits=d,n_layers=3,p_deco=g,topology="ring",noise="depol",seed=seed)
    q.fit(Xtr,ytr,epochs=epochs,lr=0.12)
    rec={"clean":float(accuracy_score(yte,q.predict(Xte)))}
    for e in eps_list:
        rec[f"fgsm@{e}"]=float(accuracy_score(yte,q.predict(fgsm_q(q,Xte,yte,e))))
        rec[f"pgd@{e}"]=float(accuracy_score(yte,q.predict(pgd_q(q,Xte,yte,e,steps=5))))
    R["qnns"][str(g)]=rec
    json.dump(R,open(fn,"w"),indent=1)
    print(f"  gamma={g:.2f} clean={rec['clean']:.3f} fgsm@.3={rec['fgsm@0.3']:.3f} pgd@.3={rec['pgd@0.3']:.3f} ({time.time()-t0:.0f}s)",flush=True)

# classical (fast) - only once
if not R["classical"]:
    surr=LogisticRegression(max_iter=2000).fit(Xtr,ytr)
    for name,clf in {"RF":RandomForestClassifier(n_estimators=200,random_state=seed),
                     "MLP":MLPClassifier(hidden_layer_sizes=(64,32),max_iter=800,random_state=seed),
                     "SVM-RBF":SVC(kernel="rbf")}.items():
        clf.fit(Xtr,ytr)
        rec={"clean":float(accuracy_score(yte,clf.predict(Xte)))}
        for e in eps_list:
            rec[f"fgsm@{e}"]=float(accuracy_score(yte,clf.predict(fgsm_classical(clf,surr,Xte,yte,e))))
        R["classical"][name]=rec
    json.dump(R,open(fn,"w"),indent=1)
    print("  classical:", {k:round(v['clean'],3) for k,v in R['classical'].items()})
