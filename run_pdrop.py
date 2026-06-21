import sys, json, os, warnings, time
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.metrics import accuracy_score
from qnns_program import QNNS_Dropout, load_nslkdd

DEPTH = 5; N = 40; SEEDS = list(range(10)); EPOCHS = 40
CONDS = {"VQC (none)":   dict(p_deco=0.0,  p_drop=0.0),
         "Depol 0.05":   dict(p_deco=0.05, p_drop=0.0),
         "QDrop 0.5":    dict(p_deco=0.0,  p_drop=0.5),
         "QDrop 0.7":    dict(p_deco=0.0,  p_drop=0.7)}
FN = "pdrop_results.json"
R = json.load(open(FN)) if os.path.exists(FN) else {}
Xtr_full, Xte, ytr_full, yte = load_nslkdd()

for cond, cfg in CONDS.items():
    for sd in SEEDS:
        key = f"{cond}|s{sd}"
        if key in R: continue
        idx = np.random.default_rng(1000+sd).choice(len(Xtr_full), size=N, replace=False)
        m = QNNS_Dropout(4, DEPTH, seed=sd, **cfg).fit(Xtr_full[idx], ytr_full[idx], epochs=EPOCHS)
        tr = accuracy_score(ytr_full[idx], m.predict(Xtr_full[idx]))
        te = accuracy_score(yte, m.predict(Xte))
        R[key] = [tr, te]
        json.dump(R, open(FN, "w"))
        print(f"  {cond:11s} s{sd}: train={tr:.3f} test={te:.3f} gap={tr-te:+.3f} [{len(R)}/{len(CONDS)*len(SEEDS)}]", flush=True)
