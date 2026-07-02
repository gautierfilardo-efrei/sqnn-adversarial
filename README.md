# SQNN — Adversarial Robustness and Noise Regularisation for Network Intrusion Detection

Code accompanying the paper:

> **Decoherence as Defence and the Magnitude of Noise Regularisation: A Rigorous *N*-Qubit Theory of Stochastic Quantum Neural Networks for Adversarially Robust Network Intrusion Detection**
> G.-E. Filardo.

A ring-entangled **Stochastic Quantum Neural Network (SQNN)** for intrusion detection, studied on the real **NSL-KDD** dataset. The repository reproduces:

- **Adversarial robustness** under white-box FGSM / PGD: an inference-time depolarising channel trades ≈1 clean point for a large, *stable* robustness gain, while classical neural detectors collapse (95%→47%).
- **Noise regularisation** (30 seeds): both depolarising noise and per-gate *quantum dropout* reduce the train–test gap by a small but significant, mutually equivalent margin — predicted by an adaptive-penalty formula.
- **Decoherence-contraction theorem**: a depolarising channel over *L* layers contracts every weight-*w* Pauli read-out (and the adversarial input gradient) by (1−4γ/3)^{wL}.

Interactive notebook (Colab): see `QDropout_experiment.ipynb`.

## Install

```bash
git clone https://github.com/gautierfilardo-efrei/sqnn-adversarial.git
cd sqnn-adversarial
pip install -r requirements.txt
```

Python ≥ 3.10. The simulator is PennyLane `default.mixed` (density matrix), which runs on **CPU**; a GPU runtime gives no speed-up for this backend.

## Reproduce the paper

NSL-KDD is downloaded automatically on first run (cached locally; not redistributed here).

| Paper result | Command | Output |
|---|---|---|
| Adversarial robustness (Table 2, Fig. 2) | `python run_adv.py 0.0 0.05 0.15 0.30` | `adv_results.json` |
| Noise regularisation, 30 seeds (Table 3, Fig. 3) | `python qnns_program.py run 30` then `python qnns_program.py fig` | `results_10seeds.json`, `fig_dropout_nslkdd.png`, `fig_dropout_mean.png` |
| Dropout-rate sweep, depth 5 (Fig. 4) | `python run_pdrop.py` | `pdrop_results.json` |
| Qadence variant (parent / Pasqal stack) | `python adversarial_qadence.py` | — |

The 30-seed run takes ≈2 h on one CPU core; it **checkpoints** to `results_10seeds.json`, so re-running resumes where it stopped. Reduce `SIZES`/seeds for a quick test.

### Files

- `qnns_program.py` — ring SQNN + per-gate dropout; 30-seed regularisation experiment and figures (`run` / `fig`).
- `qdropout.py` — standalone per-gate quantum-dropout model.
- `qnns_mq.py` — multi-qubit ring SQNN with depolarising / dephasing channels.
- `adv_qnns.py` — NSL-KDD loader, FGSM / PGD attacks, robustness experiment.
- `run_adv.py` — driver for the adversarial-robustness table.
- `run_pdrop.py` — depth-5 dropout-rate sweep.
- `adversarial_qadence.py` — Qadence (Pasqal) implementation matching the predecessor paper.
- `QDropout_experiment.ipynb` — self-contained Colab notebook.

## Citation

```bibtex
@article{filardo_sqnn_adversarial,
  title  = {Decoherence as Defence and the Magnitude of Noise Regularisation:
            A Rigorous N-Qubit Theory of Stochastic Quantum Neural Networks
            for Adversarially Robust Network Intrusion Detection},
  author = {Filardo, Gautier-Edouard},
  year   = {2026}
}
```

NSL-KDD: M. Tavallaee, E. Bagheri, W. Lu, A. A. Ghorbani, *A detailed analysis of the KDD CUP 99 data set*, IEEE CISDA, 2009.

## License

MIT — see [`LICENSE`](LICENSE).
# SQNN — Adversarial Robustness and Noise Regularisation for Network Intrusion Detection

Code accompanying the paper:

> **An *N*-Qubit Theory of Noise-Regularised Stochastic Quantum Neural Networks: Decoherence Contraction, Adaptive Dropout, and Emergent Adversarial Robustness**
> G.-E. Filardo.

A ring-entangled **Stochastic Quantum Neural Network (SQNN)** studied on the real **NSL-KDD** and **UNSW-NB15** datasets. The repository reproduces:

- **Decoherence-contraction theorem** — a depolarising channel over *L* entangling layers contracts every weight-*w* Pauli read-out by (1−4γ/3)^{wL}.
- **Adversarial robustness** under white-box FGSM / PGD — a depolarising channel **applied during training** reshapes the decision boundary into a flatter, less attackable one. Over seven seeds under exact-gradient PGD-20, the noisy SQNN is significantly more robust and, unlike the noiseless circuit and gradient-trained classical detectors (95%→47%), never suffers catastrophic collapse. The effect generalises to UNSW-NB15 — where classical adversarial training remains the most robust model: **no quantum advantage is claimed**.
- **Noise regularisation** (30 seeds) — both depolarising noise and per-gate *quantum dropout* reduce the train–test gap by a small but significant, mutually equivalent margin, matching an adaptive-penalty formula (curvature-weighted L2, maximised at p = 1/2).

Interactive notebook (Colab): `QDropout_experiment.ipynb`.

## Install

```bash
git clone https://github.com/gautierfilardo-efrei/sqnn-adversarial.git
cd sqnn-adversarial
pip install -r requirements.txt
```

Python ≥ 3.10. The SQNN simulator is PennyLane `default.mixed` (density matrix), CPU-bound; a GPU gives no speed-up for this backend.

## Reproduce the paper

NSL-KDD and UNSW-NB15 are downloaded automatically on first run (cached locally; not redistributed here).

| Paper result | Command | Output |
|---|---|---|
| Adversarial robustness, single-seed table + FGSM-degradation figure (Table 2) | `python run_adv.py 0.0 0.05 0.15 0.30` | `adv_results.json` |
| Robustness over 7 seeds — strong exact-gradient PGD-20, ℓ∞ and ℓ₂ (Table 3) | `python run_robust_check.py 0 1 2 3 4 5 6` | `robust_check.json` |
| Second dataset + defended classical baselines, UNSW-NB15 (Table 4) | `python run_unsw.py 0 1 2` | `unsw_results.json` |
| Noise regularisation, 30 seeds (Table 5) | `python qnns_program.py run 30` then `python qnns_program.py fig` | `results_10seeds.json`, `fig_dropout_nslkdd.png`, `fig_dropout_mean.png` |
| Dropout-rate sweep, depth 5 | `python run_pdrop.py` | `pdrop_results.json` |
| Neutral-atom Rydberg reservoir on XOR — hardware-mapping demo (Section 6) | `python reservoir_rydberg.py --n 8 --seeds 100` | stdout (use `--out res.json`) |
| Qadence variant (parent / Pasqal stack; **optional** deps) | `python adversarial_qadence.py` | — |

The 30-seed run takes ≈2 h on one CPU core; it **checkpoints** to `results_10seeds.json` and resumes on re-run. `run_robust_check.py` and `run_unsw.py` also checkpoint (to `robust_check.json` / `unsw_results.json`) and take arbitrary seed lists as arguments. Reduce sizes/seeds for a quick test.

> **Note on the reservoir demo.** `reservoir_rydberg.py` is a *standalone* demonstration that the ring Hamiltonian of Section 6 runs as a neutral-atom Rydberg reservoir: with a linear read-out it lifts XOR into linear separability (well above the ≈chance accuracy of a linear head on the raw inputs). It is provided as a feasibility check for the hardware mapping and is **not** a figure in the paper; the reported accuracy depends on the stated cluster-noise level and seed count.

### Files

- `qnns_program.py` — ring SQNN + per-gate dropout; 30-seed regularisation experiment and figures (`run` / `fig`).
- `qdropout.py` — standalone per-gate quantum-dropout model.
- `qnns_mq.py` — multi-qubit ring SQNN with depolarising / dephasing channels.
- `adv_qnns.py` — NSL-KDD loader, FGSM / PGD attacks, robustness experiment.
- `run_adv.py` — driver for the single-seed adversarial-robustness table.
- `run_robust_check.py` — seven-seed strong-attack study (exact-gradient PGD-20, ℓ∞/ℓ₂).
- `run_unsw.py`, `unsw_lib.py` — UNSW-NB15 study vs defended classical baselines (white-box, exact gradients).
- `run_pdrop.py` — depth-5 dropout-rate sweep.
- `reservoir_rydberg.py` — neutral-atom Rydberg reservoir on XOR (Section 6 hardware-mapping demo).
- `adversarial_qadence.py` — Qadence (Pasqal) digital variant; requires the optional extras below.
- `QDropout_experiment.ipynb` — self-contained Colab notebook.

## Citation

```bibtex
@article{filardo_sqnn_nqubit_theory,
  title  = {An N-Qubit Theory of Noise-Regularised Stochastic Quantum Neural
            Networks: Decoherence Contraction, Adaptive Dropout, and Emergent
            Adversarial Robustness},
  author = {Filardo, Gautier-Edouard},
  year   = {2026}
}
```

NSL-KDD: M. Tavallaee, E. Bagheri, W. Lu, A. A. Ghorbani, *A detailed analysis of the KDD CUP 99 data set*, IEEE CISDA, 2009.
UNSW-NB15: N. Moustafa, J. Slay, *UNSW-NB15: a comprehensive data set for network intrusion detection systems*, MilCIS, 2015.
Neutral-atom reservoir computing: P. Bravo, K. Bharti, et al., *PRX Quantum* 3, 030325 (2022).

## License

MIT — see [`LICENSE`](LICENSE).
