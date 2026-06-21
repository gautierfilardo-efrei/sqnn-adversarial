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
