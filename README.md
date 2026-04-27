# DECODE-RAPL

**Predicting Intel RAPL package power from OS-visible signals — a record of what worked, what didn't, and why.**

This repository is a research log of an attempt to build a deep-learning model that emulates the Intel RAPL (Running Average Power Limit) energy counter using only signals that are visible to a VM guest (`/proc/stat` user/system/iowait %, context switches, etc.). RAPL is implemented in silicon and is exposed only to bare-metal hosts via MSRs or `/sys/class/powercap/intel_rapl`; if a model can be trained on the bare-metal RAPL ground truth using *only* features that a VM guest would also be able to observe, that model could later run inside guest VMs that have no direct power telemetry — a direction relevant to projects like [Kepler](https://github.com/sustainable-computing-io/kepler).

> **Headline finding.** Across four model families and many hyperparameter sweeps, every variant achieves **R² ≈ 0.95–0.96, MAE ≈ 2 W on a shuffled stress-ng test set**, but degrades to **R² ≈ 0.30** with a **+3 to +15 W idle-baseline bias** on live, non-stress-ng workloads. The persistent gap suggests a *feature ceiling*: VM-visible OS metrics are blind to the microarchitectural state (instruction mix, cache pressure, functional-unit activation) that ultimately determines power. This is the canonical interim conclusion in [`DECORE-RAPL-REPORT.md`](DECORE-RAPL-REPORT.md) — that document is the most important thing to read.

---

## Where to look first

| If you want… | Read |
|---|---|
| The full story and final analysis | [`DECORE-RAPL-REPORT.md`](DECORE-RAPL-REPORT.md) |
| The 12-slide visual walkthrough (reveal.js) | [`presentation.html`](presentation.html) — open in a browser |
| A developer-level quickstart for the v2/v3/v4 model code | [`QUICKSTART.md`](QUICKSTART.md) |
| The current model architecture (1D-CNN encoder + power head) | [`decode_rapl_v4.md`](decode_rapl_v4.md) |
| Per-version training reports | [`V2_TRAINING_REPORT.md`](V2_TRAINING_REPORT.md), [`results/`](results) |
| Future direction notes (multi-machine generalization) | [`decode_rapl_v-later.md`](decode_rapl_v-later.md) |
| The MS-TCN attempt (separate architecture, also bare-metal) | [`archive/ms-tcn/docs/MS_TCN_APPROACH_REPORT.md`](archive/ms-tcn/docs/MS_TCN_APPROACH_REPORT.md), [`archive/ms-tcn/results/RESULTS.md`](archive/ms-tcn/results/RESULTS.md) |
| The very first DECODE-RAPL attempt (kept for context) | [`archive/decode-rapl-v1/README.md`](archive/decode-rapl-v1/README.md), [`archive/decode-rapl-v1/LSTM_ANALYSIS_SUMMARY.md`](archive/decode-rapl-v1/LSTM_ANALYSIS_SUMMARY.md) |

## Repository layout

The current research lives at the repo root (the v2/v3/v4 model code, configs, reports, scripts, and the trained checkpoint). Three earlier attempts that were superseded are kept under [`archive/`](archive) so the progression of ideas remains visible.

```
.
├── README.md, LICENSE
├── DECORE-RAPL-REPORT.md      ← canonical interim report — read this first
├── QUICKSTART.md               ← developer quickstart for the model code
├── decode_rapl_v3.md, decode_rapl_v4.md, decode_rapl_v-later.md
├── V2_TRAINING_REPORT.md, TRAINING.md, report-improvement.md
├── src/         (model.py, train.py, inference.py, power_predictor.py, utils.py)
├── scripts/     (run_workloads.sh, prepare_training_data.py, start_training_v4.sh, …)
├── collector/   (my_data_collector.go — Go-based 16 ms RAPL+OS-metrics collector)
├── config/      (v2/v3/v4 YAML configs for τ ∈ {1, 4, 8})
├── docs/        (architecture.md, data_collection.md, training.md, inference.md, …)
├── results/     (per-run training logs, summaries, plots)
├── checkpoints/v4_tau1/best_model.pt  (1.6 MB — shipped for inference)
└── archive/
    ├── lstm/             ← earliest baseline (per-VM CPU% → power via small LSTM)
    ├── ms-tcn/           ← MS-TCN with attention pooling; hit a data-distribution wall
    └── decode-rapl-v1/   ← first DECODE-RAPL attempt (AE+LSTM); latent bottleneck identified
```

## Test environment

All data collection and live tests were run on a single bare-metal host:

| Field | Value |
|---|---|
| Hardware | Dell PowerEdge R440 |
| CPU | Intel Xeon Silver 4210R @ 2.40 GHz (Cascade Lake R) |
| Sockets / cores / threads | 1 / 10 / 20 (HT on) |
| NUMA nodes | 1 |
| RAM | 96 GB DDR4 |
| RAPL domains | package (`/sys/class/powercap/intel-rapl:0`) |
| OS | Fedora 39 |

## Data availability

The core training dataset is **5.8 M samples** collected at 16 ms intervals across 2,025+ stress-ng workload combinations on the bare-metal data-collection rig above. **The dataset itself was collected at the author's employer and its public release is currently under internal review** — it is not yet downloadable from this repo.

Until then, the pipeline is fully reproducible from scratch on your own bare-metal hardware:

* Workload generator: [`scripts/run_workloads.sh`](scripts/run_workloads.sh)
* Low-overhead Go collector: [`collector/my_data_collector.go`](collector/my_data_collector.go)
* Preprocessing: [`scripts/prepare_training_data.py`](scripts/prepare_training_data.py)
* Training (v4): [`scripts/start_training_v4.sh`](scripts/start_training_v4.sh) + [`config/v4_tau1.yaml`](config/v4_tau1.yaml)

## Quick inference (no retraining needed)

A trained v4_tau1 checkpoint is shipped at [`checkpoints/v4_tau1/best_model.pt`](checkpoints/v4_tau1/best_model.pt). Note that **inference uses RAPL only as a comparison reference** — for prediction itself the model only needs the four VM-visible features, but the live-prediction script reads RAPL to display the error. If you run it inside a VM, drop the `--compare-rapl` flag.

```bash
sudo python src/power_predictor.py \
    --model checkpoints/v4_tau1/best_model.pt \
    --config config/v4_tau1.yaml \
    --live
```

See [`docs/inference.md`](docs/inference.md) for full inference options.

## Status

Research-quality. There is no claim of production readiness, no stable API, and the open-source release is intended to share the *thinking and the negative result*, not to ship a product. Issues, PRs, and especially counter-examples (workloads on which the v4_tau1 checkpoint behaves differently than reported) are very welcome.

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).
