# PG-POD-TCN

Physics-guided reduced-order surrogate modeling for CFD-DEM gas-solid flows.

本项目实现了一套可复现的 `CFD-DEM → POD → TCN` 代理建模流程：将颗粒数据映射到
Eulerian 网格，用 POD 压缩固相体积分数场，以因果扩张 TCN 预测未来流场，并联合预测
床层压降与床层高度。训练损失显式约束固体总体积和体积分数边界，推理端通过模型集成与
Mahalanobis 距离识别高不确定性、分布外工况。

> **Research integrity:** the repository includes a synthetic conical-bed generator only
> to test the software path. It is not a CFD-DEM solver, is not experimental evidence, and
> must not be reported as a scientific result. No CPWV or UPV/EHU research data are included.

## Features

- Complete-case train/validation/test splitting to prevent temporal leakage
- Validated `.npz` contract for real CFD-DEM cases
- Conservative particle-to-grid coarse graining
- Training-only POD fitting with configurable captured energy
- Persistence-initialized residual causal TCN and POD-LSTM baseline
- Direct multi-horizon prediction of POD coefficients
- Differentiable POD reconstruction and multi-task physics-guided loss
- Persistence and POD-DMD baselines
- Deep-ensemble uncertainty and Mahalanobis OOD detection
- Synthetic end-to-end demo, figures, metrics, tests, and GitHub Actions CI

## Quick start

Python 3.10 or newer is required.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pg-pod-tcn demo --config configs/demo.yaml
```

The demo writes artifacts to `outputs/demo/`:

```text
preprocessors.npz       # POD basis and train-only normalization
ood_detector.npz        # Mahalanobis detector
checkpoint_seed_*.pt    # best validation checkpoints
history_seed_*.json     # auditable training curves
config.json             # complete resolved experiment configuration
split.json              # complete-case split
metrics_test.json       # model and baseline metrics
predictions_test.npz    # capped sample predictions
example_test.png        # target/prediction/error visualization
```

Demo metrics are synthetic software-test results. Do not copy them into a resume or paper.

## Commands

```bash
# Only generate deterministic synthetic cases
pg-pod-tcn generate --config configs/demo.yaml

# Train an experiment
pg-pod-tcn train --config configs/demo.yaml

# Evaluate the saved ensemble
pg-pod-tcn evaluate --config configs/demo.yaml --split test

# Predict one compatible complete case
pg-pod-tcn predict data/synthetic/case_009.npz --config configs/demo.yaml
```

For real data, copy `configs/real_data.yaml`, point `data.root` at your case directory, and
adjust the parameter ranges, time horizon, POD cap, batch size, and loss weights through
validation experiments.

## Repository layout

```text
configs/                    experiment configurations
docs/                       algorithm and data-contract notes
scripts/                    conversion helpers
src/pg_pod_tcn/
  coarse_graining.py        conservative particle-to-grid mapping
  data.py                   case schema and leakage-safe window dataset
  pod.py                    proper orthogonal decomposition
  models.py                 TCN and LSTM forecasters
  losses.py                 differentiable reconstruction and physics losses
  training.py               early-stopped ensemble training
  evaluation.py             metrics, baselines, plots, saved predictions
  ood.py                    Mahalanobis out-of-distribution detector
tests/                      unit and end-to-end smoke tests
```

## Real-data contract

Each `.npz` file represents one complete operating condition and contains:

```python
case_id             # scalar string
time                # [T]
solid_fraction      # [T, *grid]
pressure_drop       # [T]
bed_height          # [T]
params              # [P]
param_names         # [P]
cell_volumes        # [*grid], zero outside the physical domain
solid_volume        # [T]
```

See [docs/data_format.md](docs/data_format.md) and
[`scripts/convert_arrays_to_case.py`](scripts/convert_arrays_to_case.py).

## Model and loss

For POD basis `Phi`, the model predicts normalized reduced coefficients and reconstructs
the future field through

```text
alpha_s(x, t) ≈ mean_alpha_s(x) + sum_i a_i(t) Phi_i(x).
```

The total objective combines coefficient Huber loss, reconstructed-field MSE, relative
solid-volume conservation, physical bounds, and normalized pressure/bed-height MSE. See
[docs/algorithm.md](docs/algorithm.md) for details and technical references.

## Reproducibility and reporting

- POD, normalizers, and OOD statistics are fitted on training cases only.
- Metrics are computed on complete unseen cases.
- Multiple seeds should be reported as mean ± standard deviation.
- Keep the split manifest, configuration, checkpoint, training log, and solver provenance.
- Call this method physics-guided, not a PINN.
- Report acceleration only after timing the same physical horizon on documented hardware.

## License

MIT. Before publishing, replace the placeholder GitHub URLs and security contact, and add
the actual author names to `pyproject.toml` and `CITATION.cff`.
