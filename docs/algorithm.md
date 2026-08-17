# Algorithm notes

PG-POD-TCN is a non-intrusive reduced-order model. It does not modify the CFD-DEM solver.

1. Lagrangian particle data are coarse-grained onto the CFD grid.
2. POD is fitted only on training-case solid-fraction snapshots.
3. POD coefficients and static operating parameters form a causal time-series input.
4. A residual dilated TCN starts from persistence and predicts corrections over a direct
   multi-step coefficient horizon.
5. Two auxiliary heads predict pressure drop and bed height.
6. The differentiable POD decoder reconstructs the field during training.
7. Data, field, solid-mass, volume-fraction-bound, and macroscopic losses are optimized
   jointly.
8. A deep ensemble estimates epistemic uncertainty; a training-fitted, validation-calibrated
   Mahalanobis detector identifies operating conditions far from the known distribution.

The method is "physics-guided," not a PINN: it enforces selected invariants and bounds but
does not minimize the full Navier-Stokes and particle-equation residuals.

## Evaluation hierarchy

- Persistence forecast: no learned dynamics.
- POD-DMD: linear reduced dynamics.
- POD-LSTM: neural recurrent baseline (implemented with the same output contract).
- POD-TCN without physics losses: ablation.
- Full PG-POD-TCN.

Report field nRMSE, global SSIM, pressure-drop MAPE, bed-height MAPE, solid-volume drift,
and model latency. A speed-up ratio is valid only when measured against the same physical
horizon and documented CFD-DEM hardware.

## Technical basis

- [CPWV publication record: conical fluidized bed, CFD-DEM, planar imaging, and pressure
  fluctuation analysis](https://www.ehu.eus/en/web/cpwv/110/-/asset_publisher/4ob4VHnPp9z3/content/art.25-multi-stage/1329573)
- [POD-LSTM reduced-order modeling for CFD-DEM particle-laden flows](https://arxiv.org/abs/2403.14283)
- [Mass-conserving physics-informed DMD for gas-solid flow prediction](https://arxiv.org/abs/2311.02600)
