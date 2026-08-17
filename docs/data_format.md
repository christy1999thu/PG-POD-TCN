# CFD-DEM data contract

The unit of splitting is a complete operating condition, not an individual snapshot.
Store every case as one compressed NumPy file. The loader uses `allow_pickle=False`.

| Array | Shape | Meaning |
|---|---:|---|
| `case_id` | scalar string | Stable public identifier |
| `time` | `[T]` | Snapshot times in seconds |
| `solid_fraction` | `[T, *grid]` | Eulerian solid volume fraction |
| `pressure_drop` | `[T]` | Aligned pressure drop, preferably Pa |
| `bed_height` | `[T]` | Aligned bed height or documented normalized height |
| `params` | `[P]` | Static operating-condition features |
| `param_names` | `[P]` | Names in exactly the same order for all cases |
| `cell_volumes` | `[*grid]` | Physical cell volumes; zero outside the domain |
| `solid_volume` | `[T]` | Reference solid volume used by the conservation loss |

## Rules that prevent misleading evaluation

- Keep units identical across all cases and document them in the dataset card.
- Split complete cases before fitting POD, normalization, or any learned preprocessing.
- Do not randomly split adjacent snapshots; this leaks nearly identical flow states.
- Exclude non-converged CFD-DEM runs before model training.
- Mark start-up and quasi-steady regimes if both are included.
- If solids enter or leave the domain, compute time-dependent `solid_volume` from the
  integrated inlet/outlet solid mass flow instead of assuming it is constant.
- A zero `cell_volumes` value is the domain mask. No prediction is scored there.

## Conversion

Use `scripts/convert_arrays_to_case.py` when your solver export has already been converted
to aligned NumPy arrays. For raw particle positions, use
`pg_pod_tcn.coarse_graining.particles_to_solid_fraction` first. The included nearest-cell
binning is conservative but intentionally simple; kernel coarse graining can be added as a
separate, validated adapter.

