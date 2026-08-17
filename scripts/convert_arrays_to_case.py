#!/usr/bin/env python3
"""Convert aligned NumPy arrays into the repository's validated case format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pg_pod_tcn.data import CaseData, save_case


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--solid-fraction", required=True, help=".npy array [time, *grid]")
    parser.add_argument("--time", required=True, help=".npy array [time]")
    parser.add_argument("--pressure-drop", required=True, help=".npy array [time]")
    parser.add_argument("--bed-height", required=True, help=".npy array [time]")
    parser.add_argument("--cell-volumes", required=True, help=".npy array [*grid]")
    parser.add_argument(
        "--params-json",
        required=True,
        help='JSON mapping in stable order, e.g. {"velocity_ratio": 1.2, "diameter_mm": 0.8}',
    )
    parser.add_argument("--solid-volume", help="Optional .npy array [time]")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    params_mapping = json.loads(args.params_json)
    if not isinstance(params_mapping, dict) or not params_mapping:
        raise ValueError("--params-json must contain a non-empty JSON object")
    fields = np.load(args.solid_fraction)
    cell_volumes = np.load(args.cell_volumes)
    if args.solid_volume:
        solid_volume = np.load(args.solid_volume)
    else:
        solid_volume = np.sum(fields * cell_volumes, axis=tuple(range(1, fields.ndim)))
    case = CaseData(
        case_id=args.case_id,
        time=np.load(args.time),
        solid_fraction=fields,
        pressure_drop=np.load(args.pressure_drop),
        bed_height=np.load(args.bed_height),
        params=np.asarray(list(params_mapping.values()), dtype=np.float32),
        param_names=tuple(params_mapping.keys()),
        cell_volumes=cell_volumes,
        solid_volume=solid_volume,
    )
    save_case(case, Path(args.output))
    print(f"Wrote {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()

