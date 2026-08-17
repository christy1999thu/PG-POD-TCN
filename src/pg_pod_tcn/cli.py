from __future__ import annotations

import argparse
import json
from pathlib import Path

from pg_pod_tcn.config import load_config
from pg_pod_tcn.evaluation import evaluate_experiment
from pg_pod_tcn.inference import predict_case_file
from pg_pod_tcn.synthetic import generate_synthetic_dataset
from pg_pod_tcn.training import train_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pg-pod-tcn",
        description="Physics-guided POD-TCN surrogate modeling for CFD-DEM",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate the synthetic demonstration data")
    generate.add_argument("--config", default="configs/demo.yaml")
    generate.add_argument("--overwrite", action="store_true")

    train = subparsers.add_parser("train", help="Fit preprocessing, OOD detector, and model ensemble")
    train.add_argument("--config", default="configs/demo.yaml")

    evaluate = subparsers.add_parser("evaluate", help="Evaluate an existing experiment")
    evaluate.add_argument("--config", default="configs/demo.yaml")
    evaluate.add_argument("--split", default="test", choices=["train", "validation", "test"])

    demo = subparsers.add_parser("demo", help="Generate, train, and evaluate end to end")
    demo.add_argument("--config", default="configs/demo.yaml")
    demo.add_argument("--overwrite", action="store_true")

    predict = subparsers.add_parser("predict", help="Run a trained ensemble on one .npz case")
    predict.add_argument("case")
    predict.add_argument("--config", default="configs/demo.yaml")
    predict.add_argument("--output")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.command == "generate":
        paths = _generate_from_config(config, overwrite=args.overwrite)
        print(json.dumps({"generated": len(paths), "root": config["data"]["root"]}, indent=2))
    elif args.command == "train":
        train_experiment(config)
        print(f"Artifacts written to {Path(config['output']['root']).resolve()}")
    elif args.command == "evaluate":
        print(json.dumps(evaluate_experiment(config, args.split), indent=2, ensure_ascii=False))
    elif args.command == "demo":
        _generate_from_config(config, overwrite=args.overwrite)
        train_experiment(config)
        print(json.dumps(evaluate_experiment(config, "test"), indent=2, ensure_ascii=False))
    elif args.command == "predict":
        destination = predict_case_file(config, args.case, args.output)
        print(f"Prediction written to {destination.resolve()}")


def _generate_from_config(config: dict, overwrite: bool = False):
    options = dict(config["data"].get("synthetic", {}))
    if not options:
        raise ValueError("The selected configuration has no data.synthetic section")
    return generate_synthetic_dataset(
        root=config["data"]["root"],
        overwrite=overwrite,
        **options,
    )


if __name__ == "__main__":
    main()

