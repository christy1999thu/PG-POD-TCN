from pg_pod_tcn.evaluation import evaluate_experiment
from pg_pod_tcn.synthetic import generate_synthetic_dataset
from pg_pod_tcn.training import train_experiment


def test_one_epoch_end_to_end(tmp_path):
    data_root = tmp_path / "data"
    output_root = tmp_path / "outputs"
    generate_synthetic_dataset(
        data_root, cases=6, timesteps=28, height=8, width=6, dt=0.02, seed=3
    )
    config = {
        "project": {"name": "smoke", "seed": 3},
        "data": {
            "root": str(data_root),
            "history": 5,
            "horizon": 2,
            "stride": 3,
            "train_fraction": 0.6,
            "val_fraction": 0.2,
            "split_seed": 3,
            "pod_energy": 0.99,
            "pod_max_modes": 6,
        },
        "model": {
            "type": "tcn",
            "channels": 12,
            "levels": 2,
            "kernel_size": 3,
            "dropout": 0.0,
        },
        "loss": {"coefficient": 1, "field": 0.5, "mass": 0.2, "bounds": 0.1, "macro": 0.5},
        "training": {
            "device": "cpu",
            "epochs": 1,
            "batch_size": 8,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "patience": 1,
            "gradient_clip": 1.0,
            "num_workers": 0,
            "seeds": [3],
        },
        "ood": {"quantile": 0.99, "regularization": 0.001},
        "output": {"root": str(output_root)},
    }
    train_experiment(config)
    metrics = evaluate_experiment(config, "test")
    assert (output_root / "checkpoint_seed_3.pt").exists()
    assert (output_root / "metrics_test.json").exists()
    assert metrics["windows"] > 0
    assert metrics["field_nrmse"] >= 0

