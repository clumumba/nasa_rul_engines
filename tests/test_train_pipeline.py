from src.data.load import DATASETS
from src.pipelines import train_pipeline


def test_run_all_training_dispatches_every_dataset(monkeypatch):
    calls = []

    def fake_run_training(dataset, config_path):
        calls.append((dataset, config_path))
        return {"dataset": dataset}

    monkeypatch.setattr(train_pipeline, "run_training", fake_run_training)

    results = train_pipeline.run_all_training("configs/test.yaml")

    assert list(results) == DATASETS
    assert calls == [
        (dataset, "configs/test.yaml")
        for dataset in DATASETS
    ]
