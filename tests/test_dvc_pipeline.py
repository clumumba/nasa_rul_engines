from pathlib import Path

import yaml


def test_dvc_pipeline_has_end_to_end_stages_without_overlapping_files():
    pipeline_path = Path("dvc.yaml")
    pipeline = yaml.safe_load(pipeline_path.read_text(encoding="utf-8"))
    stages = pipeline["stages"]

    assert list(stages) == ["ingest_data", "train_model", "serve_model"]

    for stage in stages.values():
        dependencies = set(stage.get("deps", []))
        outputs = set(stage.get("outs", []))
        assert not dependencies & outputs

    assert "data/processed/fd001_train.csv" in stages["ingest_data"]["outs"]
    assert "artifacts/fd001_model.joblib" in stages["train_model"]["outs"]
    assert "artifacts/serving_manifest.json" in stages["serve_model"]["outs"]
