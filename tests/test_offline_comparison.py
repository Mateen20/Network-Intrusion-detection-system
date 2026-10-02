import hashlib
import json

import joblib
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ml.offline_comparison import (
    FEATURE_NAMES,
    _generate_reproducible_dataset,
    load_offline_xgboost,
    run_offline_comparison,
)
from ml.trainer import MODEL_REVISION, NIDSTrainer


def _file_hashes(directory):
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.iterdir()
        if path.is_file()
    }


def _create_production_artifacts(directory, samples_per_class=30):
    features, labels = _generate_reproducible_dataset(samples_per_class)
    scaler = StandardScaler().fit(features[FEATURE_NAMES])
    label_encoder = LabelEncoder().fit(labels)
    scaled = scaler.transform(features[FEATURE_NAMES])
    encoded = label_encoder.transform(labels)
    random_forest = RandomForestClassifier(
        n_estimators=5,
        max_depth=20,
        min_samples_split=4,
        random_state=42,
        class_weight="balanced",
        n_jobs=1,
    ).fit(scaled, encoded)
    isolation_forest = IsolationForest(random_state=42).fit(scaled)

    joblib.dump(random_forest, directory / "rf.pkl")
    joblib.dump(isolation_forest, directory / "iso.pkl")
    joblib.dump(scaler, directory / "scaler.pkl")
    joblib.dump(label_encoder, directory / "label_enc.pkl")
    with (directory / "model_metadata.json").open("w", encoding="utf-8") as metadata_file:
        json.dump({"revision": MODEL_REVISION, "features": FEATURE_NAMES}, metadata_file)


def test_offline_xgboost_uses_canonical_schema_shared_split_and_isolated_artifacts(tmp_path):
    production_dir = tmp_path / "production"
    offline_dir = tmp_path / "offline_xgboost"
    production_dir.mkdir()
    _create_production_artifacts(production_dir)
    production_hashes = _file_hashes(production_dir)

    report = run_offline_comparison(
        model_dir=str(production_dir),
        artifact_dir=str(offline_dir),
    )

    assert _file_hashes(production_dir) == production_hashes
    assert report["comparison_results"]["Random Forest"]["held_out_test_set"] == report["comparison_results"]["XGBoost"]["held_out_test_set"]
    assert report["comparison_results"]["Random Forest"]["feature_order"] == FEATURE_NAMES
    assert report["comparison_results"]["XGBoost"]["feature_order"] == FEATURE_NAMES

    model, scaler, metadata = load_offline_xgboost(str(offline_dir))
    trainer = NIDSTrainer(model_dir=str(production_dir))
    assert trainer.load() is True
    assert isinstance(trainer.rf_model, RandomForestClassifier)
    assert model.feature_names_in_.tolist() == FEATURE_NAMES
    assert trainer.scaler.n_features_in_ == scaler.n_features_in_ == len(FEATURE_NAMES)
    assert metadata["features"] == FEATURE_NAMES
    assert metadata["classes"] == trainer.label_enc.classes_.tolist()
    assert metadata["model_name"] == "XGBoost"
    assert metadata["runtime_role"] == "offline_comparison_only"
    assert metadata["production_compatible"] is False
    assert metadata["xgboost_version"] == report["xgboost_version"]

    features, _ = _generate_reproducible_dataset(30)
    encoded_predictions = model.predict(scaler.transform(features[FEATURE_NAMES]))
    assert set(encoded_predictions).issubset(set(range(len(metadata["classes"]))))
    assert set(report["comparison_results"]) == {"Random Forest", "XGBoost"}