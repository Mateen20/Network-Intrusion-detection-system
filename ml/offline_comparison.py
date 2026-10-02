"""Offline-only comparison of Random Forest and XGBoost.

Run with ``python -m ml.offline_comparison``. This module is not imported by
the live or simulation detection paths. XGBoost artifacts are deliberately
stored separately from the production Random Forest artifacts.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from .trainer import FEATURE_NAMES, MODEL_DIR, MODEL_REVISION, NIDSTrainer, generate_synthetic_dataset


OFFLINE_MODEL_DIR = os.path.join(MODEL_DIR, "offline_xgboost")
SPLIT_SEED = 42
SYNTHETIC_SEED = 42
XGBOOST_MODEL_FILENAME = "xgboost_model.json"
METADATA_FILENAME = "metadata.json"


def _generate_reproducible_dataset(samples_per_class: int) -> tuple[pd.DataFrame, pd.Series]:
    random_state = np.random.get_state()
    try:
        np.random.seed(SYNTHETIC_SEED)
        features, labels = generate_synthetic_dataset(samples_per_class)
    finally:
        np.random.set_state(random_state)
    return features, labels


def _production_trainer(model_dir: str) -> NIDSTrainer:
    trainer = NIDSTrainer(model_dir=model_dir)
    if not trainer.load():
        raise RuntimeError(f"Production Random Forest artifacts are unavailable or invalid in {model_dir}")
    if trainer.scaler.n_features_in_ != len(FEATURE_NAMES):
        raise ValueError("Production scaler does not match the canonical 24-feature schema")
    if trainer.label_enc is None or trainer.rf_model is None:
        raise ValueError("Production Random Forest artifacts are incomplete")
    if trainer.rf_model.n_features_in_ != len(FEATURE_NAMES):
        raise ValueError("Production Random Forest does not match the canonical 24-feature schema")
    return trainer


def _weighted_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
    class_ids = np.arange(len(class_names))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=class_ids, average="weighted", zero_division=0,
    )
    class_precision, class_recall, class_f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=class_ids, average=None, zero_division=0,
    )
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "per_class": {
            name: {
                "precision": float(class_precision[index]),
                "recall": float(class_recall[index]),
                "f1": float(class_f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(class_names)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=class_ids).tolist(),
    }


def _test_indices_hash(indices: np.ndarray) -> str:
    ordered_indices = np.asarray(indices, dtype="<i8")
    return hashlib.sha256(ordered_indices.tobytes()).hexdigest()


def _assert_offline_destination(model_dir: str, artifact_dir: str) -> None:
    production_dir = Path(model_dir).resolve()
    destination = Path(artifact_dir).resolve()
    if destination == production_dir:
        raise ValueError("Offline XGBoost artifacts must use a separate directory from production artifacts")


def run_offline_comparison(
    model_dir: str = MODEL_DIR,
    artifact_dir: str = OFFLINE_MODEL_DIR,
    samples_per_class: int | None = None,
) -> dict:
    """Fit offline model copies on one split, compare them, and save XGBoost only."""
    _assert_offline_destination(model_dir, artifact_dir)
    trainer = _production_trainer(model_dir)
    class_names = [str(name) for name in trainer.label_enc.classes_]

    scaler_sample_count = int(np.asarray(trainer.scaler.n_samples_seen_).max())
    if samples_per_class is None:
        if scaler_sample_count % len(class_names):
            raise ValueError("Cannot infer synthetic samples per class from the production scaler")
        samples_per_class = scaler_sample_count // len(class_names)

    features, labels = _generate_reproducible_dataset(samples_per_class)
    if list(features.columns) != FEATURE_NAMES:
        raise ValueError("Training data does not match the canonical 24-feature order")
    if scaler_sample_count != len(features):
        raise ValueError(
            "Generated dataset size does not match the production scaler's training sample count "
            f"({len(features)} != {scaler_sample_count}); refusing an inconsistent comparison"
        )
    if set(labels.unique()) != set(class_names):
        raise ValueError("Generated dataset classes do not match the production class set")

    y = trainer.label_enc.transform(labels)
    scaled = pd.DataFrame(
        trainer.scaler.transform(features[FEATURE_NAMES]),
        columns=FEATURE_NAMES,
    )
    if list(scaled.columns) != FEATURE_NAMES:
        raise ValueError("Preprocessed data changed the canonical feature order")

    all_indices = np.arange(len(features))
    train_indices, test_indices = train_test_split(
        all_indices,
        test_size=0.2,
        random_state=SPLIT_SEED,
        stratify=y,
    )
    X_train = scaled.iloc[train_indices]
    X_test = scaled.iloc[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    random_forest = clone(trainer.rf_model)
    start = time.perf_counter()
    random_forest.fit(X_train, y_train)
    random_forest_training_seconds = time.perf_counter() - start

    try:
        import xgboost
        from xgboost import XGBClassifier
    except ImportError as error:
        raise RuntimeError("XGBoost is required; install dependencies with `pip install -r requirements.txt`") from error

    xgboost_model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(class_names),
        eval_metric="mlogloss",
        tree_method="hist",
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=SPLIT_SEED,
        n_jobs=-1,
        verbosity=0,
    )
    start = time.perf_counter()
    xgboost_model.fit(X_train, y_train)
    xgboost_training_seconds = time.perf_counter() - start

    if list(random_forest.feature_names_in_) != FEATURE_NAMES:
        raise ValueError("Offline Random Forest feature order differs from the canonical schema")
    if list(xgboost_model.feature_names_in_) != FEATURE_NAMES:
        raise ValueError("XGBoost feature order differs from the canonical schema")

    start = time.perf_counter()
    rf_predictions = random_forest.predict(X_test)
    random_forest_prediction_seconds = time.perf_counter() - start
    start = time.perf_counter()
    xgb_predictions = xgboost_model.predict(X_test)
    xgboost_prediction_seconds = time.perf_counter() - start

    declared_class_ids = set(range(len(class_names)))
    if not set(np.asarray(rf_predictions, dtype=int)).issubset(declared_class_ids):
        raise ValueError("Random Forest predicted a class outside the declared class set")
    if not set(np.asarray(xgb_predictions, dtype=int)).issubset(declared_class_ids):
        raise ValueError("XGBoost predicted a class outside the declared class set")

    test_indices_digest = _test_indices_hash(test_indices)
    held_out = {
        "sample_count": len(test_indices),
        "indices_sha256": test_indices_digest,
    }
    results = {
        "Random Forest": {
            **_weighted_metrics(y_test, rf_predictions, class_names),
            "feature_order": FEATURE_NAMES,
            "training_time_seconds": random_forest_training_seconds,
            "prediction_time_seconds": random_forest_prediction_seconds,
            "held_out_test_set": held_out,
        },
        "XGBoost": {
            **_weighted_metrics(y_test, xgb_predictions, class_names),
            "feature_order": FEATURE_NAMES,
            "training_time_seconds": xgboost_training_seconds,
            "prediction_time_seconds": xgboost_prediction_seconds,
            "held_out_test_set": held_out,
        },
    }

    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)
    model_path = artifact_path / XGBOOST_MODEL_FILENAME
    xgboost_model.save_model(model_path)
    scaler_path = artifact_path / "scaler.joblib"
    joblib.dump(trainer.scaler, scaler_path)

    metadata = {
        "model_name": "XGBoost",
        "model_revision": 1,
        "model_type": "xgboost.XGBClassifier",
        "runtime_role": "offline_comparison_only",
        "production_compatible": False,
        "xgboost_version": xgboost.__version__,
        "model_artifact": XGBOOST_MODEL_FILENAME,
        "scaler_artifact": scaler_path.name,
        "features": FEATURE_NAMES,
        "classes": class_names,
        "preprocessing": {
            "scaler": "StandardScaler loaded from the unchanged production artifact",
            "scaler_fit_scope": "full training dataset, matching existing trainer behavior",
            "scaled_feature_order": FEATURE_NAMES,
        },
        "training_dataset": {
            "source": "ml.trainer.generate_synthetic_dataset",
            "samples_per_class": samples_per_class,
            "sample_count": len(features),
            "historical_production_sample_count": scaler_sample_count,
            "seed": SYNTHETIC_SEED,
        },
        "train_test_split": {
            "test_size": 0.2,
            "stratified": True,
            "random_seed": SPLIT_SEED,
            "train_sample_count": len(train_indices),
            "test_sample_count": len(test_indices),
            "test_indices_sha256": test_indices_digest,
        },
        "random_seed": SPLIT_SEED,
        "random_forest_comparison": {
            "method": "offline refit using a clone of the production Random Forest parameters",
            "production_artifact_modified": False,
        },
    }
    with (artifact_path / METADATA_FILENAME).open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
    with (artifact_path / "comparison_results.json").open("w", encoding="utf-8") as results_file:
        json.dump(results, results_file, indent=2)

    return {
        "xgboost_version": xgboost.__version__,
        "model_artifact": str(model_path),
        "metadata_artifact": str(artifact_path / METADATA_FILENAME),
        "comparison_results": results,
    }


def load_offline_xgboost(artifact_dir: str = OFFLINE_MODEL_DIR):
    """Load a validated offline XGBoost artifact, never as a production model."""
    import xgboost
    from xgboost import XGBClassifier

    artifact_path = Path(artifact_dir)
    with (artifact_path / METADATA_FILENAME).open(encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    if (
        metadata.get("model_name") != "XGBoost"
        or metadata.get("runtime_role") != "offline_comparison_only"
        or metadata.get("production_compatible") is not False
        or metadata.get("features") != FEATURE_NAMES
    ):
        raise ValueError("Artifact metadata does not identify a valid offline-only XGBoost model")
    if metadata.get("xgboost_version") != xgboost.__version__:
        raise ValueError("Installed XGBoost version does not match the artifact metadata")

    model = XGBClassifier()
    model.load_model(artifact_path / metadata["model_artifact"])
    scaler = joblib.load(artifact_path / metadata["scaler_artifact"])
    if scaler.n_features_in_ != len(FEATURE_NAMES):
        raise ValueError("Offline XGBoost scaler does not match the canonical feature schema")
    if list(model.get_booster().feature_names or []) != FEATURE_NAMES:
        raise ValueError("Offline XGBoost artifact feature order is invalid")
    return model, scaler, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline-only Random Forest and XGBoost comparison")
    parser.add_argument("--model-dir", default=MODEL_DIR, help="Existing production RF artifact directory")
    parser.add_argument("--artifact-dir", default=OFFLINE_MODEL_DIR, help="Separate offline XGBoost artifact directory")
    parser.add_argument("--samples-per-class", type=int, default=None, help="Synthetic samples per class; defaults to production scaler size")
    args = parser.parse_args()
    report = run_offline_comparison(args.model_dir, args.artifact_dir, args.samples_per_class)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()