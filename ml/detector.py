"""
ml/detector.py — Two-model cascade inference engine.

Stage 1 → Classic RF (24 features, 10 classes)  — 99.04% accuracy
Stage 2 → Modern Threat Detector (8 features, 4 classes) — 100% accuracy

The modern detector ONLY overrides the classic result when:
  • It detects ransomware / encrypted_c2 / dga  (not normal)
  • Confidence >= 0.75
  • Classic result is not already CRITICAL with high confidence
"""

import math

import numpy as np
import pandas as pd
from .trainer import NIDSTrainer, FEATURE_NAMES, SEVERITY_MAP


# Confidence gates are kept here so prediction semantics and alert policy stay
# explicit and can be tuned without changing the trained model.
CONFIDENCE_THRESHOLDS = {
    "critical": 0.90,
    "high": 0.75,
    "medium": 0.60,
    "uncertain": 0.60,
}


def calibrate_severity(label: str, confidence: float,
                       is_anomaly: bool = False,
                       anomaly_score: float = 0.0) -> tuple[str, str]:
    """Return the effective class and evidence-gated alert severity."""
    effective_label = label
    if label == "normal":
        severity = (
            "UNCERTAIN"
            if is_anomaly and anomaly_score < -0.15
            else "CLEAN"
        )
    else:
        predicted_severity = SEVERITY_MAP.get(label, "CLEAN")
        if confidence < CONFIDENCE_THRESHOLDS["uncertain"]:
            severity = "UNCERTAIN"
        elif predicted_severity == "CRITICAL" and confidence < CONFIDENCE_THRESHOLDS["critical"]:
            severity = (
                "HIGH"
                if confidence >= CONFIDENCE_THRESHOLDS["high"]
                else "MEDIUM"
            )
        elif predicted_severity == "HIGH" and confidence < CONFIDENCE_THRESHOLDS["high"]:
            severity = "MEDIUM"
        else:
            severity = predicted_severity
    return effective_label, severity

try:
    from .modern_detector import ModernThreatDetector, MODERN_FEATURES
    _MODERN_AVAILABLE = True
except ImportError:
    _MODERN_AVAILABLE = False


class NIDSDetector:
    def __init__(self, trainer: NIDSTrainer,
                 modern=None):
        self.trainer   = trainer
        self.rf        = trainer.rf_model
        self.iso       = trainer.iso_model
        self.scaler    = trainer.scaler
        self.label_enc = trainer.label_enc
        self.modern    = modern   # ModernThreatDetector instance or None
        if self.rf is not None:
            self.rf.n_jobs = 1
        if self.iso is not None:
            self.iso.n_jobs = 1
        if self.modern is not None and self.modern.rf_model is not None:
            self.modern.rf_model.n_jobs = 1

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(self, features: dict) -> dict:
        """Run full cascade. Returns detection result dict."""

        # Stage 1: Classic 24-feature model
        result = self._classic(features)

        # Stage 2: Modern 8-feature detector
        if self.modern is not None:
            modern_result = self.modern.predict(features)
            if modern_result is not None:
                # Don't downgrade a high-confidence CRITICAL classic result
                classic_is_certain = (
                    result["severity"] == "CRITICAL" and
                    result["confidence"] >= 0.85
                )
                if not classic_is_certain:
                    modern_result["label"], modern_result["severity"] = calibrate_severity(
                        modern_result["label"], modern_result["confidence"],
                        is_anomaly=modern_result.get("anomaly", False),
                        anomaly_score=modern_result.get("anomaly_score", 0.0),
                    )
                    modern_result["anomaly"]       = result.get("anomaly", False)
                    modern_result["anomaly_score"] = result.get("anomaly_score", 0.0)
                    return modern_result

        return result

    def predict_batch(self, df: pd.DataFrame) -> list:
        return [self.predict(row.to_dict()) for _, row in df.iterrows()]

    # ── Classic 24-feature model ──────────────────────────────────────────────

    @staticmethod
    def _validate_feature_dict(features: dict, feature_names: list[str], context: str = "live feature vector") -> pd.DataFrame:
        if not isinstance(features, dict):
            raise ValueError(f"{context} must be a dict, got {type(features).__name__}")

        missing = [name for name in feature_names if name not in features]
        if missing:
            raise ValueError(f"{context} missing required feature(s): {missing}")

        row: dict[str, float] = {}
        for name in feature_names:
            value = features[name]
            if value is None:
                raise ValueError(f"{context} feature '{name}' is None; this should be calculated or marked unavailable, not silently set to zero.")
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{context} feature '{name}' is not numeric: {value!r}") from exc
            if not math.isfinite(numeric):
                raise ValueError(f"{context} feature '{name}' is NaN or infinite: {numeric!r}")
            row[name] = numeric

        return pd.DataFrame([row], columns=feature_names)

    def _classic(self, features: dict) -> dict:
        classic_24 = FEATURE_NAMES[:24]
        X = self._validate_feature_dict(features, classic_24, context="classic model input")
        X_sc = self.scaler.transform(X)

        proba  = self.rf.predict_proba(X_sc)[0]
        idx    = int(np.argmax(proba))
        label  = self.label_enc.inverse_transform([idx])[0]
        conf   = float(proba[idx])
        probs  = {
            cls: round(float(p), 4)
            for cls, p in zip(self.label_enc.classes_, proba)
        }

        iso_score  = float(self.iso.decision_function(X_sc)[0])
        is_anomaly = bool(self.iso.predict(X_sc)[0] == -1)

        label, severity = calibrate_severity(
            label, conf, is_anomaly=is_anomaly, anomaly_score=iso_score,
        )

        return {
            "label":         label,
            "severity":      severity,
            "confidence":    round(conf, 4),
            "anomaly":       is_anomaly,
            "anomaly_score": round(iso_score, 4),
            "probabilities": probs,
            "source":        "classic_model",
        }
