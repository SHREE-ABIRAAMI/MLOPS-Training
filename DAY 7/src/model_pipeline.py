from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_classification_model(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> Dict[str, object]:
    """Evaluate classification predictions and return a metrics dictionary."""
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True, zero_division=0)
    confusion = confusion_matrix(y_true, y_pred)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "classification_report": pd.DataFrame(report).transpose(),
        "confusion_matrix": confusion,
    }
    return metrics


def build_roc_data(pipeline, X_test: pd.DataFrame, y_test: np.ndarray, target_names: List[str]) -> Dict[str, object]:
    """Generate ROC curve data for a multiclass Random Forest model."""
    try:
        y_score = pipeline.predict_proba(X_test)
    except Exception:
        return {"roc_curves": [], "roc_auc": {}}

    roc_curves = []
    auc_scores = {}
    for idx, class_name in enumerate(target_names):
        fpr, tpr, _ = roc_curve((y_test == idx).astype(int), y_score[:, idx])
        auc_scores[class_name] = float(roc_auc_score((y_test == idx).astype(int), y_score[:, idx]))
        roc_curves.append({"name": class_name, "fpr": fpr.tolist(), "tpr": tpr.tolist()})
    return {"roc_curves": roc_curves, "roc_auc": auc_scores}


def summarize_classification_report(report_df: pd.DataFrame) -> pd.DataFrame:
    """Convert classification report to a styled DataFrame for display."""
    report_df = report_df.reset_index().rename(columns={"index": "metric"})
    report_df["precision"] = report_df["precision"].round(3)
    report_df["recall"] = report_df["recall"].round(3)
    report_df["f1-score"] = report_df["f1-score"].round(3)
    report_df["support"] = report_df["support"].astype(int)
    return report_df


def aggregate_feature_importances(feature_names: List[str], importances: List[float], raw_features: List[str]) -> Dict[str, float]:
    """Turn transformed feature importances into original feature-level importances."""
    aggregated = {feature: 0.0 for feature in raw_features}
    for name, importance in zip(feature_names, importances):
        for raw in raw_features:
            if name == raw or name.startswith(f"{raw}_"):
                aggregated[raw] += float(importance)
                break
    return aggregated


def build_prediction_results(
    pipeline,
    encoder,
    input_record: pd.DataFrame,
    df: pd.DataFrame,
    feature_columns: List[str],
    raw_importances: Dict[str, float],
) -> Dict[str, object]:
    """Run a prediction and return labeled class, probabilities, and interpretation."""
    predictions = pipeline.predict(input_record)
    probabilities = pipeline.predict_proba(input_record)[0]
    prediction_label = encoder.inverse_transform(predictions)[0]
    confidence = float(np.max(probabilities))
    probability_by_class = dict(zip(encoder.classes_, probabilities.round(4).tolist()))
    explanation = generate_prediction_explanation(
        input_record,
        df,
        feature_columns,
        raw_importances,
        prediction_label,
    )
    return {
        "prediction_label": prediction_label,
        "confidence": confidence,
        "probability_by_class": probability_by_class,
        "insight": explanation["insight"],
        "why": explanation["why"],
        "main_factors": explanation["main_factors"],
        "recommendations": explanation["recommendations"],
    }


def compute_feature_correlations(df: pd.DataFrame, feature_columns: List[str]) -> Dict[str, float]:
    """Compute the correlation between raw features and behavioral risk index."""
    numeric = df[feature_columns].select_dtypes(include=[np.number])
    if "digital_behavioral_risk_index" not in df.columns:
        return {}
    corr = numeric.corrwith(df["digital_behavioral_risk_index"]).to_dict()
    return corr


def humanize_feature_name(name: str) -> str:
    """Convert dataset feature names to human-readable labels."""
    mapping = {
        "daily_usage_hours": "daily screen time",
        "night_usage_hours": "night usage duration",
        "app_switch_frequency": "app switching frequency",
        "notification_check_frequency": "notification checking behavior",
        "short_video_consumption_rate": "short video consumption",
        "average_session_duration": "session duration",
        "dopamine_trigger_score": "dopamine trigger sensitivity",
        "emotional_dependency_score": "emotional dependency",
        "social_validation_dependency": "social validation dependency",
        "impulsive_usage_score": "impulsive usage",
        "emotional_instability_score": "emotional instability",
        "cognitive_fatigue_score": "cognitive fatigue",
        "loneliness_indicator_score": "loneliness indicator",
        "anxiety_level": "anxiety level",
        "attention_fragmentation_index": "attention fragmentation",
        "concentration_score": "concentration ability",
        "multitasking_dependency": "multitasking dependency",
        "content_switching_rate": "content switching rate",
        "attention_span_estimate": "attention span",
        "sleep_hours": "sleep duration",
        "sleep_disruption_score": "sleep disruption",
        "fatigue_after_usage": "post-usage fatigue",
        "morning_tiredness_score": "morning tiredness",
        "preferred_content_type": "preferred content type",
        "negativity_exposure_score": "negativity exposure",
        "overstimulation_index": "overstimulation index",
        "rage_content_exposure": "rage content exposure",
        "emotional_trigger_sensitivity": "emotional trigger sensitivity",
        "toxicity_score": "toxicity signal",
        "algorithmic_influence_score": "algorithmic influence",
        "behavioral_risk_score": "behavioral risk score",
        "digital_behavioral_risk_index": "behavioral risk index",
    }
    return mapping.get(name, name.replace("_", " "))


def generate_recommendations(prediction_label: str) -> List[str]:
    """Return a recommendation list based on the predicted risk level."""
    if prediction_label == "High":
        return [
            "Reduce total screen time and limit evening sessions.",
            "Improve your sleep schedule and avoid late-night usage.",
            "Enable usage limits and notification controls.",
            "Schedule digital detox periods and focus on offline recovery.",
        ]
    if prediction_label == "Moderate":
        return [
            "Monitor your usage patterns and set healthy digital boundaries.",
            "Balance social media time with offline tasks and rest.",
            "Review notification habits and reduce unnecessary alerts.",
        ]
    return [
        "Maintain your healthy routines and continue mindful usage.",
        "Stay aware of screen time and preserve your current balance.",
        "Keep positive content habits and prioritize sleep quality.",
    ]


def generate_prediction_explanation(
    input_record: pd.DataFrame,
    df: pd.DataFrame,
    feature_columns: List[str],
    raw_importances: Dict[str, float],
    prediction_label: str,
) -> Dict[str, object]:
    """Build the explanation and recommendation details for a prediction."""
    correlation = compute_feature_correlations(df, feature_columns)
    medians = df[feature_columns].select_dtypes(include=[np.number]).median()
    sorted_features = sorted(raw_importances.items(), key=lambda item: item[1], reverse=True)
    factors = []

    for feature, importance in sorted_features:
        if feature not in input_record.columns:
            continue
        value = input_record.iloc[0][feature]
        human_name = humanize_feature_name(feature)
        if pd.api.types.is_numeric_dtype(df[feature].dtype):
            median_value = medians[feature]
            direction = correlation.get(feature, 0.0)
            if direction > 0 and value > median_value:
                factors.append(f"High {human_name} compared to the cohort median.")
            elif direction < 0 and value < median_value:
                factors.append(f"Low {human_name} compared to the cohort median.")
            elif abs(value - median_value) / max(abs(median_value), 1) > 0.15:
                factors.append(f"Notable {human_name} level at {value}.")
        else:
            factors.append(f"Preferred {human_name}: {value}.")
        if len(factors) == 4:
            break

    if not factors:
        top_feature = sorted_features[0][0]
        factors.append(f"Key signal from {humanize_feature_name(top_feature)}.")

    insight = (
        f"The model identified the strongest behavioral signals from your profile and assigned a {prediction_label.upper()} risk classification."
    )
    why = (
        f"The Random Forest classifier flags the profile as {prediction_label.upper()} RISK based on the highest-impact behavioral and usage signals."
    )
    recommendations = generate_recommendations(prediction_label)
    return {
        "insight": insight,
        "why": why,
        "main_factors": factors,
        "recommendations": recommendations,
    }
