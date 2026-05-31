from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).resolve().parents[1] / "digital_mirror_research_dataset_870_rows.csv"
MODEL_PATH = Path(__file__).resolve().parents[1] / "digital_mirror_rf_model.joblib"
TARGET_COLUMN = "addiction_risk_level"
IDENTIFIER_COLUMNS = ["participant_id"]
TEXT_COLUMNS = ["behavioral_text"]
CATEGORICAL_COLUMNS = ["gender", "occupation", "preferred_content_type", "emotional_tone"]


def load_raw_dataset() -> pd.DataFrame:
    """Load the dataset from the project folder and return a pandas DataFrame."""
    try:
        df = pd.read_csv(DATA_PATH)
        return df
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Unable to locate dataset at {DATA_PATH}") from error


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return the list of feature columns used for training and prediction."""
    excluded = set(IDENTIFIER_COLUMNS + TEXT_COLUMNS + [TARGET_COLUMN])
    return [col for col in df.columns if col not in excluded]


def build_preprocessor(feature_columns: List[str]) -> ColumnTransformer:
    """Create a preprocessing pipeline for numeric scaling and categorical encoding."""
    sample_frame = load_raw_dataset()
    numeric_features = [col for col in feature_columns if np.issubdtype(sample_frame[col].dtype, np.number)]
    categorical_features = [col for col in feature_columns if col in CATEGORICAL_COLUMNS]

    transformers = []
    if numeric_features:
        transformers.append(("numeric", StandardScaler(), numeric_features))
    if categorical_features:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            )
        )

    transformer = ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)
    return transformer


def encode_target(y: pd.Series) -> Tuple[np.ndarray, LabelEncoder]:
    """Encode the categorical target labels into numeric classes."""
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y.astype(str))
    return y_encoded, encoder


def save_trained_pipeline(pipeline: Pipeline, encoder: LabelEncoder, feature_names: List[str], feature_columns: List[str]) -> None:
    """Save the trained pipeline and encoder to disk."""
    joblib.dump(
        {
            "pipeline": pipeline,
            "encoder": encoder,
            "feature_names": feature_names,
            "feature_columns": feature_columns,
        },
        MODEL_PATH,
    )


def load_trained_pipeline() -> Optional[Tuple[Pipeline, LabelEncoder, List[str], List[str]]]:
    """Load a saved pipeline and encoder if a joblib file exists."""
    if not MODEL_PATH.exists():
        return None
    loaded = joblib.load(MODEL_PATH)
    return loaded["pipeline"], loaded["encoder"], loaded["feature_names"], loaded["feature_columns"]


def get_dynamic_input_schema(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Build a dynamic schema for prediction form fields from the dataset."""
    feature_columns = get_feature_columns(df)
    schema = []
    for column in feature_columns:
        dtype = df[column].dtype
        if column in CATEGORICAL_COLUMNS:
            options = sorted(df[column].dropna().unique().tolist())
            schema.append(
                {
                    "name": column,
                    "type": "select",
                    "label": column.replace("_", " ").title(),
                    "options": options,
                    "default": options[0] if options else "",
                }
            )
        elif np.issubdtype(dtype, np.integer):
            schema.append(
                {
                    "name": column,
                    "type": "int",
                    "label": column.replace("_", " ").title(),
                    "min": int(df[column].min()),
                    "max": int(df[column].max()),
                    "step": 1,
                    "default": int(df[column].median()),
                }
            )
        else:
            schema.append(
                {
                    "name": column,
                    "type": "float",
                    "label": column.replace("_", " ").title(),
                    "min": float(np.nanmin(df[column].values)),
                    "max": float(np.nanmax(df[column].values)),
                    "step": 0.1,
                    "default": float(df[column].median()),
                }
            )
    return schema


def build_training_pipeline(
    df: pd.DataFrame = None,
    force_retrain: bool = False,
) -> Tuple[Pipeline, LabelEncoder, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """Train the Random Forest pipeline and return the fitted pipeline with data splits."""
    df = df if df is not None else load_raw_dataset()
    feature_columns = get_feature_columns(df)

    if not force_retrain:
        saved = load_trained_pipeline()
        if saved is not None:
            pipeline, target_encoder, feature_names, saved_columns = saved
            if saved_columns == feature_columns:
                X = df[feature_columns]
                y, _ = encode_target(df[TARGET_COLUMN])
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=0.20,
                    random_state=42,
                    stratify=y,
                )
                return pipeline, target_encoder, X_train, X_test, y_train, y_test, feature_names, feature_columns

    X = df[feature_columns]
    y, target_encoder = encode_target(df[TARGET_COLUMN])
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    preprocessor = build_preprocessor(feature_columns)
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=14,
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)

    feature_names = get_feature_names(pipeline, feature_columns)
    try:
        save_trained_pipeline(pipeline, target_encoder, feature_names, feature_columns)
    except Exception:
        pass

    return pipeline, target_encoder, X_train, X_test, y_train, y_test, feature_names, feature_columns


def _resolve_transformer(transformer):
    if hasattr(transformer, "named_steps"):
        for step in transformer.named_steps.values():
            if hasattr(step, "categories_"):
                return step
    return transformer


def get_feature_names(pipeline: Pipeline, feature_columns: List[str]) -> List[str]:
    """Extract transformed feature names from the preprocessing pipeline."""
    preprocessor = pipeline.named_steps["preprocessor"]
    transformed_names = []
    for name, transformer, columns in preprocessor.transformers_:
        if transformer is None or columns == "remainder":
            continue
        if name == "numeric":
            transformed_names.extend(columns)
        elif name == "categorical":
            fitted_transformer = _resolve_transformer(transformer)
            categories = getattr(fitted_transformer, "categories_", None)
            if categories is None:
                continue
            for col, cats in zip(columns, categories):
                transformed_names.extend([f"{col}_{cat}" for cat in cats])
    return transformed_names


def summarize_dataset(df: pd.DataFrame) -> Dict[str, object]:
    """Generate a compact dataset summary for dashboard and insights."""
    feature_columns = get_feature_columns(df)
    summary = {
        "name": DATA_PATH.name,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "feature_count": len(feature_columns),
        "feature_names": df.columns.tolist(),
        "data_types": df.dtypes.astype(str).to_dict(),
        "missing_values": int(df.isna().sum().sum()),
        "missing_by_column": df.isna().sum().to_dict(),
        "duplicate_count": int(df.duplicated().sum()),
        "target_column": TARGET_COLUMN,
        "class_distribution": df[TARGET_COLUMN].value_counts().to_dict(),
        "top_correlations": get_top_correlations(df),
    }
    return summary


def get_top_correlations(df: pd.DataFrame, top_n: int = 12) -> Dict[str, float]:
    """Return the strongest numeric correlations to the behavioral risk index."""
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr()["digital_behavioral_risk_index"].abs().sort_values(ascending=False)
    return corr.iloc[1 : min(top_n + 1, len(corr))].to_dict()


def create_prediction_record(input_data: Dict[str, object]) -> pd.DataFrame:
    """Convert a dictionary of inputs into a DataFrame for prediction."""
    return pd.DataFrame([input_data])


def compute_data_quality(df: pd.DataFrame) -> Dict[str, int]:
    """Compute missing values and duplicate counts for dataset monitoring."""
    return {
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_records": int(df.duplicated().sum()),
    }
