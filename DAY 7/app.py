from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_loader import (
    build_training_pipeline,
    get_dynamic_input_schema,
    get_feature_columns,
    load_raw_dataset,
    summarize_dataset,
)
from src.model_pipeline import (
    aggregate_feature_importances,
    build_prediction_results,
    build_roc_data,
    evaluate_classification_model,
    summarize_classification_report,
)
from src.ui_elements import (
    inject_custom_css,
    render_about_page,
    render_ai_insights,
    render_dashboard_page,
    render_data_visualization,
    render_dataset_explorer,
    render_feature_importance,
    render_landing_page,
    render_model_performance,
    render_prediction_center,
    render_sidebar,
    set_page_config,
)


def main() -> None:
    set_page_config()
    inject_custom_css()

    try:
        df = load_raw_dataset()
    except Exception as error:
        st.error(f"Unable to load dataset: {error}")
        return

    if df.empty:
        st.error("The dataset is empty. Please provide a valid dataset to proceed.")
        return

    dataset_summary = summarize_dataset(df)

    try:
        (
            pipeline,
            label_encoder,
            X_train,
            X_test,
            y_train,
            y_test,
            feature_names,
            feature_columns,
        ) = build_training_pipeline(df)
    except Exception as error:
        st.error(f"Model training failed: {error}")
        return

    y_pred = pipeline.predict(X_test)
    class_labels = label_encoder.classes_.tolist()
    metrics = evaluate_classification_model(y_test, y_pred, class_labels)
    metrics["classification_report"] = summarize_classification_report(metrics["classification_report"])

    roc_data = build_roc_data(pipeline, X_test, y_test, class_labels)
    prediction_schema = get_dynamic_input_schema(df)
    raw_feature_importances = aggregate_feature_importances(
        feature_names,
        pipeline.named_steps["classifier"].feature_importances_,
        feature_columns,
    )

    menu_items = [
        "🏠 Landing Page",
        "📊 Dashboard",
        "🗂️ Dataset Explorer",
        "📈 Data Visualization",
        "🔮 Prediction Center",
        "📉 Model Performance",
        "⭐ Feature Importance",
        "🧠 AI Insights",
        "ℹ️ About Project",
    ]
    if "menu_radio" not in st.session_state:
        st.session_state.menu_radio = menu_items[0]
    selected_page = render_sidebar(menu_items)

    if selected_page == "🏠 Landing Page":
        render_landing_page(dataset_summary)
    elif selected_page == "📊 Dashboard":
        render_dashboard_page(df, dataset_summary, metrics)
    elif selected_page == "🗂️ Dataset Explorer":
        render_dataset_explorer(df)
    elif selected_page == "📈 Data Visualization":
        render_data_visualization(df)
    elif selected_page == "🔮 Prediction Center":
        render_prediction_center(
            pipeline,
            label_encoder,
            prediction_schema,
            feature_columns,
            raw_feature_importances,
            df,
        )
    elif selected_page == "📉 Model Performance":
        render_model_performance(metrics, roc_data)
    elif selected_page == "⭐ Feature Importance":
        render_feature_importance(raw_feature_importances)
    elif selected_page == "🧠 AI Insights":
        render_ai_insights(dataset_summary, df, feature_columns, raw_feature_importances)
    elif selected_page == "ℹ️ About Project":
        render_about_page()
    else:
        render_landing_page(dataset_summary)


if __name__ == "__main__":
    main()
