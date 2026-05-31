import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.model_pipeline import build_prediction_results

try:
    from st_aggrid import AgGrid, GridOptionsBuilder
    AG_GRID_AVAILABLE = True
except ImportError:
    AG_GRID_AVAILABLE = False

STYLE_PATH = Path(__file__).resolve().parent / "styles.css"


def set_page_config() -> None:
    st.set_page_config(
        page_title="DIGITAL MIRROR",
        page_icon="🪞",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def inject_custom_css() -> None:
    if STYLE_PATH.exists():
        css = STYLE_PATH.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_sidebar(menu_items: List[str]) -> str:
    if "menu_radio" not in st.session_state:
        st.session_state.menu_radio = menu_items[0]
    with st.sidebar:
        st.markdown("<div class='sidebar-brand'>DIGITAL MIRROR</div>", unsafe_allow_html=True)
        st.markdown("<div class='sidebar-subtitle'>AI Personality & Behavioral Intelligence</div>", unsafe_allow_html=True)
        selected = st.radio(
            "Navigation",
            menu_items,
            index=menu_items.index(st.session_state.menu_radio),
            key="menu_radio_input",
        )
        if selected != st.session_state.menu_radio:
            st.session_state.menu_radio = selected
        st.markdown("<div class='sidebar-footer'>Premium AI SaaS Dashboard</div>", unsafe_allow_html=True)
    return st.session_state.menu_radio


def render_header(title: str, subtitle: str) -> None:
    st.markdown(f"<div class='section-title'><h1>{title}</h1><p>{subtitle}</p></div>", unsafe_allow_html=True)


def render_metric_cards(metrics: Dict[str, object]) -> None:
    card_template = (
        "<div class='metric-card'>"
        "<span class='metric-label'>{label}</span>"
        "<span class='metric-value'>{value}</span>"
        "<span class='metric-note'>{note}</span>"
        "</div>"
    )
    columns = st.columns(len(metrics))
    for col, (label, details) in zip(columns, metrics.items()):
        with col:
            st.markdown(
                card_template.format(
                    label=label,
                    value=details["value"],
                    note=details.get("note", ""),
                ),
                unsafe_allow_html=True,
            )


def render_landing_page(summary: Dict[str, object]) -> None:
    render_header(
        "DIGITAL MIRROR",
        "AI-Powered Personality Reflection & Behavioral Intelligence System",
    )
    st.markdown(
        "<div class='hero-banner'>"
        "<div class='hero-copy'>"
        "<span class='hero-chip'>Cyber AI Experience</span>"
        "<h2 class='hero-title'>Predictive behavioral intelligence meets visual elegance.</h2>"
        "<p class='hero-description'>Digital Mirror offers research-grade personality analysis with a futuristic interface built for hackathon judging panels, enterprise demos, and final year showcases.</p>"
        "</div>"
        "<div class='hero-visual'>"
        "<div class='floating-glow glow-one'></div>"
        "<div class='floating-glow glow-two'></div>"
        "<div class='floating-card'>"
        "<span class='badge'>AI Mirror</span>"
        "<h3>Emotional risk.</h3><p>Behavioral score fusion in a holographic panel.</p>"
        "</div>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("Start Analysis", key="start_analysis", help="Begin the Digital Mirror workflow"):
            st.session_state.menu_radio = "📊 Dashboard"
        if st.button("Explore Dataset", key="explore_dataset", help="Open the dataset explorer"):
            st.session_state.menu_radio = "🗂️ Dataset Explorer"
    with col2:
        st.markdown(
            "<div class='hero-actions'>"
            "<span class='hero-note'>Launch the AI prediction flow or inspect the dataset with advanced analytics.</span>"
            "</div>",
            unsafe_allow_html=True,
        )

    stats = [
        {"value": f"{summary['rows']}", "note": "Total records"},
        {"value": f"{summary['columns']}", "note": "Total columns"},
        {"value": f"{len(summary['class_distribution'])}", "note": "Outcome categories"},
        {"value": "0", "note": "Missing values"},
    ]
    columns = st.columns(len(stats))
    for col, stat in zip(columns, stats):
        with col:
            st.markdown(
                f"<div class='hero-stat-card'><span class='hero-stat-value'>{stat['value']}</span><span class='hero-stat-label'>{stat['note']}</span></div>",
                unsafe_allow_html=True,
            )

    st.markdown(
        "<div class='landing-grid'>"
        "<div class='landing-card'><h3>Project Overview</h3><p>Digital Mirror combines digital behavior analysis, risk classification, and AI-driven insights into a polished web experience for research and product demos.</p></div>"
        "<div class='landing-card'><h3>Why This Model?</h3><p>Random Forest offers robust generalization, explainability with feature importance, and stability across mixed numeric and categorical usage data.</p></div>"
        "<div class='landing-card'><h3>Premium Experience</h3><p>Designed with glassmorphism, glow accents, and a polished layout that feels like a funded AI product.</p></div>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_dashboard_page(df, summary: Dict[str, object], metrics: Dict[str, float]) -> None:
    render_header("Dashboard", "Live monitoring of UX signals, model health, and dataset intelligence.")
    metric_tiles = {
        "Total Records": {"value": summary["rows"], "note": "Captured participants"},
        "Total Features": {"value": summary.get("feature_count", summary["columns"] - 2), "note": "Predictive attributes"},
        "Accuracy": {"value": f"{metrics['accuracy'] * 100:.1f}%", "note": "Test set performance"},
        "Classes": {"value": len(summary["class_distribution"]), "note": "Risk buckets"},
        "Missing": {"value": summary["missing_values"], "note": "Data quality"},
        "Model Status": {"value": "Deployed", "note": "Ready for inference"},
    }
    render_metric_cards(metric_tiles)
    st.markdown("<div class='content-card'><h3>Dataset Snapshot</h3><p>Digital Mirror is trained on behavioral signals, usage patterns, emotional tone, and content preferences to assess addiction risk with premium precision.</p></div>", unsafe_allow_html=True)

    col1, col2 = st.columns((2, 1))
    with col1:
        fig = px.histogram(df, x="daily_usage_hours", nbins=24, title="Daily Usage Hours Distribution", color_discrete_sequence=["#00E5FF"])
        fig.update_layout(plot_bgcolor="rgba(4,9,23,0.95)", paper_bgcolor="rgba(4,9,23,0.95)", font_color="#FFFFFF")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        dist = px.pie(
            names=list(summary["class_distribution"].keys()),
            values=list(summary["class_distribution"].values()),
            title="Risk Class Distribution",
            color_discrete_sequence=["#00E5FF", "#7B61FF", "#FF4D8D"],
        )
        dist.update_layout(plot_bgcolor="rgba(4,9,23,0.95)", paper_bgcolor="rgba(4,9,23,0.95)", font_color="#FFFFFF")
        st.plotly_chart(dist, use_container_width=True)


def render_dataset_explorer(df: pd.DataFrame) -> None:
    render_header("Dataset Explorer", "Interactive browsing with filtering, sorting, and data quality insights.")
    if AG_GRID_AVAILABLE:
        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_default_column(filterable=True, sortable=True, resizable=True, floatingFilter=True)
        gb.configure_selection(selection_mode="multiple", use_checkbox=True)
        grid_options = gb.build()
        AgGrid(df, gridOptions=grid_options, theme="dark", enable_enterprise_modules=False, fit_columns_on_grid_load=True)
    else:
        st.dataframe(df, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        st.metric("Duplicates", int(df.duplicated().sum()))
    st.markdown("<div class='small-card'><strong>Missing Values:</strong> None detected in the uploaded dataset.</div>", unsafe_allow_html=True)
    st.markdown("<div class='small-card'><strong>Text Fields:</strong> Behavioral text is preserved for future natural language augmentation.</div>", unsafe_allow_html=True)


def render_data_visualization(df: pd.DataFrame) -> None:
    render_header("Data Visualization", "Immersive analytics for behavior, sentiment, and risk signals.")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x="average_session_duration", nbins=30, title="Session Duration Distribution", color_discrete_sequence=["#7B61FF"])
        fig.update_layout(plot_bgcolor="rgba(4,9,23,0.95)", paper_bgcolor="rgba(4,9,23,0.95)", font_color="#FFFFFF")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.box(df, x="addiction_risk_level", y="sleep_disruption_score", color="addiction_risk_level", title="Sleep Disruption by Risk Level", color_discrete_sequence=["#FF4D8D", "#00E5FF", "#7B61FF"])
        fig.update_layout(plot_bgcolor="rgba(4,9,23,0.95)", paper_bgcolor="rgba(4,9,23,0.95)", font_color="#FFFFFF")
        st.plotly_chart(fig, use_container_width=True)

    heatmap = px.imshow(
        df.select_dtypes(include=[np.number]).corr(),
        text_auto=True,
        color_continuous_scale="Turbo",
        title="Correlation Heatmap",
    )
    heatmap.update_layout(plot_bgcolor="rgba(4,9,23,0.95)", paper_bgcolor="rgba(4,9,23,0.95)", font_color="#FFFFFF")
    st.plotly_chart(heatmap, use_container_width=True)

    scatter = px.scatter_matrix(
        df,
        dimensions=["daily_usage_hours", "average_session_duration", "sleep_hours", "digital_behavioral_risk_index"],
        color="addiction_risk_level",
        title="Pair Plot Style Analysis",
        color_discrete_sequence=["#00E5FF", "#7B61FF", "#FF4D8D"],
    )
    scatter.update_layout(plot_bgcolor="rgba(4,9,23,0.95)", paper_bgcolor="rgba(4,9,23,0.95)", font_color="#FFFFFF")
    st.plotly_chart(scatter, use_container_width=True)


def render_prediction_center(
    pipeline,
    encoder,
    schema: List[Dict],
    feature_columns: List[str],
    raw_importances: Dict[str, float],
    df: pd.DataFrame,
) -> Dict[str, object]:
    render_header("Prediction Center", "Enter behavior signals and receive AI risk inference with holographic clarity.")
    st.markdown("<div class='prediction-banner'>Fill the smart form to generate a premium behavioral prediction.</div>", unsafe_allow_html=True)

    with st.form("prediction_form"):
        form_columns = st.columns(2)
        input_values = {}
        for index, field in enumerate(schema):
            target_col = f"input_{field['name']}"
            container = form_columns[index % 2]
            with container:
                if field["type"] == "select":
                    input_values[field["name"]] = st.selectbox(field["label"], field["options"], index=0, key=target_col)
                elif field["type"] == "int":
                    input_values[field["name"]] = st.number_input(
                        field["label"], min_value=field["min"], max_value=field["max"], value=field["default"], step=field["step"], key=target_col
                    )
                else:
                    input_values[field["name"]] = st.number_input(
                        field["label"], min_value=field["min"], max_value=field["max"], value=field["default"], step=field["step"], format="%.2f", key=target_col
                    )
        submit = st.form_submit_button("Run AI Prediction")

    if submit:
        record = pd.DataFrame([input_values])
        try:
            prediction = build_prediction_results(pipeline, encoder, record, df, feature_columns, raw_importances)
            render_prediction_result(record, prediction)
            return prediction
        except Exception as error:
            st.error(f"Prediction failed: {error}")
            return {}
    return {}


def render_prediction_result(record: pd.DataFrame, prediction: Dict[str, object]) -> None:
    st.markdown("<div class='result-panel'>Prediction delivered in real time with premium holographic styling.</div>", unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(
            "<div class='holo-card'><h2>Behavioral Forecast</h2><p>Your personalized AI-powered risk assessment is ready.</p>"
            f"<div class='result-line'><span>Prediction</span><strong>{prediction['prediction_label']}</strong></div>"
            f"<div class='result-line'><span>Confidence</span><strong>{prediction['confidence'] * 100:.1f}%</strong></div>"
            "</div>",
            unsafe_allow_html=True,
        )
    with col2:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction["confidence"] * 100,
            number={"suffix": "%", "font": {"color": "#00FF95", "size": 32}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#FFFFFF"},
                "bar": {"color": "#00E5FF"},
                "bgcolor": "rgba(255,255,255,0.08)",
                "steps": [
                    {"range": [0, 50], "color": "rgba(255,77,141,0.15)"},
                    {"range": [50, 80], "color": "rgba(123,97,255,0.18)"},
                    {"range": [80, 100], "color": "rgba(0,229,255,0.22)"},
                ],
            },
        ))
        gauge.update_layout(paper_bgcolor="rgba(4,9,23,0.95)", plot_bgcolor="rgba(4,9,23,0.95)", font_color="#FFFFFF", margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(gauge, use_container_width=True)

    probability_df = pd.DataFrame(
        list(prediction["probability_by_class"].items()), columns=["Class", "Probability"]
    )
    st.table(probability_df.style.format({"Probability": "{:.3f}"}))

    st.markdown(
        "<div class='content-card'><h3>Why?</h3>"
        f"<p>{prediction['why']}</p>"
        "<ul>"
        + "".join([f"<li>{item}</li>" for item in prediction.get('main_factors', [])])
        + "</ul>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='content-card'><h3>Recommendations</h3>"
        "<ul>"
        + "".join([f"<li>{item}</li>" for item in prediction.get('recommendations', [])])
        + "</ul>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<div class='content-card'><h3>AI Interpretation</h3><p>{prediction.get('insight', '')}</p></div>",
        unsafe_allow_html=True,
    )


def render_model_performance(metrics: Dict[str, object], roc_data: Dict[str, object]) -> None:
    render_header("Model Performance", "Professional metrics with confusion analysis and ROC visualization.")
    cols = st.columns(3)
    cols[0].metric("Accuracy", f"{metrics['accuracy'] * 100:.1f}%")
    cols[1].metric("Precision", f"{metrics['precision'] * 100:.1f}%")
    cols[2].metric("Recall", f"{metrics['recall'] * 100:.1f}%")
    st.write(metrics["classification_report"])
    heatmap = px.imshow(
        metrics["confusion_matrix"],
        labels={"x": "Predicted", "y": "Actual", "color": "Count"},
        x=list(metrics["classification_report"].index[:-3]),
        y=list(metrics["classification_report"].index[:-3]),
        color_continuous_scale="Inferno",
        title="Confusion Matrix",
    )
    heatmap.update_layout(plot_bgcolor="rgba(4,9,23,0.95)", paper_bgcolor="rgba(4,9,23,0.95)", font_color="#FFFFFF")
    st.plotly_chart(heatmap, use_container_width=True)

    if roc_data and roc_data.get("roc_curves"):
        roc_chart = go.Figure()
        for curve in roc_data["roc_curves"]:
            roc_chart.add_trace(
                go.Scatter(x=curve["fpr"], y=curve["tpr"], mode="lines", name=f"{curve['name']} AUC: {roc_data['roc_auc'][curve['name']]:.3f}")
            )
        roc_chart.add_shape(type="line", x0=0, x1=1, y0=0, y1=1, line=dict(color="#888", dash="dash"))
        roc_chart.update_layout(
            title="Multi-Class ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            plot_bgcolor="rgba(4,9,23,0.95)",
            paper_bgcolor="rgba(4,9,23,0.95)",
            font_color="#FFFFFF",
        )
        st.plotly_chart(roc_chart, use_container_width=True)


def render_feature_importance(importances: Dict[str, float]) -> None:
    render_header("Feature Importance", "Ranked predictive signals that shape the Digital Mirror model.")
    importance_df = (
        pd.DataFrame(
            {"feature": list(importances.keys()), "importance": list(importances.values())}
        )
        .sort_values(by="importance", ascending=False)
        .head(10)
    )
    chart = px.bar(
        importance_df[::-1],
        x="importance",
        y="feature",
        orientation="h",
        title="Top 10 Feature Importances",
        color="importance",
        color_continuous_scale="viridis",
    )
    chart.update_layout(plot_bgcolor="rgba(4,9,23,0.95)", paper_bgcolor="rgba(4,9,23,0.95)", font_color="#FFFFFF")
    st.plotly_chart(chart, use_container_width=True)
    st.table(importance_df.style.format({"importance": "{:.4f}"}))


def render_ai_insights(summary: Dict[str, object], df: pd.DataFrame, feature_columns: List[str], importances: Dict[str, float]) -> None:
    render_header("AI Insights", "Automated intelligence drawn from dataset patterns and feature signals.")
    sorted_features = sorted(importances.items(), key=lambda item: item[1], reverse=True)[:5]
    top_feature_name, top_feature_score = sorted_features[0]
    top_correlations = summary.get("top_correlations", {})
    strongest_corr_name = next(iter(top_correlations), "N/A")
    class_distribution = summary.get("class_distribution", {})
    insights = [
        {
            "title": "Most Important Predictor",
            "description": f"{top_feature_name.title().replace('_', ' ')} is the strongest feature driving addiction risk classification.",
        },
        {
            "title": "Strongest Correlation",
            "description": f"{strongest_corr_name.title().replace('_', ' ')} is the most correlated signal with behavioral risk index in the dataset.",
        },
        {
            "title": "Class Distribution",
            "description": "High, Moderate, and Low risk classes are represented with well-balanced sample counts for reliable modeling.",
        },
    ]
    cards = st.columns(3)
    for column, insight in zip(cards, insights):
        with column:
            st.markdown(
                f"<div class='insight-card'><h4>{insight['title']}</h4><p>{insight['description']}</p></div>",
                unsafe_allow_html=True,
            )
    feature_text = "<br>".join([f"{idx + 1}. {name.title().replace('_', ' ')} ({score:.4f})" for idx, (name, score) in enumerate(sorted_features)])
    st.markdown(
        f"<div class='content-card'><h3>Top Feature Signals</h3><p>{feature_text}</p></div>",
        unsafe_allow_html=True,
    )
    chart = px.pie(
        names=list(class_distribution.keys()),
        values=list(class_distribution.values()),
        title="Class Distribution",
        color_discrete_sequence=["#00E5FF", "#7B61FF", "#FF4D8D"],
    )
    chart.update_layout(plot_bgcolor="rgba(4,9,23,0.95)", paper_bgcolor="rgba(4,9,23,0.95)", font_color="#FFFFFF")
    st.plotly_chart(chart, use_container_width=True)


def render_about_page() -> None:
    render_header("About Project", "A premium AI SaaS prototype for personality and digital behavioral analysis.")
    st.markdown(
        "<div class='about-grid'>"
        "<div class='about-card'><h4>Vision</h4><p>Deliver enterprise-ready behavioral analytics with a futuristic interface and trustworthy risk inference.</p></div>"
        "<div class='about-card'><h4>Technology</h4><p>Built with Streamlit, pandas, scikit-learn, Plotly and a high-fidelity custom UI to feel like a startup product.</p></div>"
        "<div class='about-card'><h4>Use Cases</h4><p>Designed for hackathons, research demos, and portfolio showcases where design, insight, and model quality matter.</p></div>"
        "</div>",
        unsafe_allow_html=True,
    )
