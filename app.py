"""Streamlit dashboard for artifacts exported by financial_news_sentiment.ipynb.

Run from the project folder:
    python -m streamlit run app.py

Finish notebook Steps 18 and 19 to export inference code and dashboard data.
"""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Financial News Sentiment",
    page_icon="📈",
    layout="centered",
)
st.title("Financial News Sentiment")
st.caption("Classify English financial tweets and headlines as Bearish, Bullish, or Neutral.")


def find_saved_runs():
    """Find exported runs next to this app or in the notebook's output folder."""
    candidates = [APP_DIR]
    output_dir = APP_DIR / "financial_sentiment_outputs"
    if output_dir.is_dir():
        candidates.extend(sorted(output_dir.glob("run-*"), reverse=True))
    return [
        directory
        for directory in candidates
        if (directory / "manifest.json").is_file()
        and (directory / "sentiment_core.py").is_file()
    ]


@st.cache_resource
def load_predictor(artifact_folder, manifest_modified, core_modified):
    """Reuse the notebook's inference code; file timestamps invalidate the cache."""
    core_path = Path(artifact_folder) / "sentiment_core.py"
    specification = importlib.util.spec_from_file_location("saved_sentiment_core", core_path)
    if specification is None or specification.loader is None:
        raise ImportError("Could not load the exported sentiment_core.py module.")
    core = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(core)
    return core.SentimentPredictor(artifact_folder, device_name="cpu")


st.sidebar.header("Saved model")
custom_folder = st.sidebar.text_input(
    "Run folder (optional)",
    placeholder="/path/to/financial_sentiment_outputs/run-...",
    help="Use a run folder exported by your notebook, including its models and manifest.",
).strip()
saved_runs = find_saved_runs()

if custom_folder:
    artifact_dir = Path(custom_folder).expanduser().resolve()
elif saved_runs:
    artifact_dir = st.sidebar.selectbox(
        "Training run", saved_runs, format_func=lambda directory: directory.name
    )
else:
    st.info(
        "No exported model is available yet. Run the notebook through Step 18 to "
        "save the inference files, then Step 19 to save the dashboard data. "
        "Refresh this page when those steps finish."
    )
    st.code("financial_sentiment_outputs/run-.../manifest.json\n"
            "financial_sentiment_outputs/run-.../sentiment_core.py\n"
            "financial_sentiment_outputs/run-.../models/")
    st.stop()

missing = [
    name for name in ("manifest.json", "sentiment_core.py")
    if not (artifact_dir / name).is_file()
]
if missing:
    st.error("The selected run folder is missing: " + ", ".join(missing))
    st.stop()

try:
    predictor = load_predictor(
        str(artifact_dir),
        (artifact_dir / "manifest.json").stat().st_mtime_ns,
        (artifact_dir / "sentiment_core.py").stat().st_mtime_ns,
    )
except (ImportError, OSError, ValueError, KeyError, RuntimeError) as error:
    st.error(f"Could not load the saved model: {error}")
    st.info("Use the complete exported run folder, including model weights and tokenizer files.")
    st.stop()

st.caption(f"Selected model: {predictor.model_name}")
text = st.text_area(
    "Financial text",
    height=140,
    placeholder="Paste a financial tweet, headline, or short paragraph...",
)

if st.button("Predict sentiment", type="primary"):
    try:
        with st.spinner("Predicting sentiment..."):
            result = predictor.predict(text)
        st.subheader(result["sentiment"])
        probabilities = pd.DataFrame({
            "Sentiment": list(result["probabilities"]),
            "Probability": list(result["probabilities"].values()),
        }).set_index("Sentiment")
        st.bar_chart(probabilities)
        st.dataframe(probabilities.style.format("{:.1%}"), width="stretch")
        if result["truncated"]:
            st.warning("This input exceeds the model limit; only its beginning was classified.")
        st.caption("Probabilities are model estimates and have not been calibrated.")
    except (TypeError, ValueError) as error:
        st.warning(str(error))
    except RuntimeError as error:
        st.error(f"Prediction failed: {error}")

reports_dir = artifact_dir / "reports"
distribution_path = reports_dir / "validation_class_distribution.csv"
comparison_path = reports_dir / "model_comparison.csv"

st.subheader("Evaluation sample: class distribution")
if distribution_path.is_file():
    try:
        distribution = pd.read_csv(distribution_path)
        st.bar_chart(distribution.set_index("Sentiment")[["Count"]])
    except (OSError, ValueError, KeyError) as error:
        st.warning(f"Could not display the class distribution: {error}")
else:
    st.info("Run notebook Step 19 to export the validation class distribution.")

with st.expander("Measured model comparison"):
    if comparison_path.is_file():
        try:
            st.dataframe(pd.read_csv(comparison_path), width="stretch")
        except (OSError, ValueError) as error:
            st.warning(f"Could not display model comparisons: {error}")
    else:
        st.info("Run notebook Step 16 to export the measured model comparison.")

st.caption("Evaluation used financial tweets. Paragraph generalization needs separate evaluation.")
st.caption("This app estimates sentiment; it does not establish future price direction.")
