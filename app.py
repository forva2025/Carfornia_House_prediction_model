from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


st.set_page_config(
    page_title="California Home Value Studio",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path(__file__).with_name("house_price_pipeline.joblib")
TARGET = "MedHouseVal"
FEATURE_LABELS = {
    "MedInc": "Median income",
    "HouseAge": "Median house age",
    "AveRooms": "Average rooms",
    "AveBedrms": "Average bedrooms",
    "Population": "Block population",
    "AveOccup": "Average occupancy",
    "Latitude": "Latitude",
    "Longitude": "Longitude",
}


st.markdown(
    """
    <style>
      .stApp { background: #f7f8f3; }
      [data-testid="stSidebar"] { background: #17352d; }
      [data-testid="stSidebar"] * { color: #f7f5ed; }
      [data-testid="stMetric"] {
        background: white; border: 1px solid #dfe5dc; border-radius: 14px;
        padding: 16px 18px; box-shadow: 0 8px 24px rgba(28, 52, 43, .06);
      }
      .hero {
        padding: 28px 32px; border-radius: 22px; color: #f8f4e8;
        background: linear-gradient(125deg, #17352d 0%, #28614f 62%, #c97845 160%);
        margin-bottom: 22px; box-shadow: 0 16px 38px rgba(23, 53, 45, .18);
      }
      .eyebrow { text-transform: uppercase; letter-spacing: .14em; font-size: .76rem; opacity: .78; }
      .hero h1 { margin: .3rem 0 .35rem; font-size: clamp(2rem, 4vw, 3.35rem); line-height: 1.04; }
      .hero p { margin: 0; max-width: 740px; font-size: 1.03rem; opacity: .86; }
      .prediction {
        background: #fff; border-left: 6px solid #c97845; border-radius: 16px;
        padding: 22px 24px; margin: 12px 0 20px; box-shadow: 0 8px 24px rgba(28, 52, 43, .07);
      }
      .prediction small { color: #5e6b65; text-transform: uppercase; letter-spacing: .1em; }
      .prediction strong { display: block; color: #17352d; font-size: 2.25rem; margin-top: 3px; }
      .fine-print { color: #64716b; font-size: .86rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading the California housing dataset…")
def load_data() -> pd.DataFrame:
    return fetch_california_housing(as_frame=True).frame


@st.cache_resource(show_spinner="Training the linear regression model…")
def train_model():
    data = load_data()
    x = data.drop(columns=TARGET)
    y = data[TARGET]
    features = list(x.columns)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.4, random_state=42
    )

    model = None
    model_source = "trained for this deployment"
    if MODEL_PATH.exists():
        try:
            artifact = joblib.load(MODEL_PATH)
            if isinstance(artifact, dict):
                candidate = artifact.get("model")
                saved_features = artifact.get("features")
                if saved_features is not None and list(saved_features) != features:
                    raise ValueError("The saved model uses a different feature order.")
            else:
                # Support the estimator-only file produced by house_prices.py.
                candidate = artifact

            if not hasattr(candidate, "predict"):
                raise TypeError("The saved artifact does not contain a model.")
            if getattr(candidate, "n_features_in_", len(features)) != len(features):
                raise ValueError("The saved model expects a different number of features.")

            model = candidate
            model_source = "loaded from the saved joblib artifact"
        except Exception:
            # A model serialized by another scikit-learn version may not load.
            # Retraining keeps the deployed app usable in that situation.
            model = None

    if model is None:
        model = LinearRegression()
        model.fit(x_train, y_train)
        try:
            joblib.dump({"model": model, "features": features}, MODEL_PATH)
        except OSError:
            # Some deployment filesystems are read-only; the in-memory model is enough.
            pass

    predictions = model.predict(x_test)
    metrics = {
        "mse": mean_squared_error(y_test, predictions),
        "rmse": np.sqrt(mean_squared_error(y_test, predictions)),
        "r2": r2_score(y_test, predictions),
    }
    return (
        model,
        x,
        x_train,
        x_test,
        y_train,
        y_test,
        predictions,
        metrics,
        model_source,
    )


try:
    data = load_data()
    (
        model,
        x,
        x_train,
        x_test,
        y_train,
        y_test,
        y_pred,
        metrics,
        model_source,
    ) = train_model()
except Exception as exc:
    st.error(
        "The dataset could not be loaded. On first launch, the app needs internet access "
        "so scikit-learn can download the California Housing dataset."
    )
    st.exception(exc)
    st.stop()


st.markdown(
    """
    <section class="hero">
      <div class="eyebrow">Linear regression · California housing</div>
      <h1>Home Value Studio</h1>
      <p>Explore the model, tune the neighborhood profile, and estimate the median home value for a California census block.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Neighborhood profile")
    st.caption("Set all eight model inputs, then calculate an estimate.")

    user_values = {}
    defaults = x.median(numeric_only=True)
    for feature in x.columns:
        series = x[feature]
        min_value = float(series.quantile(0.01))
        max_value = float(series.quantile(0.99))
        default = float(np.clip(defaults[feature], min_value, max_value))
        step = max((max_value - min_value) / 100.0, 0.01)
        user_values[feature] = st.number_input(
            FEATURE_LABELS[feature],
            min_value=min_value,
            max_value=max_value,
            value=default,
            step=step,
        )

    calculate = st.button(
        "Calculate estimate", type="primary", use_container_width=True
    )


if calculate:
    input_frame = pd.DataFrame([user_values], columns=x.columns)
    predicted_value = float(model.predict(input_frame)[0])
    predicted_dollars = predicted_value * 100_000
    st.markdown(
        f"""
        <div class="prediction">
          <small>Estimated median home value</small>
          <strong>${predicted_dollars:,.0f}</strong>
          <span class="fine-print">
            Model output: {predicted_value:.3f} in units of $100,000.
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.info("Set the neighborhood profile and select **Calculate estimate**.")


st.subheader("Model performance")
metric_columns = st.columns(4)
metric_columns[0].metric("Test R²", f"{metrics['r2']:.3f}")
metric_columns[1].metric("Test RMSE", f"${metrics['rmse'] * 100_000:,.0f}")
metric_columns[2].metric("Test MSE", f"{metrics['mse']:.3f}")
metric_columns[3].metric("Training rows", f"{len(x_train):,}")
st.caption(
    f"The model was {model_source}. The evaluation uses {len(x_test):,} held-out rows."
)


performance_tab, coefficients_tab, data_tab = st.tabs(
    ["Actual vs. predicted", "Model coefficients", "Dataset explorer"]
)

with performance_tab:
    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.scatter(y_test, y_pred, alpha=0.28, color="#28614f", edgecolors="none")
    comparison_min = min(float(y_test.min()), float(y_pred.min()))
    comparison_max = max(float(y_test.max()), float(y_pred.max()))
    axis.plot(
        [comparison_min, comparison_max],
        [comparison_min, comparison_max],
        linestyle="--",
        linewidth=2,
        color="#c97845",
        label="Perfect prediction",
    )
    axis.set_title("Actual versus predicted median home values")
    axis.set_xlabel("Actual value ($100,000s)")
    axis.set_ylabel("Predicted value ($100,000s)")
    axis.legend(frameon=False)
    axis.grid(alpha=0.18)
    figure.tight_layout()
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)

with coefficients_tab:
    coefficient_data = pd.DataFrame(
        {
            "Feature": x.columns,
            "Label": [FEATURE_LABELS[feature] for feature in x.columns],
            "Coefficient": model.coef_,
        }
    )
    coefficient_data["Absolute effect"] = coefficient_data["Coefficient"].abs()
    coefficient_data = coefficient_data.sort_values(
        "Absolute effect", ascending=False
    ).drop(columns="Absolute effect")
    st.dataframe(
        coefficient_data,
        use_container_width=True,
        hide_index=True,
        column_config={"Coefficient": st.column_config.NumberColumn(format="%.4f")},
    )
    st.caption(
        "Coefficients show the linear change in the target for a one-unit change "
        "in each feature, with the other features held constant."
    )

with data_tab:
    st.dataframe(data, use_container_width=True, hide_index=True, height=430)
    st.download_button(
        "Download dataset as CSV",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name="california_housing.csv",
        mime="text/csv",
    )
