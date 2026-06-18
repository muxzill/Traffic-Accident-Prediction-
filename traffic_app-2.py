# app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time

# ---------------------------
# Page config
st.set_page_config(
    page_title="🚦 Accident Severity Prediction — Modern UI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Global CSS (glassmorphism + clean cards, no Lottie)
st.markdown(
    """
    <style>
    :root{
      --accent1: #00f7ff;
      --bg1: #0f1724;
      --card: rgba(255,255,255,0.04);
      --glass-border: rgba(255,255,255,0.06);
    }
    .stApp {
      background: linear-gradient(135deg, #071023 0%, #071a2b 50%, #0b2a3b 100%);
      color: #e6f7fb;
      font-family: 'Segoe UI', Roboto, sans-serif;
    }

    /* Page container */
    .block-container {
      padding: 1.6rem 2rem;
    }

    /* Top header */
    .header {
      display: flex;
      align-items: center;
      gap: 16px;
      margin-bottom: 12px;
    }
    .brand {
      font-size: 1.4rem;
      font-weight: 700;
      color: var(--accent1);
      text-shadow: 0 0 10px rgba(0,247,255,0.12);
    }

    /* Cards */
    .kpi {
      background: var(--card);
      border: 1px solid var(--glass-border);
      border-radius: 12px;
      padding: 14px;
      box-shadow: 0 8px 22px rgba(0,0,0,0.35);
      transition: transform .18s ease, box-shadow .18s ease;
    }
    .kpi:hover { transform: translateY(-6px); box-shadow: 0 18px 40px rgba(0,0,0,0.45); }

    /* Tabs look */
    .stTabs [data-baseweb="tab"] {
      background: rgba(255,255,255,0.03);
      color: var(--accent1);
      border-radius: 10px;
      padding: 8px 14px;
      font-weight: 600;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
      background: linear-gradient(90deg, rgba(0,247,255,0.15), rgba(0,195,255,0.12));
      color: #e8fbff;
    }

    /* Buttons */
    .stButton>button {
      background: linear-gradient(90deg,#00f7ff,#00c3ff) !important;
      color: #001012 !important;
      font-weight: 700;
      border-radius: 10px;
      padding: 8px 16px;
      box-shadow: 0 8px 20px rgba(0,195,255,0.08);
    }

    /* Download button */
    .stDownloadButton>button {
      background: #e8fbff !important;
      color: #001012 !important;
      font-weight: 700;
      border-radius: 10px;
    }

    /* Dataframe card */
    [data-testid="stDataFrame"] {
      border-radius: 10px;
      border: 1px solid rgba(255,255,255,0.04);
    }

    /* Small helper */
    .muted { color: rgba(255,255,255,0.6); font-size: 0.95rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
TARGET_COLUMN = "Severity"

# ---------------------------
@st.cache_data
def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    df = pd.DataFrame({
        "Vehicles_Involved": np.random.randint(1, 6, size=n_samples),
        "Road_Type": np.random.choice(["Highway", "Local", "Intersection", "T-Junction"], size=n_samples),
        "Weather": np.random.choice(["Clear", "Fog", "Rain", "Snow"], size=n_samples),
        "Severity": np.random.choice(["Low", "Medium", "High"], p=[0.6, 0.3, 0.1], size=n_samples)
    })
    df["Accident_ID"] = np.arange(n_samples)
    df["Date"] = pd.to_datetime("2023-01-01") + pd.to_timedelta(np.random.randint(0, 365, size=n_samples), unit="d")
    df["Location"] = np.random.choice(["City_A", "City_B", "City_C"], size=n_samples)
    return df

@st.cache_data
def preprocess_data(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    to_encode = []
    for c in ["Weather", "Road_Type"]:
        if c in df.columns:
            df[c] = df[c].fillna("Unknown")
            to_encode.append(c)
    if to_encode:
        df = pd.get_dummies(df, columns=to_encode, drop_first=True)
    df = df.drop([c for c in ["Accident_ID", "Date", "Location"] if c in df.columns], axis=1, errors="ignore")
    return df

@st.cache_resource
def train_model(df):
    df_p = preprocess_data(df)
    if TARGET_COLUMN not in df_p.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' missing.")
    y = df_p[TARGET_COLUMN].astype("category")
    X = df_p.drop(TARGET_COLUMN, axis=1)
    X.columns = X.columns.astype(str)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=y.cat.categories)
    # feature importances aligned to X columns
    fi = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    return model, X.columns, acc, report, cm, fi, y.cat.categories

# ---------------------------
# Sidebar: data upload + settings + stats cards
with st.sidebar:
    st.markdown("<div class='header'><div class='brand'>🚦 Accident Severity — Demo</div></div>", unsafe_allow_html=True)
    st.header("📂 Data & Settings")
    uploaded_file = st.file_uploader("Upload CSV (optional)", type=["csv"])
    use_synthetic = st.checkbox("Use Synthetic Data (recommended demo)", value=True if not uploaded_file else False)
    st.markdown("---")
    st.markdown("**Model options**")
    n_estimators = st.slider("RandomForest trees", 50, 300, 150, step=25)
    st.markdown("---")
    st.caption("Tip: For demo, keep synthetic ON. If you upload CSV, must contain 'Severity' column.")

# ---------------------------
# Load data
if uploaded_file and not use_synthetic:
    try:
        df_original = pd.read_csv(uploaded_file)
        st.sidebar.success(f"Loaded {len(df_original)} rows from file")
    except Exception as e:
        st.sidebar.error(f"Load error: {e}")
        df_original = generate_synthetic_data()
else:
    df_original = generate_synthetic_data()

# replace n_estimators in model training by re-wrapping train_model if changed
# We'll create a small wrapper to re-train with chosen trees (non-cached)
def train_model_with_params(df, n_estimators_local):
    df_p = preprocess_data(df)
    if TARGET_COLUMN not in df_p.columns:
        st.error(f"Target column '{TARGET_COLUMN}' not found in dataset.")
        st.stop()
    y = df_p[TARGET_COLUMN].astype("category")
    X = df_p.drop(TARGET_COLUMN, axis=1)
    X.columns = X.columns.astype(str)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=n_estimators_local, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=y.cat.categories)
    fi = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    return model, X.columns, acc, report, cm, fi, y.cat.categories

# Train (with selected trees)
try:
    model, feature_names, accuracy, report, cm, fi, classes = train_model_with_params(df_original, n_estimators)
except Exception as e:
    st.error(f"Training Error: {e}")
    st.stop()

# Sidebar KPI cards
with st.sidebar:
    st.markdown("### 🔍 Quick Stats")
    k1, k2 = st.columns([1,1])
    with k1:
        st.markdown(f"<div class='kpi'><h4>Rows</h4><h2>{len(df_original):,}</h2><div class='muted'>Dataset size</div></div>", unsafe_allow_html=True)
    with k2:
        st.markdown(f"<div class='kpi'><h4>Classes</h4><h2>{len(df_original[TARGET_COLUMN].unique())}</h2><div class='muted'>Severity categories</div></div>", unsafe_allow_html=True)
    st.markdown("---")
    st.metric("Model Accuracy", f"{accuracy:.2%}")
    st.markdown("---")
    if st.button("Retrain (with current settings)"):
        with st.spinner("Retraining..."):
            model, feature_names, accuracy, report, cm, fi, classes = train_model_with_params(df_original, n_estimators)
            st.success("Retrained ✅")

# ---------------------------
# Main layout: header + tabs
st.markdown("<div class='header'><div style='flex:1'><h1>🚨 Accident Severity Prediction</h1><div class='muted'>Clean modern interface — ready for presentation</div></div></div>", unsafe_allow_html=True)
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Analysis", "🤖 Live Prediction", "📈 Model Metrics", "📥 Download"])

# ---------------------------
# Tab 1: Data Analysis
with tab1:
    st.subheader("Exploratory Data Analysis")
    left, right = st.columns([2,1])
    with left:
        st.markdown("**Sample Data**")
        st.dataframe(df_original.head(20), use_container_width=True)
        st.markdown("---")
        st.markdown("**Distribution of Severity**")
        fig, ax = plt.subplots(figsize=(6,3))
        sns.countplot(data=df_original, x=TARGET_COLUMN, order=df_original[TARGET_COLUMN].value_counts().index, ax=ax)
        ax.set_title("Severity Counts")
        st.pyplot(fig, clear_figure=True)
        st.markdown("---")
        st.markdown("**Numerical Correlation (heatmap)**")
        fig, ax = plt.subplots(figsize=(7,5))
        sns.heatmap(df_original.corr(numeric_only=True), annot=True, cmap="vlag", center=0, ax=ax)
        st.pyplot(fig, clear_figure=True)
    with right:
        st.markdown("**Feature Importance (Top features)**")
        st.markdown("<div class='kpi'>", unsafe_allow_html=True)
        # show top 8 features
        topn = fi.head(8)
        for feat, val in topn.items():
            pct = f"{val*100:.1f}%"
            st.markdown(f"**{feat}** — {pct}")
            st.progress(min(100, int(val*100)))  # simple bar
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("**Class breakdown**")
        st.dataframe(pd.DataFrame(df_original[TARGET_COLUMN].value_counts()).rename(columns={TARGET_COLUMN:"Count"}))

# ---------------------------
# Tab 2: Live Prediction
with tab2:
    st.subheader("Simulate a Live Accident Case")
    col_v, col_r, col_w = st.columns(3)
    with col_v:
        vehicles = st.slider("Vehicles Involved", 1, 10, 2)
    with col_r:
        road = st.selectbox("Road Type", sorted(df_original["Road_Type"].unique()))
    with col_w:
        weather = st.selectbox("Weather", sorted(df_original["Weather"].unique()))

    st.markdown(" ")
    if st.button("⚡ Run Prediction"):
        with st.spinner("Predicting..."):
            time.sleep(0.8)
            # prepare input vector aligned with trained features
            input_data = {"Vehicles_Involved": vehicles}
            input_df = pd.DataFrame([input_data])
            # one-hot encode the road/weather into same columns as original df
            for opt in df_original["Road_Type"].unique():
                input_df[f"Road_Type_{opt}"] = 1 if opt == road else 0
            for opt in df_original["Weather"].unique():
                input_df[f"Weather_{opt}"] = 1 if opt == weather else 0
            # drop base columns if present (preprocessing used drop_first)
            # create final input with zeros for all feature_names
            final_input = pd.DataFrame(0, index=[0], columns=feature_names)
            for col in input_df.columns:
                if col in final_input.columns:
                    final_input[col] = input_df[col].iloc[0]
            # predict
            pred = model.predict(final_input)[0]
            probs = None
            try:
                probs = model.predict_proba(final_input)[0]
            except:
                pass

        # show results with nice badges + advice
        if pred == "High":
            st.error(f"🔴 **CRITICAL — Predicted: {pred}**\n\n➡️ Immediate emergency response recommended.")
        elif pred == "Medium":
            st.warning(f"🟠 **MODERATE — Predicted: {pred}**\n\n➡️ Send standard response / caution.")
        else:
            st.success(f"🟢 **LOW — Predicted: {pred}**\n\n➡️ Minimal response required.")

        # show class probabilities if available
        if probs is not None:
            prob_df = pd.DataFrame({"Class": list(classes), "Probability": probs})
            st.markdown("**Prediction Probabilities**")
            st.dataframe(prob_df.style.format({"Probability": "{:.2%}"}), use_container_width=True)

# ---------------------------
# Tab 3: Metrics
with tab3:
    st.subheader("Model Performance & Debugging")
    st.markdown("**Accuracy & Summary**")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Accuracy", f"{accuracy:.2%}")
        st.markdown("**Classification Report**")
        st.dataframe(pd.DataFrame(report).transpose())
    with c2:
        st.markdown("**Confusion Matrix**")
        fig, ax = plt.subplots(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="mako", xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig, clear_figure=True)

    st.markdown("---")
    st.markdown("**Top Feature Importance (bar)**")
    fig, ax = plt.subplots(figsize=(8,4))
    fi.head(12).sort_values().plot(kind="barh", ax=ax)
    ax.set_title("Feature Importance (RandomForest)")
    st.pyplot(fig, clear_figure=True)

# ---------------------------
# Tab 4: Download
with tab4:
    st.subheader("Export Predictions & Sample")
    sample = df_original.head(200).copy()
    # prepare X for sample and predict
    X_sample = preprocess_data(sample.drop([TARGET_COLUMN], axis=1, errors="ignore"))
    # ensure columns
    for feat in feature_names:
        if feat not in X_sample.columns:
            X_sample[feat] = 0
    X_sample = X_sample[list(feature_names)]
    preds = model.predict(X_sample)
    sample["Predicted_Severity"] = preds
    st.dataframe(sample.head(30), use_container_width=True)
    csv = sample.to_csv(index=False).encode("utf-8")
    st.download_button("💾 Download sample with predictions", csv, "predictions_sample.csv", "text/csv")

# ---------------------------
# Footer small note
st.markdown("<div class='muted' style='margin-top:18px'>Built for demo & presentation. For production, validate with domain data and apply explainability (SHAP) + monitoring.</div>", unsafe_allow_html=True)
