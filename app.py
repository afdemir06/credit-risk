import streamlit as st
import requests
import pandas as pd
import json
import io
from src.utils import download_results

st.set_page_config(
    page_title="Credit Risk Engine",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background-color: #080C10;
    color: #C8D6E5;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem 3rem; max-width: 1200px; }

.hero {
    border-left: 3px solid #FF4D6D;
    padding: 1.2rem 0 1.2rem 1.6rem;
    margin-bottom: 2.5rem;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    color: #F0F6FF;
    letter-spacing: -0.03em;
    margin: 0 0 0.3rem 0;
}
.hero p {
    font-size: 0.78rem;
    color: #4A6080;
    margin: 0;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #FF4D6D;
    margin-bottom: 0.8rem;
    margin-top: 2rem;
}

.divider {
    height: 1px;
    background: linear-gradient(90deg, #FF4D6D 0%, #1A2535 60%);
    margin: 2rem 0;
    border: none;
}

/* Widgets */
div[data-testid="stFileUploader"] {
    background: #0D1117 !important;
    border: 1px dashed #1A2535 !important;
    border-radius: 4px !important;
}
div[data-testid="stFileUploader"]:hover { border-color: #FF4D6D !important; }

label[data-testid="stWidgetLabel"] p {
    font-size: 0.7rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #4A6080 !important;
}
div[data-testid="stSelectbox"] > div > div {
    background: #0D1117 !important;
    border-color: #1A2535 !important;
    color: #C8D6E5 !important;
    border-radius: 4px !important;
}
div[data-testid="stTextInput"] input {
    background: #0D1117 !important;
    border-color: #1A2535 !important;
    color: #C8D6E5 !important;
    border-radius: 4px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
}
div[data-testid="stTextInput"] input:focus { border-color: #FF4D6D !important; }

div[data-testid="stButton"] > button {
    background: #FF4D6D !important;
    color: #F0F6FF !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 2px !important;
    padding: 0.65rem 2rem !important;
    margin-top: 0.8rem !important;
}
div[data-testid="stButton"] > button:hover { opacity: 0.85 !important; }

div[data-testid="stTabs"] button {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #4A6080 !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #FF4D6D !important;
    border-bottom-color: #FF4D6D !important;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #1A2535 !important;
    border-radius: 4px !important;
}
div[data-testid="stAlert"] {
    background: #0D1117 !important;
    border-radius: 4px !important;
    font-size: 0.78rem !important;
}
div[data-testid="stCheckbox"] label p {
    font-size: 0.76rem !important;
    color: #C8D6E5 !important;
    text-transform: none !important;
    letter-spacing: 0.04em !important;
}
div[data-testid="stMetric"] {
    background: #0D1117 !important;
    border: 1px solid #1A2535 !important;
    border-radius: 4px !important;
    padding: 1rem !important;
}
div[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
    color: #F0F6FF !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>Credit Risk Engine</h1>
    <p>XGBoost · SHAP Explainability · Probability of Default</p>
</div>
""", unsafe_allow_html=True)

if "ratio_pairs" not in st.session_state:
    st.session_state["ratio_pairs"] = []

if "features" not in st.session_state:
    response_model_info=requests.get(
        url="http://api:8000/model/info"
    )
    if response_model_info.status_code==200:
        if response_model_info.json()["model_exist"]:
            st.session_state["features"]=response_model_info.json()["features"]
            st.session_state["metrics"]=response_model_info.json()["metrics"]
            st.session_state["target_column"]=response_model_info.json()["target_column"]

st.markdown('<div class="section-label">01 — Training Data</div>', unsafe_allow_html=True)
train_data = st.file_uploader("Upload training file", type=["csv", "xlsx"], label_visibility="collapsed")

if train_data is None:
    st.markdown("""
    <div style="background:#0D1117;border:1px dashed #1A2535;border-radius:4px;
                padding:2.5rem;text-align:center;margin-top:0.5rem;">
        <span style="font-size:0.76rem;color:#4A6080;letter-spacing:0.08em;">
            Drop a CSV or XLSX file · Requires a binary target column (0 = good, 1 = default)
        </span>
    </div>
    """, unsafe_allow_html=True)
else:
    train_data.seek(0)
    train_df = pd.read_csv(train_data) if train_data.name.endswith(".csv") else pd.read_excel(train_data)

    st.markdown(f"""
    <div style="display:flex;gap:0.6rem;align-items:center;margin:0.8rem 0 1.6rem 0;">
        <span style="background:#0D1117;border:1px solid #1A2535;border-radius:2px;
                     padding:0.3rem 0.8rem;font-size:0.7rem;color:#FF4D6D;">
            {train_data.name}
        </span>
        <span style="font-size:0.7rem;color:#4A6080;">
            {len(train_df):,} rows · {len(train_df.columns)} columns
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">02 — Configuration</div>', unsafe_allow_html=True)
    target_column = st.selectbox("Target column", options=train_df.columns)

    fill_strategies = {}
    nan_cols = [i for i in train_df.columns if train_df[i].isnull().mean() > 0.05]
    if nan_cols:
        st.markdown(f"""
        <div style="font-size:0.72rem;color:#4A6080;margin:0.8rem 0 0.4rem 0;letter-spacing:0.06em;">
            {len(nan_cols)} column(s) with >5% missing values — select fill strategy
        </div>
        """, unsafe_allow_html=True)
        cols = st.columns(min(3, len(nan_cols)))
        for i, col in enumerate(nan_cols):
            with cols[i % 3]:
                fill_strategies[col] = st.selectbox(f"'{col}'", options=["median", "mode", "zero"])

    if st.checkbox("Add ratio features (e.g. debt / income)"):
        rc1, rc2 = st.columns(2)
        with rc1:
            first_column = st.selectbox("Numerator column", options=train_df.columns)
        with rc2:
            second_column = st.selectbox("Denominator column", options=train_df.columns)
        if st.button("Add ratio →"):
            if first_column != second_column:
                st.session_state["ratio_pairs"].append([first_column, second_column])
        if st.session_state["ratio_pairs"]:
            pairs_display = " · ".join([f"{p[0]}/{p[1]}" for p in st.session_state["ratio_pairs"]])
            st.markdown(f"""
            <div style="font-size:0.72rem;color:#00E5B4;margin-top:0.4rem;letter-spacing:0.06em;">
                Active pairs: {pairs_display}
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    if st.button("Train Model →"):
        with st.spinner("Training — running Optuna hyperparameter search..."):
            train_data.seek(0)
            response_train = requests.post(
                url="http://api:8000/train",
                files={"file":(train_data.name,train_data.getvalue(),"text/csv")},
                data={
                    "fill_strategies": json.dumps(fill_strategies),
                    "ratio_pairs": json.dumps(st.session_state["ratio_pairs"]),
                    "target_column": target_column,
                },
            )
        if response_train.status_code != 200:
            st.error(f"Training failed: {response_train.json().get('detail', 'Unknown error')}")
        else:
            st.rerun()

    if "features" in st.session_state:    
        metrics=st.session_state["metrics"]

        st.markdown('<div class="section-label">Training Results</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("AUC-ROC", f"{metrics['auc_score']:.3f}")
        with m2:
            st.metric("Precision", f"{metrics['precision']:.3f}")
        with m3:
            st.metric("Recall", f"{metrics['recall_score']:.3f}")
        with m4:
            st.metric("F1 Score", f"{metrics['f1_score']:.3f}")

        st.markdown('<div class="section-label">03 — Prediction</div>', unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["◈  Single Applicant", "◈  Batch Scoring"])

        with tab1:
            st.markdown("""
            <div style="font-size:0.72rem;color:#4A6080;margin:0.8rem 0 1.2rem 0;letter-spacing:0.06em;">
                Enter applicant details — system will return probability of default and feature attribution
            </div>
            """, unsafe_allow_html=True)

            client_dict = {}
            features = st.session_state["features"]
            n_cols = 3
            rows = [features[i:i+n_cols] for i in range(0, len(features), n_cols)]
            for row in rows:
                cols = st.columns(n_cols)
                for j, feature in enumerate(row):
                    with cols[j]:
                        client_dict[feature] = st.text_input(feature)

            for key, value in client_dict.items():
                try:
                    client_dict[key] = float(value)
                except:
                    pass

            if st.button("Score Applicant →"):
                with st.spinner("Scoring..."):
                    response_single_predict = requests.post(
                        url="http://api:8000/predict/single",
                        json={"data": client_dict},
                    )
                if response_single_predict.status_code != 200:
                    st.error(f"Prediction failed: {response_single_predict.json().get('detail', 'Unknown error')}")
                else:
                    st.session_state["single_results"] = response_single_predict.json()["results"]

            if "single_results" in st.session_state:
                pd_score = st.session_state["single_results"]["predict_proba"][0]
                feature_importances = st.session_state["single_results"]["feature_importances"]

                color = "#FF4D6D" if pd_score >= 50 else "#00E5B4"
                risk = "HIGH RISK" if pd_score >= 50 else "LOW RISK"

                st.markdown(f"""
                <div style="background:#0D1117;border:1px solid #1A2535;border-left:3px solid {color};
                            border-radius:4px;padding:1.4rem 1.6rem;margin:1.2rem 0;">
                    <div style="font-family:'Syne',sans-serif;font-size:2.4rem;font-weight:800;
                                color:{color};line-height:1;">{pd_score:.1f}%</div>
                    <div style="font-size:0.7rem;color:#4A6080;letter-spacing:0.15em;
                                text-transform:uppercase;margin-top:0.3rem;">
                        Probability of Default · {risk}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown('<div class="section-label">Feature Attribution (SHAP)</div>', unsafe_allow_html=True)
                shap_df = pd.DataFrame.from_dict(
                    feature_importances, orient="index", columns=["SHAP Value"]
                ).sort_values("SHAP Value", ascending=False)
                st.bar_chart(shap_df)

        with tab2:
            st.markdown("""
            <div style="font-size:0.72rem;color:#4A6080;margin:0.8rem 0 1.2rem 0;letter-spacing:0.06em;">
                Upload a file with multiple applicants — results will include PD score for each row
            </div>
            """, unsafe_allow_html=True)

            prediction_batch_file = st.file_uploader(
                "Upload scoring file", type=["csv", "xlsx"], label_visibility="collapsed"
            )

            if prediction_batch_file is not None:
                batch_df = pd.read_csv(prediction_batch_file) if prediction_batch_file.name.endswith(".csv") \
                else pd.read_excel(prediction_batch_file)
                if st.session_state.get("target_column") in batch_df.columns:
                    batch_df=batch_df.drop(columns=[st.session_state["target_column"]])
                if set(batch_df.columns)!=set(st.session_state["features"]):
                    st.error("Please ensure your data columns are the same as train data")
                else:
                    batch_df=batch_df[st.session_state["features"]]
                    
                    st.markdown(f"""
                    <div style="font-size:0.7rem;color:#4A6080;margin:0.6rem 0 1rem 0;">
                        {len(batch_df):,} applicants loaded
                    </div>
                    """, unsafe_allow_html=True)

                    if st.button("Score Batch →"):
                        buffer=io.BytesIO()
                        batch_df.to_csv(buffer,index=False)
                        buffer.seek(0)
                        with st.spinner(f"Scoring {len(batch_df):,} applicants..."):
                            response_batch_predict = requests.post(
                                url="http://api:8000/predict/batch",
                                files={"file":("batch.csv",buffer,"text/csv")},
                            )
                        if response_batch_predict.status_code != 200:
                            st.error(f"Batch scoring failed: {response_batch_predict.json().get('detail', 'Unknown error')}")
                        else:
                            st.session_state["batch_results"] = response_batch_predict.json()["results"]

                    if "batch_results" in st.session_state:
                        batch_predict_proba = st.session_state["batch_results"]["predict_proba"]
                        batch_feature_importances = st.session_state["batch_results"]["feature_importances"]

                        batch_df["PD Score (%)"] = batch_predict_proba
                        batch_df["Risk"] = batch_df["PD Score (%)"].apply(
                            lambda x: "High" if x >= 50 else "Low"
                        )

                        bm1, bm2, bm3 = st.columns(3)
                        with bm1:
                            st.metric("Total Applicants", f"{len(batch_df):,}")
                        with bm2:
                            st.metric("High Risk", f"{(batch_df['Risk'] == 'High').sum():,}")
                        with bm3:
                            st.metric("Low Risk", f"{(batch_df['Risk'] == 'Low').sum():,}")

                        st.markdown('<div class="section-label">Results</div>', unsafe_allow_html=True)
                        st.dataframe(batch_df, use_container_width=True)

                        st.markdown('<div class="section-label">Feature Attribution (SHAP)</div>', unsafe_allow_html=True)
                        shap_df = pd.DataFrame.from_dict(
                            batch_feature_importances, orient="index", columns=["SHAP Value"]
                        ).sort_values("SHAP Value", ascending=False)
                        st.bar_chart(shap_df)

                        st.download_button(
                            label="Download batch results",
                            data=download_results(batch_df),
                            file_name="scoring_results.csv",
                            mime="text/csv"
                        )
            else:
                st.markdown("""
                <div style="background:#0D1117;border:1px dashed #1A2535;border-radius:4px;
                            padding:2rem;text-align:center;margin-top:0.5rem;">
                    <span style="font-size:0.76rem;color:#4A6080;letter-spacing:0.08em;">
                        Upload a file to begin batch scoring
                    </span>
                </div>
                """, unsafe_allow_html=True)