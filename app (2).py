
import streamlit as st
import pandas as pd
import joblib
import json

st.set_page_config(page_title="Delivery Delay Prediction", page_icon="🚚")

st.title("🚚 Delivery Delay Prediction App")

@st.cache_resource
def load_artifacts():
    model = joblib.load("trained_model.joblib")
    with open("feature_columns.json") as f:
        features = json.load(f)
    return model, features

model, feature_columns = load_artifacts()

uploaded_file = st.file_uploader("Upload CSV for prediction", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
    else:
        df = df[feature_columns]
        preds = model.predict(df)

        output = df.copy()
        output["Delivery_Delay_Prediction"] = preds
        output["Status"] = output["Delivery_Delay_Prediction"].map({0:"On Time",1:"Delayed"})

        st.success("Prediction completed")
        st.dataframe(output)

        st.download_button(
            "Download Predictions",
            output.to_csv(index=False),
            "predictions.csv",
            "text/csv"
        )
else:
    st.info("Upload CSV file to begin")
