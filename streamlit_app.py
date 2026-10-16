
import streamlit as st
import requests
import numpy as np

st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="⚙️",
    layout="centered"
)

st.title("⚙️ Predictive Maintenance System")

st.write(
    "Predict engine failure risk using historical industrial sensor measurements "
    "and a trained LSTM deep learning model."
)

st.info("Input required: 20 time steps × 21 sensor features.")

API_URL = "http://api:8000/predict"

st.subheader("Engine Sensor Data")

uploaded_file = st.file_uploader(
    "Upload a CSV file containing 20 rows and 21 sensor columns",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        window = np.loadtxt(uploaded_file, delimiter=",", ndmin=2)

        if window.shape != (20, 21):
            st.error(
                f"Invalid shape: {window.shape}. "
                "Expected exactly 20 rows and 21 columns."
            )
        elif not np.isfinite(window).all():
            st.error("The file contains invalid or missing numerical values.")
        else:
            st.success("Sensor data loaded successfully.")

            with st.expander("Preview sensor data"):
                st.dataframe(window)

            if st.button("Predict Risk", type="primary"):
                try:
                    response = requests.post(
                        API_URL,
                        json={"window": window.tolist()},
                        timeout=30
                    )
                    response.raise_for_status()
                    result = response.json()

                    st.subheader("Prediction Results")
                    st.json(result)

                except requests.exceptions.RequestException as e:
                    st.error(f"API request failed: {e}")

    except (ValueError, OSError) as e:
        st.error(f"Could not read the CSV file: {e}")
else:
    st.warning("Upload a CSV file before making a prediction.")
