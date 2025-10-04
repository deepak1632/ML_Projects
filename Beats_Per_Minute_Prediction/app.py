# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import base64
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from model_pipeline import preprocess_data, transform_features, train_model, evaluate_model, predict
from sklearn.model_selection import train_test_split
import time

# ------------------------------
# 🎯 Page Configuration
# ------------------------------
st.set_page_config(
    page_title="🎵 BPM Prediction App",
    page_icon="🎧",
    layout="wide"
)

# ------------------------------
# 🎨 CSS for Gradient & Cards
# ------------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(to right, #fceabb, #f8b500);
}
h1, h2, h3, h4, h5 {
    color: #2c3e50;
}
.card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 3px 3px 15px rgba(0,0,0,0.1);
    text-align:center;
}
.stButton>button {
    background-color: #f8b500;
    color: white;
    border-radius: 10px;
    padding: 0.5em 1em;
    font-weight: bold;
}
.stButton>button:hover {
    background-color: #f39c12;
    color: white;
}
.stSelectbox>div>div>div>select {
    color: #2c3e50;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# 🎨 Header
# ------------------------------
st.markdown("""
<div style='text-align:center;'>
    <h1>🎧 Beats Per Minute (BPM) Prediction App</h1>
    <p style='font-size:18px;'>Predict BPM using song features with XGBoost, LightGBM, Gradient Boosting, and more</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------
# 🧠 Load all models & scalers from pickle folder
# ------------------------------
@st.cache_resource
def load_models_and_scalers(folder_path="pickles_files"):
    models = {}
    scalers = {}
    if not os.path.exists(folder_path):
        st.warning(f"Folder '{folder_path}' not found! Using fallback XGB model.")
        models["fallback_XGB"] = XGBRegressor(
            random_state=42, n_estimators=500, learning_rate=0.05, max_depth=6
        )
        scalers["fallback_XGB"] = StandardScaler()
        return models, scalers

    for file in os.listdir(folder_path):
        if file.endswith(".pkl") and "scaler" not in file.lower():
            model_name = file.replace(".pkl", "")
            with open(os.path.join(folder_path, file), "rb") as f:
                models[model_name] = pickle.load(f)
            # Try to load corresponding scaler if exists
            scaler_file = os.path.join(folder_path, model_name + "_scaler.pkl")
            if os.path.exists(scaler_file):
                with open(scaler_file, "rb") as sf:
                    scalers[model_name] = pickle.load(sf)
            else:
                scalers[model_name] = StandardScaler()  # fallback
    return models, scalers

models, scalers = load_models_and_scalers("Beats_Per_Minute_Prediction/pickles_files")
st.sidebar.success(f"✅ {len(models)} model(s) & scalers loaded from folder")


# ------------------------------
# 🔖 Tabs
# ------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Home", "📊 Single Prediction", "📁 Batch Prediction", "⚙️ Train & Evaluate"
])

feature_names = [
    'RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality',
    'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore',
    'TrackDurationMs', 'Energy'
]

# ------------------------------
# 🏠 Home Tab
# ------------------------------
with tab1:
    st.markdown(
    """
    <div class='card' style='background-color: #f9f9ff; padding: 10px; border-radius: 10px;'>
        <h2 style='color: #0072B5; text-align: center;'>Project Overview 🎵</h2>
    </div>
    """,
    unsafe_allow_html=True
)
    st.markdown("""
        - **Objective:** Predict BPM using song features.
        - **Columns:** RhythmScore, AudioLoudness, VocalContent, AcousticQuality,
          InstrumentalScore, LivePerformanceLikelihood, MoodScore, TrackDurationMs, Energy.
        - **Tech Stack:** Python, Streamlit, XGBoost, LightGBM, CatBoost.
        - **Metric:** RMSE (Root Mean Squared Error)
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # KPI cards
    col1, col2, col3 = st.columns(3)
    col1.metric("Loaded Models", len(models))
    col2.metric("Total Features", len(feature_names))
    col3.metric("Ready to Predict", "✅")

# ------------------------------
# 📊 Single Prediction Tab
# ------------------------------
with tab2:
    st.markdown("<h2 style='color:#2c3e50;'>🎧 Single Song BPM Prediction</h2>", unsafe_allow_html=True)
    st.info("Enter song features to predict BPM.")

    user_input = {}
    for feature in feature_names:
        user_input[feature] = st.number_input(f"{feature}", value=0.0, format="%.4f")

    selected_model = st.selectbox("Select Model (or use Ensemble)", ["Ensemble"] + list(models.keys()))

    if st.button("🔮 Predict BPM", key="single_predict"):
        with st.spinner("Predicting BPM..."):
            input_df = pd.DataFrame([user_input])
            
            if selected_model == "Ensemble":
                preds = []
                for model_name in models.keys():
                    scaler = scalers[model_name]
                    X_scaled = scaler.transform(input_df)
                    preds.append(models[model_name].predict(X_scaled)[0])
                prediction = np.mean(preds)
            else:
                scaler = scalers[selected_model]
                X_scaled = scaler.transform(input_df)
                prediction = models[selected_model].predict(X_scaled)[0]

            st.success(f"🎵 Predicted BPM: **{prediction:.2f}**")
            st.balloons()


# ------------------------------
# 📁 Batch Prediction Tab
# ------------------------------
with tab3:
    st.markdown("<h2 style='color:#2c3e50;'>📁 Batch BPM Prediction</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    selected_model_batch = st.selectbox("Select Model for Batch Prediction", ["Ensemble"] + list(models.keys()), key="batch_model")

    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            st.write("✅ File uploaded successfully!")
            st.write("Preview:", data.head())

            if set(feature_names).issubset(data.columns):
                with st.spinner("Predicting BPM for batch..."):
                    all_preds = []

                    if selected_model_batch == "Ensemble":
                        for model_name in models.keys():
                            scaler = scalers[model_name]
                            X_scaled = scaler.transform(data[feature_names])
                            all_preds.append(models[model_name].predict(X_scaled))
                        preds = np.mean(np.array(all_preds), axis=0)
                    else:
                        scaler = scalers[selected_model_batch]
                        X_scaled = scaler.transform(data[feature_names])
                        preds = models[selected_model_batch].predict(X_scaled)

                    data["Predicted_BPM"] = preds

                    col1, col2 = st.columns(2)
                    col1.metric("Songs in CSV", len(data))
                    col2.metric("Predictions Generated", len(preds))

                    st.write("🎵 Predicted Results:", data.head())
                    csv = data.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="predicted_bpm.csv">📥 Download Predictions</a>'
                    st.markdown(href, unsafe_allow_html=True)
            else:
                st.warning(f"⚠️ CSV must contain columns: {', '.join(feature_names)}")
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
    else:
        st.info("Upload a CSV file to get predictions")

# ------------------------------
# ⚙️ Train & Evaluate Tab
# ------------------------------
with tab4:
    st.markdown("<h2 style='color:#2c3e50;'>⚙️ Train & Evaluate Models</h2>", unsafe_allow_html=True)
    st.sidebar.header("Upload CSV Files")
    train_file = st.sidebar.file_uploader("Upload Training Data", type=["csv"])
    test_file = st.sidebar.file_uploader("Upload Test Data (optional)", type=["csv"])

    if train_file:
        train_df = pd.read_csv(train_file)
        st.write("### Training Data Preview")
        st.dataframe(train_df.head())

        target_col = st.selectbox("Select Target Column", train_df.columns, index=len(train_df.columns)-1)
        model_name = st.selectbox("Choose Model", ["Decision Tree", "Gradient Boosting", "XGBoost", "LightGBM"])

        if st.button("Train Model"):
            with st.spinner("Training model..."):
                # Preprocess
                X, y = preprocess_data(train_df, target_col)
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Transform features and get scaler/pt
                X_train_scaled, X_val_scaled, pt, scaler = transform_features(X_train, X_val)
                
                # Train
                model = train_model(X_train_scaled, y_train, model_name)
                
                # Evaluate
                rmse, r2 = evaluate_model(model, X_val_scaled, y_val)

                # KPI Cards
                col1, col2 = st.columns(2)
                col1.metric("Validation RMSE", f"{rmse:.3f}")
                col2.metric("Validation R² Score", f"{r2:.3f}")
                st.success("✅ Model trained successfully!")

                # Save model & transformers
                import pickle, os
                os.makedirs("pickles_files", exist_ok=True)
                with open(f"pickles_files/{model_name}_model.pkl", "wb") as f:
                    pickle.dump(model, f)
                with open(f"pickles_files/{model_name}_pt.pkl", "wb") as f:
                    pickle.dump(pt, f)
                with open(f"pickles_files/{model_name}_scaler.pkl", "wb") as f:
                    pickle.dump(scaler, f)
                st.info("💾 Model, PowerTransformer, and Scaler saved to 'pickles_files' folder")

                # Predict on test file if uploaded
                if test_file:
                    try:
                        test_df = pd.read_csv(test_file)
                        X_test, _ = preprocess_data(test_df)
                        
                        # Use trained pt and scaler
                        _, X_test_scaled, _, _ = transform_features(X_train, X_test, pt=pt, scaler=scaler)
                        preds = predict(model, X_test_scaled)
                        
                        submission = pd.DataFrame({
                            "id": test_df["id"] if "id" in test_df.columns else range(len(test_df)),
                            "BeatsPerMinute": preds
                        })
                        st.write("### Preview of Predictions")
                        st.dataframe(submission.head())
                        st.download_button(
                            "📥 Download Submission",
                            submission.to_csv(index=False),
                            "submission.csv",
                            "text/csv"
                        )
                    except Exception as e:
                        st.error(f"❌ Error processing test file: {e}")
