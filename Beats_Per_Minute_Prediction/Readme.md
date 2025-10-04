# 🎧 Beats Per Minute (BPM) Prediction App
#### Thumbnail
![App Screenshot]("https://github.com/user-attachments/assets/1de65839-d713-4fa5-9087-91a5dbd23ab2")  

📘 Project Overview

The BPM Prediction App is a machine learning project that predicts the Beats Per Minute (BPM) of songs based on their audio features. Built with Python and Streamlit, the app allows both single-song predictions and batch predictions from CSV files. Users can choose from multiple models, including XGBoost, LightGBM, Gradient Boosting, or an Ensemble of all models for more accurate results.

**This project was inspired by the Kaggle Playground Series – Season 5, Episode 9 challenge on predicting the BPM of songs using synthetic tabular data.**

---

## 🚀 Features

* Single Song Prediction: Enter audio feature values manually to predict BPM.
* Batch Prediction: Upload CSV files to predict BPM for multiple songs at once.
* Train & Evaluate Models: Upload custom datasets to train, validate, and evaluate models in the app.
* Multiple Models Supported: Decision Tree, Gradient Boosting, XGBoost, LightGBM, and Ensemble.
* Download Predictions: Export batch predictions as CSV files.
* Interactive Dashboard: Modern, gradient-themed UI using Streamlit.

**📊 Audio Features Used**

- RhythmScore	: Score indicating rhythmic complexity
- AudioLoudness	: Loudness level of the track
- VocalContent :  Vocal presence score
- AcousticQuality : Acoustic quality metric
- InstrumentalScore : Instrumental richness score
- LivePerformanceLikelihood : Likelihood of live performance style
- MoodScore	: Mood intensity score
- TrackDurationMs :	Duration of track in milliseconds
- Energy : Overall energy of the track

---
## 🛠 Tech Stack

* Programming Language: Python
* Web Framework: Streamlit
* Machine Learning Models: XGBoost, LightGBM, Gradient Boosting, Decision Tree
* Preprocessing & Scaling: Scikit-learn (StandardScaler, PowerTransformer)
* Data Handling: Pandas, NumPy
* Model Storage: Pickle
---

## 🔹 Installation

1. Clone this repository:

```bash
git clone https://github.com/<your-username>/bpm-prediction-app.git
cd bpm-prediction-app

