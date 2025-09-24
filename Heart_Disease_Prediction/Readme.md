# ❤️ Heart Disease Prediction Project

## 📌 Overview
This project predicts the likelihood of heart disease based on patient health parameters.  
It applies multiple **Machine Learning algorithms** and compares their performance.  
The project also includes a **Streamlit web app** for user interaction, where users can input health details and get predictions in real time.

---

## 🎯 Objectives
- Analyze patient health data for heart disease risk factors.
- Build and evaluate ML models for classification.
- Compare algorithms based on performance metrics.
- Deploy a **Streamlit-based user-friendly application**.
- Save trained models using **Pickle** for reusability.

---

## 📂 Project Structure
Heart_Diseases_Project/
│── app.py # Streamlit app
│── models/ # Saved pickle models
│ ├── Decision_Tree.pkl
│ ├── RandomForest.pkl
│ ├── LogisticRegression.pkl
│ ├── SVM.pkl
│ ├── KNN.pkl
│── data/ # Dataset folder
│ └── heart.csv
│── notebooks/ # Jupyter notebooks for analysis
│ └── EDA_Modeling.ipynb
│── requirements.txt # Dependencies
│── README.md # Project documentation
