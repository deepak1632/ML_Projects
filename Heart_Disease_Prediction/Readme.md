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
```
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
```
---

---

## ⚙️ Technologies Used
- **Python** (3.8+)
- **Libraries**: pandas, numpy, matplotlib, seaborn, scikit-learn, pickle, streamlit
- **Machine Learning Models**:
  - Decision Tree
  - Random Forest
  - Logistic Regression
  - Support Vector Classifier (SVC)
  - K-Nearest Neighbors (KNN)

---

## 📊 Methodology
1. **Data Preprocessing**
   - Handled missing values and categorical encoding.
   - Standardized numerical features.
2. **Exploratory Data Analysis (EDA)**
   - Distribution plots, correlations, and feature importance.
3. **Model Training**
   - Applied multiple ML algorithms.
   - Hyperparameter tuning with `GridSearchCV` and `RandomizedSearchCV`.
4. **Model Evaluation**
   - Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC.
   - Compared all models.
5. **Model Saving**
   - Best models stored as `.pkl` files using `pickle`.
6. **Deployment**
   - Streamlit app for real-time prediction.

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository
```
bash
git clone https://github.com/yourusername/Heart_Diseases_Project.git
cd Heart_Diseases_Project

### 2️⃣ Install Dependencies
pip install -r requirements.txt

### 3️⃣ Run the Streamlit App
streamlit run app.py

### 4️⃣ Access in Browser
Open the URL shown in terminal (default: http://localhost:8501).
```
---

---
### 🔮 Future Improvements

* Add Deep Learning models (e.g., ANN).
* Deploy app on Streamlit Cloud / Heroku.
* Integrate with real-world health datasets.
* Improve UI/UX with visual explanations.
