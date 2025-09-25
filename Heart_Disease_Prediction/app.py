import streamlit as st
import pandas as pd
import numpy as np
import pickle 
import base64
import os

# Function to create a download link for the CSV file
def get_binary_file_downloader_html(df, filename = "file.csv", text = "Download CSV file"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'

    return href 
# ------------------------- #
# Page Config
# ------------------------- #
st.set_page_config(page_title="❤️ Heart Disease Predictor", layout="wide")
st.title("❤️ Heart Disease Predictor")
st.write("This model predicts whether a person has heart disease based on health parameters.")

# ------------------------- #
# Sidebar Info
# ------------------------- #
with st.sidebar:
    st.info("ℹ️ Fill in the details and click **Submit** to check your risk of heart disease.")
    st.markdown("⚠️ **Disclaimer:** This is for educational purposes only and not a medical diagnosis.")

# ------------------------- #
# Tabs
# ------------------------- #
tab1, tab2, tab3 = st.tabs(["Prediction", "Bulk Prediction", "Model Information"])

# ------------------------- #
# Tab1
# ------------------------- #
with tab1:
    st.header("Single Prediction")
    st.write("Enter the health parameters to predict heart disease risk:")

    # --- Arrange in Columns ---
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=0, max_value=120)
        sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x==0 else "Male")
        chest_pain_type = st.selectbox(
            "Chest Pain Type",
            options=[0, 1, 2, 3],
            format_func=lambda x: {
                0: "Typical Angina",
                1: "Atypical Angina",
                2: "Non-Anginal Pain",
                3: "Asymptomatic"
            }[x]
        )
        resting_blood_pressure = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300)
        cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=0, max_value=1000)
        fasting_blood_sugar = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl",
            options=[0, 1],
            format_func=lambda x: "≤120 mg/dl" if x==0 else ">120 mg/dl"
        )

    with col2:
        resting_ecg = st.selectbox(
            "Resting ECG Results",
            options=[0, 1, 2],
            format_func=lambda x: {
                0: "Normal",
                1: "ST-T Wave Abnormality",
                2: "Left Ventricular Hypertrophy"
            }[x]
        )
        max_heart_rate_achieved = st.number_input("Max Heart Rate Achieved", min_value=0, max_value=250)
        exercise_induced_angina = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "No" if x==0 else "Yes")
        st_depression = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, step=0.1)
        slope_of_st = st.selectbox(
            "Slope of Peak Exercise ST Segment",
            options=[0, 1, 2],
            format_func=lambda x: {
                0: "Upsloping",
                1: "Flat",
                2: "Downsloping"
            }[x]
        )
        number_of_major_vessels = st.selectbox(
    "Number of Major Vessels Blocked",
    options=[0, 1, 2, 3],
    format_func=lambda x: {
        0: "No major vessels are significantly blocked",
        1: "One major vessel is significantly blocked",
        2: "Two major vessels are significantly blocked",
        3: "Three major vessels are significantly blocked (often indicating triple-vessel disease)"
    }[x]
        )
        thalassemia = st.selectbox(
            "Thalassemia",
            options=[0, 1, 2, 3],
            format_func=lambda x: {
                0: "Normal",
                1: "Fixed Defect",
                2: "Reversible Defect",
                3: "Unknown"
            }[x]
        )

# Create a DataFrame with user inputs
input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'chest_pain_type': [chest_pain_type],
    'resting_blood_pressure': [resting_blood_pressure],
    'cholesterol': [cholesterol],
    'fasting_blood_sugar': [fasting_blood_sugar],
    'resting_ecg': [resting_ecg],
    'max_heart_rate_achieved': [max_heart_rate_achieved],
    'exercise_induced_angina': [exercise_induced_angina],
    'st_depression': [st_depression],
    'slope_of_st': [slope_of_st],
    'number_of_major_vessels': [number_of_major_vessels],
    'thalassemia': [thalassemia]

})

### Now we Should Load the Model and Make Predictions
algonames = [
    'Decision Tree',
    'K-Nearest Neighbors', 
    'Logistic Regression', 
    'Naive Bayes', 
    'RandomForest with Hyperparameter Tuning',
    'Random Forest',   
    'Support Vector Classifier', 
    
 ]
modelsnames = [
     'DecisionTree.pkl',
     'KNN.pkl', 
     'LogisticRegression.pkl', 
     'NaiveBayes.pkl', 
     'RandomForest_RandomCV.pkl',
     'RandomForest.pkl', 
     'SVC.pkl', 
     ]

predictions = []

def predict_heart_disease(data):
    # ✅ Rename columns to match training feature names
    data = data.rename(columns={
        "number_of_major_vessels": "num_major_vessels",
        "resting_ecg": "rest_ecg",
        "slope_of_st": "st_slope"
    })

    predictions.clear()  # reset before each prediction
    for modelname in modelsnames:
        model_path = os.path.join(os.path.dirname(__file__), "pickle_files", modelname)
        prediction = model.predict(data)
        predictions.append(prediction)
    return predictions  # Return all Predictions


### Create a Submit button to make predictions
if st.button("Submit"):
    st.subheader("Results...")
    st.markdown("---")

    results = predict_heart_disease(input_data)

    for i, pred in enumerate(results):
        st.subheader(algonames[i])
        if pred[0] == 0:
           st.write(" ✅ No Heart Disease Detected")
        else:
            st.write("⚠️ Heart Disease Detected")
        st.markdown("---")

# ------------------------- #
# Tab2
# ------------------------- #
with tab2:
    st.title("Upload CSV File")

    st.subheader("Instructions to note before  uploading the file: ")
    st.info("""
    1. No Nan values are allowed.
    2. The CSV file should contain the following columns:\n
            - age 
            - sex
            - chest_pain_type
            - resting_blood_pressure
            - cholesterol
            - fasting_blood_sugar
            - rest_ecg
            - max_heart_rate_achieved
            - exercise_induced_angina
            - st_depression
            - st_Slope
            - num_major_vessels
            - thalassemia
    3. Ensure that the column names match exactly.
    4. Feature value conventions:\n
           - Age: age of the patient in years (e.g., 45)
           - Sex: sex of the patient (1 = male, 0 = female)
           - ChestPainType: chest pain type (
                0: "Typical Angina",
                1: "Atypical Angina",
                2: "Non-Anginal Pain",
                3: "Asymptomatic") 
           - RestingBloodPressure: resting blood pressure in mm Hg (e.g., 120)
           - FastingBloodSugar: fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
           - RestingECG: resting electrocardiographic results (0 = normal; 1 = having ST-T wave abnormality; 2 = showing probable or definite left ventricular hypertrophy)
           - MaxHeartRateAchieved: maximum heart rate achieved (e.g., 150)
           - ExerciseAngina: exercise-induced angina (1 = yes; 0 = no)
           - st_depression: ST depression induced by exercise relative to rest (e.g., 1.4)
           - SlopeOfST: the slope of the peak exercise ST segment (0 = upsloping; 1 = flat; 2 = downsloping )
           - NumberOfMajorVessels: number of major vessels (0-3) colored by fluoroscopy
           - Thalassemia: thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect; 0 = unknown)
               
""")
    ### Create a file uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read the uploaded CSV file
        input_data = pd.read_csv(uploaded_file)
       

        # Ensure that the input dataframe matches the model's expected columns and format.
        column_mapping = { 
           "age": "age",
           "sex": "sex",
           "ChestPainType": "chest_pain_type",
           "RestingBP": "resting_blood_pressure",
           "Cholesterol": "cholesterol",
           "FastingBS": "fasting_blood_sugar",
           "RestingECG": "rest_ecg",
           "MaxHR": "max_heart_rate_achieved",
           "ExerciseAngina": "exercise_induced_angina",
           "Stdepression": "st_depression",
           "ST_Slope": "st_slope",
           "NumberofMajorVessels": "num_major_vessels",
           "Thalassemia": "thalassemia"
        }
        
        # Reanme columns to match training
        input_data.rename(columns=column_mapping, inplace=True)
        # Ensure same order of features as training
        expected_features = list(column_mapping.values())

        if set(expected_features).issubset(input_data.columns):
            st.success("File Successfully Uploaded")

            ### Models List (must be in the same order as modelsnames)
            algonames = [
                'Decision Tree',
                'K-Nearest Neighbors',
                'Logistic Regression',
                'Naive Bayes',
                'Random Forest',
                'SVC'
            ]
            modelsnames = [
                'DecisionTree.pkl',
                'KNN.pkl',
                'LogisticRegression.pkl',
                'NaiveBayes.pkl',
                'RandomForest.pkl',
                'SVC.pkl'
            ]
            # Predict with each model and add column

            for algo, model_file in zip(algonames, modelsnames):
                model = pickle.load(open(model_file, 'rb'))
                input_data[f'Prediction_{algo}'] = model.predict(input_data[expected_features])  
           
            # Save results
            input_data.to_csv('PredictedHeart_AllModels.csv', index=False)

            ### Display the predictions
            st.subheader("Predictions from All Models")
            st.write(input_data)

            st.markdown(
                get_binary_file_downloader_html(
                    input_data,
                    filename="PredictedHeart_ALLModels.csv", 
                    text="📥 Download Predictions (All_Models)"),
                    unsafe_allow_html=True)
        else:
            st.warning(f"⚠️ The uploaded file does not contain the required columns. Please ensure it has the following columns: {', '.join(expected_features)}")

    else:
       st.info("Upload a CSV file to get predictions")

# ------------------------- #
# Tab3
# ------------------------- #

with tab3:
    import plotly.express as px
    import pandas as pd

    # Accuracy data
    data = {
        'Random Forest (Tuned)': 81.97,
        'Random Forest': 78.69,
        'Naive Bayes': 77.05,
        'Logistic Regression': 77.05,
        'Decision Tree': 73.77,
        'SVC': 68.85
    }

    # Create DataFrame
    df = pd.DataFrame(list(data.items()), columns=['Model', 'Accuracy'])

    # Create colorful bar chart
    fig = px.bar(
        df,
        x='Model',
        y='Accuracy',
        text='Accuracy',
        color='Accuracy',  # color bars based on value
        color_continuous_scale='Viridis',  # you can use 'Plasma', 'Cividis', etc.
        title='Model Accuracy Comparison',
        template='plotly_dark',  # dark theme, can also use 'plotly_white'
    )

    # Add styling
    fig.update_traces(
        texttemplate='%{text:.2f}%', 
        textposition='outside', 
        marker_line_color='black', 
        marker_line_width=1.5
    )
    fig.update_layout(
        yaxis=dict(title='Accuracy (%)', range=[0, 100]),
        xaxis=dict(title='Models'),
        title=dict(x=0.5, xanchor='center', font=dict(size=22)),
        uniformtext_minsize=12,
        uniformtext_mode='hide',
        bargap=0.4
    )

    st.plotly_chart(fig, use_container_width=True)
 