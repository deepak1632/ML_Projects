# model_pipeline.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

def preprocess_data(df, target_col=None):
    df = df.copy()
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
    if target_col and target_col in df.columns:
        y = df[target_col]
        X = df.drop(columns=[target_col])
    else:
        y = None
        X = df
    return X, y

def transform_features(X_train, X_test=None, pt=None, scaler=None):
    """
    Transform features using PowerTransformer + StandardScaler.
    If pt/scaler are None, fit them on X_train.
    If X_test is provided, transform it using the fitted transformers.
    """
    # Fit transformers if not provided
    if pt is None:
        pt = PowerTransformer(method='yeo-johnson')
        X_train_trans = pt.fit_transform(X_train)
    else:
        X_train_trans = pt.transform(X_train)
    
    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_trans)
    else:
        X_train_scaled = scaler.transform(X_train_trans)

    if X_test is not None:
        X_test_trans = pt.transform(X_test)
        X_test_scaled = scaler.transform(X_test_trans)
        return X_train_scaled, X_test_scaled, pt, scaler
    else:
        return X_train_scaled, pt, scaler



def get_model(name):
    models = {
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "XGBoost": XGBRegressor(random_state=42),
        "LightGBM": LGBMRegressor(random_state=42)
    }
    return models.get(name)

def train_model(X_train, y_train, model_name):
    model = get_model(model_name)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    return rmse, r2

def predict(model, X_new):
    return model.predict(X_new)
