import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score


# ----------------------------
# Streamlit Page Config
# ----------------------------
st.set_page_config(page_title="🎵 Beats Prediction Dashboard", layout="wide")

st.title("🎵 Beats Prediction Dashboard")
st.markdown("Predict **tempo (BPM)** of music tracks with ML models.")

