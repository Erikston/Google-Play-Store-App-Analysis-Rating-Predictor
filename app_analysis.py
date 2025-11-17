# ---- DATA LOADING & CLEANING ----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import streamlit as st

@st.cache_data
def load_data():
    df = pd.read_csv("googleplaystore.csv")
    df = df.drop_duplicates()
    df['Installs'] = df['Installs'].str.replace('[+,]', '', regex=True)
    df['Installs'] = pd.to_numeric(df['Installs'], errors='coerce')

    def clean_size(size):
        if pd.isna(size) or size == 'Varies with device':
            return np.nan
        size = str(size).replace(',', '')
        if 'k' in size:
            return float(size.replace('k', '')) / 1024
        elif 'M' in size:
            return float(size.replace('M', ''))
        return np.nan

    df['Size'] = df['Size'].apply(clean_size)
    df['Price'] = df['Price'].str.replace('[$]', '', regex=True)
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce').fillna(0)

    numeric_cols = ['Rating', 'Reviews', 'Installs', 'Price', 'Size']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['Rating'] = df['Rating'].fillna(df['Rating'].median())
    df['Size'] = df['Size'].fillna(df['Size'].median())
    return df

def predict_ratings(df):
    st.header("🤖 Rating Prediction Model")
    features = ['Category', 'Reviews', 'Installs', 'Price', 'Size']
    X = df[features]
    X = pd.get_dummies(X, columns=['Category'], drop_first=True)
    y = df['Rating']

    # FIXED: closing parenthesis added
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"Model MSE: {mse:.2f}")
    st.write(f"Average Error: {np.sqrt(mse):.2f} stars")
    ...
