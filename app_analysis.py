import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import streamlit as st

# ---- DATA LOADING & CLEANING ----
def load_data():
    # Load the dataset
    df = pd.read_csv("googleplaystore.csv")
    
    # Drop duplicates
    df = df.drop_duplicates()
    
    # Clean 'Installs' column
    df['Installs'] = df['Installs'].str.replace('[+,]', '', regex=True)
    df['Installs'] = pd.to_numeric(df['Installs'], errors='coerce')
    
    # Clean 'Size' column - more robust handling
    def clean_size(size):
        if pd.isna(size) or size == 'Varies with device':
            return np.nan
        size = str(size).replace(',', '')  # Remove commas
        if 'k' in size:
            return float(size.replace('k', '')) / 1024  # Convert KB to MB
        elif 'M' in size:
            return float(size.replace('M', ''))
        return np.nan
    
    df['Size'] = df['Size'].apply(clean_size)
    
    # Clean 'Price' column
    df['Price'] = df['Price'].str.replace('[$]', '', regex=True)
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce').fillna(0)
    
    # Convert other numeric columns
    numeric_cols = ['Rating', 'Reviews', 'Installs', 'Price', 'Size']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Handle missing values
    df['Rating'] = df['Rating'].fillna(df['Rating'].median())
    df['Size'] = df['Size'].fillna(df['Size'].median())
    
    return df

# ---- EDA & VISUALIZATION ----
def show_eda(df):
    st.header("ðŸ“Š Exploratory Data Analysis")
    
    # 1. Top Categories
    st.subheader("Top 10 App Categories")
    top_cats = df['Category'].value_counts().head(10)
    st.bar_chart(top_cats)
    
    # 2. Ratings Distribution
    st.subheader("App Ratings Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['Rating'], bins=20, kde=True, ax=ax)
    st.pyplot(fig)
    
    # 3. Paid vs Free Apps
    st.subheader("Free vs Paid Apps")
    paid_free = df['Type'].value_counts()
    st.bar_chart(paid_free)
    
    # 4. Size vs Rating
    st.subheader("App Size vs Rating")
    fig, ax = plt.subplots()
    sns.scatterplot(x='Size', y='Rating', data=df, alpha=0.3, ax=ax)
    st.pyplot(fig)

# ---- MACHINE LEARNING (Rating Prediction) ----
def predict_ratings(df):
    st.header("ðŸ¤– Rating Prediction Model")
    
    # Prepare features
    features = ['Category', 'Reviews', 'Installs', 'Price', 'Size']
    X = df[features]
    
    # One-hot encode categorical features
    X = pd.get_dummies(X, columns=['Category'], drop_first=True)
    
    # Target variable
    y = df['Rating']
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"Model MSE: {mse:.2f}")
    st.write(f"Average Error: {np.sqrt(mse):.2f} stars")
    
    # Prediction Interface
    st.subheader("Try Predicting a New App!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        category = st.selectbox("Category", df['Category'].unique())
        reviews = st.number_input("Number of Reviews", min_value=0, value=1000)
        installs = st.number_input("Installs", min_value=0, value=10000)
    
    with col2:
        price = st.number_input("Price ($)", min_value=0.0, value=0.0, step=0.99)
        size = st.number_input("Size (MB)", min_value=0.0, value=10.0)
    
    if st.button("Predict Rating"):
        input_data = pd.DataFrame({
            'Category': [category],
            'Reviews': [reviews],
            'Installs': [installs],
            'Price': [price],
            'Size': [size]
        })
        
        # One-hot encode the input
        input_data = pd.get_dummies(input_data, columns=['Category'], drop_first=True)
        
        # Align columns with training data
        missing_cols = set(X.columns) - set(input_data.columns)
        for col in missing_cols:
            input_data[col] = 0
        input_data = input_data[X.columns]
        
        prediction = model.predict(input_data)
        st.success(f"Predicted Rating: {prediction[0]:.1f} â­ (out of 5)")

# ---- STREAMLIT APP ----
def main():
    st.title("ðŸ“± Google Play Store App Analysis")
    
    # Load data
    df = load_data()
    
    # Sidebar menu
    menu = st.sidebar.selectbox("Menu", ["Dataset Overview", "EDA", "Rating Prediction"])
    
    if menu == "Dataset Overview":
        st.header("Dataset Overview")
        st.write(f"Total Apps: {len(df)}")
        st.write("First 10 rows:")
        st.dataframe(df.head(10))
        st.write("Summary Statistics:")
        st.dataframe(df.describe())
        
    elif menu == "EDA":
        show_eda(df)
        
    elif menu == "Rating Prediction":
        predict_ratings(df)

if __name__ == "__main__":
    main()
