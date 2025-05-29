import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import joblib

# Set page configuration
st.set_page_config(
    page_title="Customer Segmentation Analysis",
    page_icon="👥",
    layout="wide"
)

# Custom CSS for better appearance
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 10px;
        border: none;
        border-radius: 4px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("Customer Segmentation Analysis")
st.markdown("""
This application predicts customer segments based on their characteristics.
Please input the customer information below to get a prediction.
""")

# Load model and preprocessing objects
@st.cache_resource
def load_model():
    try:
        # Load all files with pickle
        with open('km.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('lb.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        with open('sc.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        return model, label_encoders, scaler
    except FileNotFoundError as e:
        st.error(f"Error: Model files not found. Please make sure km.pkl, lb.pkl, and sc.pkl are in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the model: {str(e)}")
        st.stop()

try:
    model, label_encoders, scaler = load_model()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Create two columns for input fields
col1, col2 = st.columns(2)

with col1:
    st.subheader("Personal Information")
    education = st.selectbox(
        "Education Level",
        options=['2n Cycle', 'Basic', 'Graduation', 'Master', 'PhD']
    )
    
    marital_status = st.selectbox(
        "Marital Status",
        options=['Alone', 'Divorced', 'Married', 'Single', 'Together', 'Widow', 'YOLO']
    )
    
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    
    kids = st.number_input("Number of Kids", min_value=0, max_value=10, value=0)

with col2:
    st.subheader("Financial Information")
    income = st.number_input("Income", min_value=0, max_value=200000, value=50000, step=1000)
    
    expenses = st.number_input("Expenses", min_value=0, max_value=100000, value=1000, step=100)
    
    total_accepted_cmp = st.number_input("Total Accepted Campaigns", min_value=0, max_value=10, value=0)
    
    num_total_purchases = st.number_input("Total Number of Purchases", min_value=0, max_value=100, value=0)
    
    days_engaged = st.number_input("Days Engaged", min_value=0, max_value=3650, value=365)

# Create input dictionary
input_dict = {
    'Education': education,
    'Marital_Status': marital_status,
    'Income': income,
    'Kids': kids,
    'Expenses': expenses,
    'TotalAcceptedCmp': total_accepted_cmp,
    'NumTotalPurchases': num_total_purchases,
    'Age': age,
    'day_engaged': days_engaged
}

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

def preprocess(df):
    try:
        # Create a copy of the dataframe
        df_processed = df.copy()
        
        # Transform categorical features using their respective encoders
        df_processed['Education'] = label_encoders['Education'].transform(df_processed['Education'])
        df_processed['Marital_Status'] = label_encoders['Marital_Status'].transform(df_processed['Marital_Status'])
        
        # Scale numeric columns
        numeric_cols = ['Income', 'Kids', 'Expenses', 'TotalAcceptedCmp', 
                       'NumTotalPurchases', 'Age', 'day_engaged']
        df_processed[numeric_cols] = scaler.transform(df_processed[numeric_cols])
        
        return df_processed
    except Exception as e:
        st.error(f"Error in preprocessing: {str(e)}")
        st.stop()

# Add predict button
if st.button("Predict Customer Segment"):
    try:
        # Preprocess the input
        processed_input = preprocess(input_df)
        
        # Make prediction
        prediction = model.predict(processed_input)
        
        # Display results
        st.success(f"Predicted Customer Segment: {prediction[0]}")
        
        # Add segment explanation
        st.markdown("""
        ### Customer Segments Explanation:
        - **Segment 0**: Low-value customers
            - Lower income and spending
            - Less engaged with campaigns
            - Fewer purchases
        
        - **Segment 1**: High-value customers
            - Higher income and spending
            - More engaged with campaigns
            - Frequent purchases
        
        - **Segment 2**: Mid-value customers
            - Moderate income and spending
            - Some campaign engagement
            - Regular purchases
        """)
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

# Add additional information
st.markdown("""
---
### About the Model
This model uses customer data to predict their segment based on various characteristics including:
- Education level
- Marital status
- Income
- Number of kids
- Expenses
- Campaign acceptance
- Purchase history
- Age
- Customer engagement duration
""") 