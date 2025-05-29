import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import joblib
from typing import Dict, Tuple, Any

# Set page configuration
st.set_page_config(
    page_title="Customer Segmentation Analysis",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance and animations
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding: 2rem;
        animation: fadeIn 0.5s ease-in;
    }
    
    /* Button styling with hover animation */
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 10px;
        border: none;
        border-radius: 4px;
        font-weight: bold;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transform: translateY(-2px);
    }
    
    .stButton>button:active {
        transform: translateY(0);
    }
    
    /* Input field styling with focus animation */
    .stSelectbox, .stNumberInput {
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .stSelectbox:hover, .stNumberInput:hover {
        transform: translateY(-2px);
    }
    
    /* Card-like containers */
    .segment-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        background-color: #f0f2f6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        animation: slideIn 0.5s ease-out;
    }
    
    .segment-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Animated headers */
    h1, h2, h3 {
        animation: slideIn 0.5s ease-out;
    }
    
    /* Success message animation */
    .stSuccess {
        animation: slideIn 0.5s ease-out;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        animation: fadeIn 0.5s ease-in;
    }
    
    /* Keyframe animations */
    @keyframes fadeIn {
        from {
            opacity: 0;
        }
        to {
            opacity: 1;
        }
    }
    
    @keyframes slideIn {
        from {
            transform: translateY(20px);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }
    
    /* Custom container for prediction results */
    .prediction-container {
        padding: 2rem;
        border-radius: 10px;
        background: linear-gradient(145deg, #ffffff, #f0f2f6);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
        animation: slideIn 0.5s ease-out;
    }
    
    /* Animated list items */
    .prediction-container ul li {
        animation: fadeIn 0.5s ease-in;
        animation-fill-mode: both;
    }
    
    .prediction-container ul li:nth-child(1) { animation-delay: 0.1s; }
    .prediction-container ul li:nth-child(2) { animation-delay: 0.2s; }
    .prediction-container ul li:nth-child(3) { animation-delay: 0.3s; }
    .prediction-container ul li:nth-child(4) { animation-delay: 0.4s; }
    
    /* Custom styling for metrics */
    .metric-container {
        padding: 1rem;
        border-radius: 8px;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    
    .metric-container:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)

def load_model() -> Tuple[Any, Dict[str, Any], Any]:
    """
    Load the model and preprocessing objects.
    
    Returns:
        Tuple containing the model, label encoders, and scaler
    """
    try:
        model_path = 'km.pkl'
        label_encoders_path = 'label_encoders.joblib'
        scaler_path = 'scaler.joblib'
        
        # Check if files exist
        for path in [model_path, label_encoders_path, scaler_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required file not found: {path}")
        
        # Load files using joblib
        model = joblib.load(model_path)
        label_encoders = joblib.load(label_encoders_path)
        scaler = joblib.load(scaler_path)
        
        return model, label_encoders, scaler
    
    except FileNotFoundError as e:
        st.error(f"Error: {str(e)}")
        st.info("Please ensure all model files are in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the model: {str(e)}")
        st.info("This might be due to incompatible model files. Please ensure the model files were saved using the same Python version.")
        st.stop()

def preprocess_input(input_dict: Dict[str, Any], label_encoders: Dict[str, Any], scaler: Any) -> pd.DataFrame:
    """
    Preprocess the input data for prediction.
    
    Args:
        input_dict: Dictionary containing input features
        label_encoders: Dictionary of label encoders
        scaler: Scaler object
    
    Returns:
        Preprocessed DataFrame
    """
    try:
        # Create DataFrame
        df = pd.DataFrame([input_dict])
        
        # Transform categorical features
        df['Education'] = label_encoders['Education'].transform(df['Education'])
        df['Marital_Status'] = label_encoders['Marital_Status'].transform(df['Marital_Status'])
        
        # Scale numeric features
        numeric_cols = ['Income', 'Kids', 'Expenses', 'TotalAcceptedCmp', 
                       'NumTotalPurchases', 'Age', 'day_engaged']
        df[numeric_cols] = scaler.transform(df[numeric_cols])
        
        return df
    
    except KeyError as e:
        st.error(f"Missing label encoder for: {str(e)}")
        st.info("Please ensure the label encoders are properly saved and loaded.")
        st.stop()
    except Exception as e:
        st.error(f"Error in preprocessing: {str(e)}")
        st.info("Please ensure the input data matches the expected format and values.")
        st.stop()

def get_segment_info(segment: int) -> Dict[str, Any]:
    """
    Get detailed information about a customer segment.
    
    Args:
        segment: Segment number
    
    Returns:
        Dictionary containing segment information
    """
    segment_info = {
        0: {
            "name": "Low-Value Customers",
            "description": "Customers with lower engagement and spending patterns",
            "characteristics": [
                "Lower income and spending capacity",
                "Less responsive to marketing campaigns",
                "Fewer purchases overall",
                "May be new or occasional customers"
            ],
            "recommendations": [
                "Focus on increasing engagement through personalized offers",
                "Implement loyalty programs to encourage more purchases",
                "Target with value-based promotions",
                "Consider introductory offers to build relationship"
            ],
            "color": "🔵"
        },
        1: {
            "name": "High-Value Customers",
            "description": "Premium customers with high engagement and spending",
            "characteristics": [
                "Higher income and spending capacity",
                "Highly responsive to marketing campaigns",
                "Frequent and regular purchases",
                "Strong brand loyalty"
            ],
            "recommendations": [
                "Maintain high-quality service and premium offerings",
                "Implement VIP programs or exclusive benefits",
                "Focus on retention and relationship building",
                "Offer early access to new products or services"
            ],
            "color": "🟢"
        },
        2: {
            "name": "Mid-Value Customers",
            "description": "Customers with moderate engagement and spending patterns",
            "characteristics": [
                "Moderate income and spending",
                "Some campaign engagement",
                "Regular but not frequent purchases",
                "Potential for growth"
            ],
            "recommendations": [
                "Balance between value and premium offerings",
                "Regular engagement through mixed marketing strategies",
                "Focus on gradual growth and increased engagement",
                "Consider cross-selling opportunities"
            ],
            "color": "🟡"
        }
    }
    return segment_info.get(segment, {
        "name": "Unknown Segment",
        "description": "Segment information not available",
        "characteristics": [],
        "recommendations": [],
        "color": "⚪"
    })

def display_prediction(segment: int):
    """
    Display the prediction results with detailed information.
    
    Args:
        segment: Predicted segment number
    """
    # Get segment information
    info = get_segment_info(segment)
    
    # Create a container for the prediction results with animation
    st.markdown('<div class="prediction-container">', unsafe_allow_html=True)
    
    # Display segment header with animation
    st.markdown(f"""
    ### {info['color']} Predicted Customer Segment: {info['name']}
    """)
    
    # Display description
    st.markdown(f"""
    **Description:** {info['description']}
    """)
    
    # Create two columns for characteristics and recommendations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.markdown("#### 📊 Customer Characteristics")
        for char in info['characteristics']:
            st.markdown(f"- {char}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.markdown("#### 💡 Recommendations")
        for rec in info['recommendations']:
            st.markdown(f"- {rec}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Add a divider
    st.markdown("---")
    
    # Add additional insights with animation
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.markdown("""
    #### 📈 Next Steps
    - Monitor customer behavior and adjust strategies accordingly
    - Track the effectiveness of recommended actions
    - Consider A/B testing different approaches
    - Regularly review and update customer segmentation
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    # Title and description with animation
    st.markdown('<div class="segment-box">', unsafe_allow_html=True)
    st.title("Customer Segmentation Analysis")
    st.markdown("""
    This application predicts customer segments based on their characteristics.
    Please input the customer information below to get a prediction.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Load model and preprocessing objects
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
            options=['2n Cycle', 'Basic', 'Graduation', 'Master', 'PhD'],
            help="Select the customer's highest education level"
        )
        
        marital_status = st.selectbox(
            "Marital Status",
            options=['Alone', 'Divorced', 'Married', 'Single', 'Together', 'Widow', 'YOLO'],
            help="Select the customer's marital status"
        )
        
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30,
            help="Enter the customer's age"
        )
        
        kids = st.number_input(
            "Number of Kids",
            min_value=0,
            max_value=10,
            value=0,
            help="Enter the number of children"
        )
    
    with col2:
        st.subheader("Financial Information")
        income = st.number_input(
            "Income",
            min_value=0,
            max_value=200000,
            value=50000,
            step=1000,
            help="Enter annual income"
        )
        
        expenses = st.number_input(
            "Expenses",
            min_value=0,
            max_value=100000,
            value=1000,
            step=100,
            help="Enter annual expenses"
        )
        
        total_accepted_cmp = st.number_input(
            "Total Accepted Campaigns",
            min_value=0,
            max_value=10,
            value=0,
            help="Number of marketing campaigns accepted"
        )
        
        num_total_purchases = st.number_input(
            "Total Number of Purchases",
            min_value=0,
            max_value=100,
            value=0,
            help="Total number of purchases made"
        )
        
        days_engaged = st.number_input(
            "Days Engaged",
            min_value=0,
            max_value=3650,
            value=365,
            help="Number of days the customer has been engaged"
        )
    
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
    
    # Add predict button
    if st.button("Predict Customer Segment", help="Click to get the prediction"):
        try:
            # Preprocess the input
            processed_input = preprocess_input(input_dict, label_encoders, scaler)
            
            # Make prediction
            prediction = model.predict(processed_input)
            segment = prediction[0]
            
            # Display the prediction with detailed information
            display_prediction(segment)
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
    
    # Add additional information in the sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This application uses machine learning to segment customers based on their characteristics.
        
        ### Features Used
        - Education level
        - Marital status
        - Income
        - Number of kids
        - Expenses
        - Campaign acceptance
        - Purchase history
        - Age
        - Customer engagement duration
        
        ### Model Information
        The model uses KMeans clustering to identify distinct customer segments.
        """)

if __name__ == "__main__":
    main() 