import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Create and fit label encoders for categorical variables
education_encoder = LabelEncoder()
marital_encoder = LabelEncoder()

# Fit the encoders with the correct categories
education_encoder.fit(['2n Cycle', 'Basic', 'Graduation', 'Master', 'PhD'])
marital_encoder.fit(['Alone', 'Divorced', 'Married', 'Single', 'Together', 'Widow', 'YOLO'])

# Create a dictionary of encoders
label_encoders = {
    'Education': education_encoder,
    'Marital_Status': marital_encoder
}

# Create and fit the scaler
scaler = StandardScaler()

# Define the numeric columns that need scaling
numeric_cols = ['Income', 'Kids', 'Expenses', 'TotalAcceptedCmp', 
                'NumTotalPurchases', 'Age', 'day_engaged']

# Create sample data with typical ranges for each numeric feature
# These ranges should match your training data
sample_data = pd.DataFrame({
    'Income': [0, 200000],  # Range from 0 to 200,000
    'Kids': [0, 10],        # Range from 0 to 10
    'Expenses': [0, 100000], # Range from 0 to 100,000
    'TotalAcceptedCmp': [0, 10], # Range from 0 to 10
    'NumTotalPurchases': [0, 100], # Range from 0 to 100
    'Age': [18, 100],       # Range from 18 to 100
    'day_engaged': [0, 3650] # Range from 0 to 3650 (10 years)
})

# Fit the scaler with the sample data
scaler.fit(sample_data[numeric_cols])

# Save both the encoders and scaler
joblib.dump(label_encoders, 'label_encoders.joblib')
joblib.dump(scaler, 'scaler.joblib')

print("Preprocessing objects saved successfully!")
print("\nEducation categories:", education_encoder.classes_)
print("Marital Status categories:", marital_encoder.classes_)
print("\nScaler parameters:")
print("Mean:", scaler.mean_)
print("Scale:", scaler.scale_) 