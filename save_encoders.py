import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

# Create label encoders for categorical variables
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

# Save the encoders
joblib.dump(label_encoders, 'label_encoders.joblib')

print("Encoders saved successfully!")
print("\nEducation categories:", education_encoder.classes_)
print("Marital Status categories:", marital_encoder.classes_) 