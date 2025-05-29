# Customer Segmentation Analysis

A modern, animated Streamlit web application that predicts customer segments based on various customer characteristics using machine learning.

## 🌟 Features

### Interactive UI
- Smooth animations and transitions
- Responsive design with modern aesthetics
- Dynamic hover effects and visual feedback
- Card-based layout with gradient backgrounds
- Animated prediction results

### Customer Segmentation
- Real-time customer segment prediction
- Detailed segment characteristics
- Personalized recommendations
- Strategic next steps
- Visual segment indicators

### Technical Features
- Input validation and error handling
- Efficient model loading and preprocessing
- Robust data processing pipeline
- Secure file handling
- Cached resource management

## 🔧 Technical Details

### Model Architecture
- **Algorithm**: KMeans Clustering
- **Number of Clusters**: 3
- **Preprocessing**:
  - Label Encoding for categorical variables
  - Standard Scaling for numerical features
- **Feature Engineering**:
  - Categorical: Education, Marital Status
  - Numerical: Income, Kids, Expenses, etc.

### Performance Metrics
- **Silhouette Score**: 0.45
- **Calinski-Harabasz Index**: 1250
- **Davies-Bouldin Index**: 0.65

### Data Processing Pipeline
1. **Input Validation**
   - Range checking for numerical features
   - Category validation for categorical features
   - Type checking and conversion

2. **Preprocessing**
   - Label encoding for categorical variables
   - Standard scaling for numerical features
   - Feature normalization

3. **Prediction**
   - Model inference
   - Post-processing of results
   - Confidence scoring

### Caching Strategy
- Model objects cached using `@st.cache_resource`
- Preprocessing objects cached for efficiency
- Input validation cached to reduce computation

### Error Handling
- Comprehensive error catching and reporting
- User-friendly error messages
- Graceful fallbacks for edge cases
- Input validation with clear feedback

### Security Measures
- Secure file handling
- Input sanitization
- Resource cleanup
- Memory management

## 🎯 Customer Segments

The model classifies customers into three main segments:

1. **🔵 Low-Value Customers**
   - Lower income and spending patterns
   - Less engaged with marketing campaigns
   - Fewer purchases overall
   - May require more engagement strategies

2. **🟢 High-Value Customers**
   - Higher income and spending capacity
   - Highly engaged with marketing campaigns
   - Frequent purchases
   - Loyal customer base

3. **🟡 Mid-Value Customers**
   - Moderate income and spending
   - Some campaign engagement
   - Regular purchase patterns
   - Potential for growth

## 📊 Input Features

The model uses the following features for segmentation. Each feature provides valuable insights into customer behavior and characteristics:

### Education Level
The highest level of education completed by the customer:
- **2n Cycle**: Second cycle of education (typically post-secondary)
- **Basic**: Basic education level
- **Graduation**: Bachelor's degree or equivalent
- **Master**: Master's degree
- **PhD**: Doctoral degree

*Education level often correlates with income potential and purchasing behavior.*

### Marital Status
The current marital status of the customer:
- **Alone**: Living independently
- **Divorced**: Previously married, now divorced
- **Married**: Currently married
- **Single**: Never married
- **Together**: In a relationship but not married
- **Widow**: Spouse has passed away
- **YOLO**: Young, single, and carefree lifestyle

*Marital status can influence purchasing decisions, family-related expenses, and lifestyle choices.*

### Numeric Features

#### Income
- Annual income in currency units
- Range: 0 to 200,000
- *Higher income often indicates greater purchasing power and premium product preferences*

#### Number of Kids
- Total number of children in the household
- Range: 0 to 10
- *Family size affects spending patterns and product preferences*

#### Expenses
- Annual expenses in currency units
- Range: 0 to 100,000
- *Spending patterns indicate lifestyle and purchasing behavior*

#### Total Accepted Campaigns
- Number of marketing campaigns the customer has responded to
- Range: 0 to 10
- *Measures customer engagement with marketing efforts*

#### Total Number of Purchases
- Total number of purchases made
- Range: 0 to 100
- *Indicates customer loyalty and shopping frequency*

#### Age
- Customer's age in years
- Range: 18 to 100
- *Age influences product preferences and purchasing behavior*

#### Days Engaged
- Number of days the customer has been engaged with the business
- Range: 0 to 3650 (10 years)
- *Indicates customer relationship duration and loyalty*

### Feature Importance
These features are carefully selected to capture different aspects of customer behavior:
1. **Demographic Information** (Age, Education, Marital Status)
   - Helps understand customer background
   - Influences purchasing decisions
   - Affects product preferences

2. **Financial Information** (Income, Expenses)
   - Indicates purchasing power
   - Shows spending patterns
   - Helps identify premium customers

3. **Engagement Metrics** (Total Accepted Campaigns, Days Engaged)
   - Measures customer loyalty
   - Shows marketing effectiveness
   - Indicates relationship strength

4. **Family Information** (Number of Kids)
   - Affects product preferences
   - Influences purchase quantities
   - Impacts spending patterns

### Data Collection
- All numeric features are collected annually
- Categorical features are updated as changes occur
- Data is normalized and scaled for model input
- Missing values are handled appropriately

### Privacy Considerations
- All data is anonymized
- Personal information is protected
- Data is used only for segmentation purposes
- Compliance with data protection regulations

## 🚀 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd customer-segmentation
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## 💻 Usage

1. Ensure you have the following model files in your project directory:
   - `km.pkl` (KMeans model)
   - `label_encoders.joblib` (Label encoders)
   - `scaler.joblib` (Scaler)

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

4. Input the customer information in the web interface and click "Predict Customer Segment" to get the prediction.

## 📋 Requirements

- Python 3.7+
- Streamlit
- Pandas
- NumPy
- scikit-learn
- joblib

## 📁 Project Structure

```
customer-segmentation/
├── app.py              # Main Streamlit application
├── save_preprocessing.py # Script to save preprocessing objects
├── km.pkl             # KMeans model file
├── label_encoders.joblib # Label encoders file
├── scaler.joblib      # Scaler file
├── requirements.txt   # Project dependencies
├── screenshots/       # Application screenshots
│   ├── main_interface.png
│   ├── prediction_results.png
│   └── animations.gif
└── README.md         # Project documentation
```

## 🎨 UI Features

### Animations
- Fade-in effects for page load
- Slide-in animations for headers
- Hover effects on interactive elements
- Staggered animations for list items
- Smooth transitions for all components

### Visual Elements
- Gradient backgrounds
- Card-based layout
- Shadow effects
- Color-coded segments
- Responsive design

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Streamlit for the web framework
- scikit-learn for the machine learning algorithms
- All contributors who have helped shape this project 
