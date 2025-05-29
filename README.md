# Customer Segmentation Analysis

A Streamlit web application that predicts customer segments based on various customer characteristics using machine learning.

## Overview

This application uses a clustering model to segment customers into different groups based on their characteristics such as education level, marital status, income, spending patterns, and engagement metrics. The segmentation helps businesses understand their customer base better and make data-driven decisions.

## Features

- Interactive web interface built with Streamlit
- Real-time customer segment prediction
- Input validation and error handling
- Detailed segment explanations
- Responsive design with modern UI

## Customer Segments

The model classifies customers into three main segments:

1. **Segment 0**: Low-value customers
   - Lower income and spending
   - Less engaged with campaigns
   - Fewer purchases

2. **Segment 1**: High-value customers
   - Higher income and spending
   - More engaged with campaigns
   - Frequent purchases

3. **Segment 2**: Mid-value customers
   - Moderate income and spending
   - Some campaign engagement
   - Regular purchases

## Input Features

The model uses the following features for segmentation:

- Education Level (2n Cycle, Basic, Graduation, Master, PhD)
- Marital Status (Alone, Divorced, Married, Single, Together, Widow, YOLO)
- Income
- Number of Kids
- Expenses
- Total Accepted Campaigns
- Total Number of Purchases
- Age
- Days Engaged

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd customer-segmentation
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Ensure you have the following model files in your project directory:
   - `km.pkl` (KMeans model)
   - `lb.pkl` (Label encoders)
   - `sc.pkl` (Scaler)

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

4. Input the customer information in the web interface and click "Predict Customer Segment" to get the prediction.

## Requirements

- Python 3.7+
- Streamlit
- Pandas
- NumPy
- scikit-learn
- pickle5 (for Python < 3.8)

## Project Structure

```
customer-segmentation/
├── app.py              # Main Streamlit application
├── km.pkl             # KMeans model file
├── lb.pkl             # Label encoders file
├── sc.pkl             # Scaler file
├── requirements.txt   # Project dependencies
└── README.md         # Project documentation
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Streamlit for the web framework
- scikit-learn for the machine learning algorithms
- All contributors who have helped shape this project 