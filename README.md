# Credit Risk Score Analysis

A comprehensive web application that analyzes credit risk based on financial behavior patterns and provides a detailed credit score assessment with actionable insights.

## Overview

This application simulates a credit risk assessment system similar to those used by financial institutions. Users can input their financial data, including payment history, credit utilization, and other relevant factors. The system analyzes this information using a machine learning model to generate a credit risk assessment, FICO score equivalent, and loan approval probabilities for different types of loans.

The project aims to demonstrate how machine learning can be applied to financial risk analysis and to educate users about the factors that influence their credit scores.

## Features

- **Credit Score Prediction**: Calculates an estimated FICO score (300-850) based on user inputs
- **Risk Assessment**: Determines the probability of credit default
- **Interactive UI**: User-friendly interface with informative tooltips and guidance
- **Detailed Explanations**: Provides clear explanations for each credit score component
- **Loan Approval Simulator**: Estimates approval chances and interest rates for different loan types
- **Credit Improvement Tips**: Personalized recommendations for improving credit score
- **Visual Representations**: Score gauge, color-coded risk levels, and comparison charts

## Technical Architecture

The application follows a client-server architecture:

- **Backend**: FastAPI application that handles data processing, model inference, and serves API endpoints
- **Frontend**: HTML/CSS/JavaScript that provides an interactive user interface
- **Machine Learning Model**: Random Forest classifier trained on credit card default data
- **Data Processing**: Feature engineering pipeline for raw input data

## Installation

### Prerequisites
- Python 3.7+
- pip or conda package manager

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/yourusername/credit-risk-score-analysis.git
cd credit-risk-score-analysis
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Download the dataset:
   - Download the Credit Card Default dataset from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)
   - Rename the downloaded file to `credit_data.csv` and place it in the `data/` directory
   - Alternatively, you can use any credit default dataset with similar features, just ensure the column names match those used in the application

5. Start the application:
```bash
python main.py
```

5. Open your browser and navigate to:
```
http://localhost:8000
```

## Usage

1. **Enter Personal Information**: Provide basic information like credit limit, gender, education level, etc.

2. **Input Payment History**: Select your payment status for each of the last 6 months.

3. **Enter Bill Amounts**: Input your monthly bill amounts for the last 6 months.

4. **Enter Payment Amounts**: Input how much you've paid toward each bill.

5. **Submit for Analysis**: Click the "Get My Credit Score Estimate" button to submit your data.

6. **Review Results**: Analyze your credit risk assessment, including:
   - FICO score equivalent and risk category
   - Detailed breakdown of credit score components
   - Loan approval probabilities and likely interest rates
   - Personalized tips to improve your score

## Machine Learning Model

The credit risk prediction model is a Random Forest classifier trained on a dataset containing credit card default information. The model has the following characteristics:

- **Algorithm**: Random Forest (ensemble of decision trees)
- **Features**: 23 input features + 5 engineered features
- **Target Variable**: Binary classification (default/no default)
- **Performance Metrics**:
  - Accuracy: 81.5%
  - Precision: 78.3%
  - Recall: 72.1%
  - F1 Score: 75.1%
  - ROC AUC: 0.83

### Dataset

The model is trained using the "Default of Credit Card Clients" dataset from the UCI Machine Learning Repository. This dataset contains information on default payments, demographic factors, credit data, history of payment, and bill statements of credit card clients in Taiwan from April 2005 to September 2005.

- **Source**: UCI Machine Learning Repository
- **URL**: [Default of Credit Card Clients Dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)
- **Features**: 23 attributes including payment history, bill amounts, and payment amounts
- **Size**: ~30,000 instances
- **Format**: CSV file (not included in the repository due to size constraints)

To run this application, you need to download the dataset from the link above and place it in the `data/` directory as `credit_data.csv`. The training script will automatically preprocess the data and train the model.

### Feature Engineering

The system creates the following additional features to improve prediction accuracy:

1. **Debt Ratio**: Average bill amount / credit limit
2. **Payment Ratios**: Payment amount / bill amount for each month
3. **Average Payment Delay**: Mean of payment delay values across all months
4. **Previous Delay Indicator**: Binary indicator of any previous payment delays
5. **Delay Month Count**: Total number of months with payment delays

## Credit Score Components

The application explains the five key factors that make up a credit score:

1. **Payment History (35%)**: Record of on-time payments
2. **Credit Utilization (30%)**: Percentage of available credit being used
3. **Length of Credit History (15%)**: How long accounts have been open
4. **Credit Mix (10%)**: Variety of credit account types
5. **New Credit (10%)**: Recently opened accounts and credit inquiries

Each factor includes detailed explanations and personalized tips for improvement.

## API Documentation

The application exposes several REST API endpoints:

### Prediction Endpoint
- **URL**: `/predict`
- **Method**: POST
- **Description**: Analyze credit risk based on user data
- **Request Body**: JSON with credit data fields
```json
{
  "LIMIT_BAL": 200000.0,
  "SEX": 2,
  "EDUCATION": 2,
  "MARRIAGE": 1,
  "AGE": 35,
  "PAY_0": 0,
  "PAY_2": 0,
  "PAY_3": 0,
  "PAY_4": 0,
  "PAY_5": 0,
  "PAY_6": 0,
  "BILL_AMT1": 30000.0,
  "BILL_AMT2": 28000.0,
  "BILL_AMT3": 25000.0,
  "BILL_AMT4": 23000.0,
  "BILL_AMT5": 21000.0,
  "BILL_AMT6": 20000.0,
  "PAY_AMT1": 5000.0,
  "PAY_AMT2": 5000.0,
  "PAY_AMT3": 5000.0,
  "PAY_AMT4": 5000.0,
  "PAY_AMT5": 5000.0,
  "PAY_AMT6": 5000.0
}
```

- **Response**: Comprehensive JSON with risk assessment results
```json
{
  "risk_label": "Good",
  "risk_probability": 0.12,
  "fico_score": 720,
  "risk_details": {
    "default_probability": 0.12,
    "non_default_probability": 0.88,
    "risk_level": "Good",
    "us_fico_score": 720,
    "top_factors": {
      "message": "No specific risk factors identified. Good credit profile!"
    }
  },
  "loan_approval": {
    "mortgage": {
      "approval_probability": 85.5,
      "likely_rate": "Below Average",
      "approved": true
    },
    "auto_loan": {
      "approval_probability": 92.3,
      "likely_rate": "Average",
      "approved": true
    },
    "credit_card": {
      "approval_probability": 95.7,
      "likely_rate": "Below Average",
      "approved": true
    },
    "personal_loan": {
      "approval_probability": 88.0,
      "likely_rate": "Average",
      "approved": true
    }
  }
}
```

### Documentation Endpoints
- **URL**: `/field-info`
- **Method**: GET
- **Description**: Get field descriptions and metadata

- **URL**: `/credit-factor-explanations`
- **Method**: GET
- **Description**: Get detailed explanations of credit score factors

### Health Endpoint
- **URL**: `/health`
- **Method**: GET
- **Description**: Check API health status
```json
{
  "status": "ok"
}
```

For detailed API documentation, visit http://localhost:8000/docs when the application is running.

## Technologies Used

### Backend
- **FastAPI**: Web framework for API development
- **Pydantic**: Data validation and settings management
- **Joblib**: Model serialization
- **Pandas/NumPy**: Data manipulation
- **Scikit-learn**: Machine learning tools

### Frontend
- **HTML5/CSS3**: Structure and styling
- **JavaScript**: Dynamic content and interactivity
- **Fetch API**: Asynchronous API calls

## Future Improvements

- **More Advanced Model**: Implement a more sophisticated model with higher accuracy
- **User Accounts**: Allow users to save their profiles and track changes over time
- **Scenario Analysis**: Enable "what-if" analysis for different financial decisions
- **Data Visualization**: Add more charts and graphs for better data interpretation
- **Mobile App**: Develop a mobile application version
- **Integration**: Connect with real credit bureaus for actual credit data

---

*Note: This application is for educational purposes only and should not be used for actual credit decisions. The predictions and assessments are based on simulated data and a simplified model of credit risk evaluation.*
