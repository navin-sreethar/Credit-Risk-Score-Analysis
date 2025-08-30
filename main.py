import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Union, Optional
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Credit Risk Prediction API",
    description="API for predicting credit risk using a trained Random Forest model",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Define paths
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "credit_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# Load model and scaler on startup
@app.on_event("startup")
async def startup_event():
    global model, scaler
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print(f"Model and scaler loaded successfully from {MODEL_DIR}")
    except Exception as e:
        print(f"Error loading model or scaler: {e}")
        raise RuntimeError(f"Could not load model or scaler: {e}")

# Input validation model
class CreditData(BaseModel):
    # Original features
    LIMIT_BAL: float = Field(..., description="Credit limit")
    SEX: int = Field(..., description="Gender (1=male, 2=female)")
    EDUCATION: int = Field(..., description="Education level (1=graduate, 2=university, 3=high school, 4=others)")
    MARRIAGE: int = Field(..., description="Marital status (1=married, 2=single, 3=others)")
    AGE: int = Field(..., description="Age in years")
    PAY_0: int = Field(..., description="Payment status in September (-2 to 9)")
    PAY_2: int = Field(..., description="Payment status in August (-2 to 9)")
    PAY_3: int = Field(..., description="Payment status in July (-2 to 9)")
    PAY_4: int = Field(..., description="Payment status in June (-2 to 9)")
    PAY_5: int = Field(..., description="Payment status in May (-2 to 9)")
    PAY_6: int = Field(..., description="Payment status in April (-2 to 9)")
    BILL_AMT1: float = Field(..., description="Bill amount in September")
    BILL_AMT2: float = Field(..., description="Bill amount in August")
    BILL_AMT3: float = Field(..., description="Bill amount in July")
    BILL_AMT4: float = Field(..., description="Bill amount in June")
    BILL_AMT5: float = Field(..., description="Bill amount in May")
    BILL_AMT6: float = Field(..., description="Bill amount in April")
    PAY_AMT1: float = Field(..., description="Payment amount in September")
    PAY_AMT2: float = Field(..., description="Payment amount in August")
    PAY_AMT3: float = Field(..., description="Payment amount in July")
    PAY_AMT4: float = Field(..., description="Payment amount in June")
    PAY_AMT5: float = Field(..., description="Payment amount in May")
    PAY_AMT6: float = Field(..., description="Payment amount in April")
    
    # Validators
    @validator('SEX')
    def sex_must_be_valid(cls, v):
        if v not in [1, 2]:
            raise ValueError('SEX must be either 1 (male) or 2 (female)')
        return v
    
    @validator('EDUCATION')
    def education_must_be_valid(cls, v):
        if v not in [1, 2, 3, 4]:
            raise ValueError('EDUCATION must be between 1 and 4')
        return v
    
    @validator('MARRIAGE')
    def marriage_must_be_valid(cls, v):
        if v not in [1, 2, 3]:
            raise ValueError('MARRIAGE must be between 1 and 3')
        return v
    
    @validator('AGE')
    def age_must_be_valid(cls, v):
        if v < 18 or v > 100:
            raise ValueError('AGE must be between 18 and 100')
        return v
    
    @validator('PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6')
    def pay_must_be_valid(cls, v):
        if v < -2 or v > 9:
            raise ValueError('Payment status must be between -2 and 9')
        return v
        
    class Config:
        schema_extra = {
            "example": {
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
        }

# Response model
class PredictionResponse(BaseModel):
    risk_label: str
    risk_probability: float
    fico_score: int
    risk_details: Dict[str, Union[str, float, Dict[str, Union[str, float, int]]]]
    loan_approval: Dict[str, Dict[str, Union[float, str]]]
    
# Credit score factors explanation
class CreditFactorExplanation(BaseModel):
    name: str
    description: str
    impact_weight: float
    tips: List[str]

# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok"}

# Root endpoint - serve the HTML form
@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def root():
    return RedirectResponse(url="/static/index.html")

# Credit score factors explanation endpoint
@app.get("/credit-factor-explanations", tags=["Documentation"])
async def get_credit_factor_explanations():
    """Return detailed explanations about credit score factors
    
    This endpoint provides information about the five main components of a credit score:
    - Payment History (35%)
    - Credit Utilization (30%)
    - Length of Credit History (15%)
    - Credit Mix (10%)
    - New Credit (10%)
    
    Each factor includes a description, impact weight, and tips for improvement.
    """
    return {
        "payment_history": {
            "name": "Payment History",
            "description": "Your track record of paying bills on time. This is the most significant factor in your credit score.",
            "impact_weight": 0.35,
            "tips": [
                "Pay all your bills on time",
                "If you miss a payment, get current as soon as possible",
                "Set up automatic payments or payment reminders"
            ]
        },
        "credit_utilization": {
            "name": "Credit Utilization",
            "description": "The percentage of your available credit that you're using. Lower utilization is better.",
            "impact_weight": 0.30,
            "tips": [
                "Keep your credit card balances low",
                "Pay off debt rather than moving it around",
                "Keep unused credit cards open unless there's a good reason to close them"
            ]
        },
        "credit_history_length": {
            "name": "Length of Credit History",
            "description": "How long you've been using credit. Longer history typically results in higher scores.",
            "impact_weight": 0.15,
            "tips": [
                "Don't close older accounts if possible",
                "Be patient - building a long credit history takes time"
            ]
        },
        "credit_mix": {
            "name": "Credit Mix",
            "description": "The variety of credit accounts you have (credit cards, loans, mortgages, etc.).",
            "impact_weight": 0.10,
            "tips": [
                "Have a mix of credit types, but only apply for credit you actually need",
                "Don't open multiple accounts in a short period"
            ]
        },
        "new_credit": {
            "name": "New Credit",
            "description": "Recently opened accounts and credit inquiries.",
            "impact_weight": 0.10,
            "tips": [
                "Research and rate-shop within a focused period",
                "Only apply for new credit accounts when necessary",
                "Be careful about generating many credit inquiries"
            ]
        }
    }

# Field descriptions endpoint
@app.get("/field-info", tags=["Documentation"])
async def get_field_info():
    """Provides human-readable descriptions for all input fields"""
    field_info = {
        "LIMIT_BAL": {
            "name": "Credit Limit ($)",
            "description": "Your total available credit limit",
            "type": "number",
            "help_text": "Higher limits generally indicate better creditworthiness"
        },
        "SEX": {
            "name": "Gender",
            "description": "Your gender",
            "type": "select",
            "options": [
                {"value": 1, "label": "Male"},
                {"value": 2, "label": "Female"}
            ]
        },
        "EDUCATION": {
            "name": "Education Level",
            "description": "Your highest level of education completed",
            "type": "select",
            "options": [
                {"value": 1, "label": "Graduate School"},
                {"value": 2, "label": "University"},
                {"value": 3, "label": "High School"},
                {"value": 4, "label": "Other"}
            ]
        },
        "MARRIAGE": {
            "name": "Marital Status",
            "description": "Your current marital status",
            "type": "select",
            "options": [
                {"value": 1, "label": "Married"},
                {"value": 2, "label": "Single"},
                {"value": 3, "label": "Other"}
            ]
        },
        "AGE": {
            "name": "Age",
            "description": "Your current age in years",
            "type": "number",
            "min": 18
        },
        "payment_status": {
            "name": "Payment Status Values",
            "description": "What each payment status value means",
            "help_text": "These values indicate your payment behavior for each month",
            "values": {
                "-2": "No consumption (no bill this month)",
                "-1": "Paid in full (no balance carried over)",
                "0": "Revolving credit (paid minimum, carried balance)",
                "1": "Payment delay for 1 month",
                "2": "Payment delay for 2 months",
                "3+": "Payment delay for 3 or more months"
            }
        },
        "bill_amounts": {
            "name": "Bill Amounts",
            "description": "The total amount due on your statement for each month",
            "help_text": "This is your statement balance before any payments"
        },
        "payment_amounts": {
            "name": "Payment Amounts",
            "description": "How much you paid toward each month's bill",
            "help_text": "Higher payments relative to bill amounts show responsible credit management"
        }
    }
    
    # Add payment status fields
    months = ["September", "August", "July", "June", "May", "April"]
    payment_fields = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    
    for i, field in enumerate(payment_fields):
        field_info[field] = {
            "name": f"{months[i]} Payment Status",
            "description": f"Your payment status for {months[i]}",
            "type": "select",
            "options": [
                {"value": -2, "label": "No consumption"},
                {"value": -1, "label": "Paid in full"},
                {"value": 0, "label": "Revolving credit"},
                {"value": 1, "label": "1 month late"},
                {"value": 2, "label": "2 months late"},
                {"value": 3, "label": "3 months late"},
                {"value": 4, "label": "4+ months late"}
            ]
        }
    
    # Add bill amount fields
    bill_fields = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
    for i, field in enumerate(bill_fields):
        field_info[field] = {
            "name": f"{months[i]} Bill Amount",
            "description": f"Your total bill amount for {months[i]}",
            "type": "number",
            "help_text": "The statement balance for this month"
        }
    
    # Add payment amount fields
    payment_amt_fields = ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]
    for i, field in enumerate(payment_amt_fields):
        field_info[field] = {
            "name": f"{months[i]} Payment Amount",
            "description": f"Amount you paid for {months[i]}'s bill",
            "type": "number",
            "help_text": "How much you paid toward this month's bill"
        }
    
    return JSONResponse(content=field_info)

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"], 
         description="IMPORTANT: Use POST method with JSON body containing all credit data fields")
async def predict(data: CreditData):
    try:
        # Convert input data to DataFrame
        input_df = pd.DataFrame([data.dict()])
        
        # Create engineered features
        # 1. Calculate debt ratio
        input_df['DEBT_RATIO'] = input_df[['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']].mean(axis=1) / input_df['LIMIT_BAL'].replace(0, 1)
        
        # 2. Calculate payment ratios
        for i in range(1, 7):
            bill_col = f'BILL_AMT{i}'
            pay_col = f'PAY_AMT{i}'
            ratio_col = f'PAY_RATIO_{i}'
            input_df[ratio_col] = input_df[pay_col] / input_df[bill_col].replace(0, 1)
            input_df[ratio_col] = input_df[ratio_col].replace([np.inf, -np.inf], 0).fillna(0)
        
        # 3. Calculate average payment delay
        input_df['AVG_PAY_DELAY'] = input_df[['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']].mean(axis=1)
        
        # 4. Binary indicator of previous delay
        input_df['HAS_PREV_DELAY'] = ((input_df[['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']] > 0).sum(axis=1) > 0).astype(int)
        
        # 5. Number of months with delay
        input_df['NUM_DELAY_MONTHS'] = (input_df[['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']] > 0).sum(axis=1)
        
        # Scale features
        scaled_features = scaler.transform(input_df)
        
        # Make prediction
        prediction_proba = model.predict_proba(scaled_features)[0]
        default_probability = prediction_proba[1]
        
        # Convert to US FICO Score range (300-850)
        # Lower default probability = higher credit score
        fico_score = 850 - int(default_probability * 550)  # Scale to FICO range
        
        # Determine risk label based on FICO score ranges
        if fico_score >= 740:
            risk_label = "Excellent"
        elif fico_score >= 670:
            risk_label = "Good"
        elif fico_score >= 580:
            risk_label = "Fair"
        else:
            risk_label = "Poor"
        
        # Calculate debt ratio for loan approval
        debt_ratio = float(input_df['DEBT_RATIO'].values[0])
        
        # Calculate loan approval probabilities
        loan_approval_data = calculate_loan_approval(
            fico_score=fico_score,
            default_probability=float(default_probability),
            debt_ratio=debt_ratio
        )
        
        # Prepare response
        response = {
            "risk_label": risk_label,
            "risk_probability": float(default_probability),
            "fico_score": fico_score,
            "risk_details": {
                "default_probability": float(default_probability),
                "non_default_probability": float(prediction_proba[0]),
                "risk_level": risk_label,
                "us_fico_score": fico_score,
                "top_factors": get_top_factors(input_df)
            },
            "loan_approval": loan_approval_data
        }
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

def get_top_factors(df):
    """Get the top risk factors for this prediction based on US credit standards"""
    factors = {}
    
    # Payment history (35% of FICO score)
    delays = df['NUM_DELAY_MONTHS'].values[0]
    if delays > 0:
        factors["payment_history"] = f"{int(delays)} month(s) with late payments"
        factors["impact"] = "high"
    
    # Credit utilization (30% of FICO score)
    debt_ratio = df['DEBT_RATIO'].values[0]
    if debt_ratio > 0.3:  # US standard is to keep utilization below 30%
        util_percent = min(100, round(debt_ratio * 100, 1))
        factors["credit_utilization"] = f"{util_percent}% (recommended: below 30%)"
        if debt_ratio > 0.7:
            factors["impact"] = "very high" 
        else:
            factors["impact"] = "high"
    
    # Payment amounts (15% of FICO score)
    avg_payment_ratio = df[['PAY_RATIO_1', 'PAY_RATIO_2', 'PAY_RATIO_3']].mean(axis=1).values[0]
    if avg_payment_ratio < 0.1:  # Only paying minimum amounts
        factors["minimum_payments"] = "Consistently making minimum payments only"
        factors["impact"] = "medium"
    
    # Length of credit history - estimated from data patterns
    recent_activity = abs(df[['PAY_0', 'PAY_2', 'PAY_3']].mean(axis=1).values[0])
    if recent_activity > 3:
        factors["recent_credit_issues"] = "Recent credit problems detected"
        factors["impact"] = "high"
    
    # If no factors found, provide default message
    if not factors:
        return {"message": "No specific risk factors identified. Good credit profile!"}
    
    return factors

def calculate_loan_approval(fico_score, default_probability, debt_ratio):
    """Calculate loan approval probabilities based on credit score and other factors"""
    loan_types = {
        "mortgage": {
            "min_score": 620,
            "good_score": 740,
            "max_dti": 0.43,  # Debt-to-income ratio
            "importance": [0.6, 0.3, 0.1]  # Weights for FICO, default prob, DTI
        },
        "auto_loan": {
            "min_score": 580,
            "good_score": 700,
            "max_dti": 0.5,
            "importance": [0.5, 0.3, 0.2]
        },
        "credit_card": {
            "min_score": 550,
            "good_score": 720,
            "max_dti": 0.45,
            "importance": [0.7, 0.2, 0.1]
        },
        "personal_loan": {
            "min_score": 600,
            "good_score": 720,
            "max_dti": 0.4,
            "importance": [0.5, 0.4, 0.1]
        }
    }
    
    results = {}
    
    for loan_type, criteria in loan_types.items():
        # Calculate score component (0-1)
        if fico_score < criteria["min_score"]:
            score_component = 0
        elif fico_score >= criteria["good_score"]:
            score_component = 1
        else:
            range_size = criteria["good_score"] - criteria["min_score"]
            score_component = (fico_score - criteria["min_score"]) / range_size
            
        # Calculate default probability component (inverted, 0-1)
        default_component = 1 - min(1, default_probability * 2)
        
        # Calculate debt ratio component (0-1)
        if debt_ratio > criteria["max_dti"]:
            dti_component = 0
        else:
            dti_component = 1 - (debt_ratio / criteria["max_dti"])
        
        # Weighted approval score
        weights = criteria["importance"]
        approval_score = (
            weights[0] * score_component + 
            weights[1] * default_component + 
            weights[2] * dti_component
        )
        
        # Convert to approval probability
        approval_probability = min(0.99, approval_score)
        
        # Determine interest rate range based on approval probability
        if approval_probability > 0.9:
            rate_range = "Lowest Available"
        elif approval_probability > 0.75:
            rate_range = "Below Average"
        elif approval_probability > 0.6:
            rate_range = "Average"
        elif approval_probability > 0.4:
            rate_range = "Above Average"
        else:
            rate_range = "High"
            
        # Include result
        results[loan_type] = {
            "approval_probability": round(approval_probability * 100, 1),
            "likely_rate": rate_range,
            "approved": approval_probability > 0.5
        }
    
    return results

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
