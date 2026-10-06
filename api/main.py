import joblib
import pandas as pd
 
from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
 
app = FastAPI(
    title="Loan Prediction API",
    description="This page shows loan prediction.",
    version="1.0"
)
 
model = joblib.load("./model/loan_default.pkl")
 
 
class LoanInput(BaseModel):
 
    current_loan_amount: float = Field(alias="Current Loan Amount")
    term: float = Field(alias="Term")
    credit_score: float = Field(alias="Credit Score")
    annual_income: float = Field(alias="Annual Income")
    years_in_current_job: float = Field(alias="Years in current job")
    home_ownership: float = Field(alias="Home Ownership")
    purpose: float = Field(alias="Purpose")
    monthly_debt: float = Field(alias="Monthly Debt")
    years_of_credit_history: float = Field(alias="Years of Credit History")
    months_since_last_delinquent: float = Field(alias="Months since last delinquent")
    number_of_open_accounts: float = Field(alias="Number of Open Accounts")
    number_of_credit_problems: float = Field(alias="Number of Credit Problems")
    current_credit_balance: float = Field(alias="Current Credit Balance")
    maximum_open_credit: float = Field(alias="Maximum Open Credit")
    bankruptcies: float = Field(alias="Bankruptcies")
    tax_liens: float = Field(alias="Tax Liens")
 
 
@app.get("/")
def home():
    return {
        "message": "Loan Prediction API"
    }
 
 
@app.get("/docs", include_in_schema=False)
def custom_docs():
 
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Loan Prediction API"
    )
 
 
@app.post("/predict")
def predict(data: LoanInput):
 
    input_data = pd.DataFrame([
        data.model_dump(by_alias=True)
    ])
 
    prediction = model.predict(input_data)
 
    return {
        "prediction": float(prediction[0])
    }
 