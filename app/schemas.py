"""
app/schemas.py

Pydantic models defining the FastAPI request/response contracts.
The request fields mirror the RAW dataset columns (before any encoding/scaling) —
the API handles preprocessing internally so consumers just send human-readable values.
"""

from typing import Literal
from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    Customer_Age: int = Field(..., ge=18, le=100, example=45)
    Gender: Literal['M', 'F'] = Field(..., example='M')
    Dependent_count: int = Field(..., ge=0, le=10, example=3)
    Education_Level: Literal[
        'Uneducated', 'High School', 'Graduate', 'Post-Graduate', 'Doctorate'
    ] = Field(..., example='Graduate')
    Marital_Status: Literal['Married', 'Single'] = Field(..., example='Married')
    Income_Category: Literal[
        'Less than $40K', '$40K - $60K', '$60K - $80K', '$80K - $120K', '$120K +'
    ] = Field(..., example='$60K - $80K')
    Card_Category: Literal['Blue', 'Silver', 'Gold', 'Platinum'] = Field(..., example='Blue')
    Months_on_book: int = Field(..., ge=0, le=60, example=36)
    Total_Relationship_Count: int = Field(..., ge=1, le=6, example=5)
    Months_Inactive_12_mon: int = Field(..., ge=0, le=12, example=1)
    Contacts_Count_12_mon: int = Field(..., ge=0, le=10, example=2)
    Credit_Limit: float = Field(..., ge=0, example=8500.0)
    Total_Revolving_Bal: float = Field(..., ge=0, example=1200.0)
    Avg_Open_To_Buy: float = Field(..., ge=0, example=7300.0)
    Total_Amt_Chng_Q4_Q1: float = Field(..., ge=0, example=0.76)
    Total_Trans_Amt: float = Field(..., ge=0, example=4500.0)
    Total_Trans_Ct: int = Field(..., ge=0, example=65)
    Total_Ct_Chng_Q4_Q1: float = Field(..., ge=0, example=0.7)
    Avg_Utilization_Ratio: float = Field(..., ge=0, le=1, example=0.14)

    class Config:
        json_schema_extra = {
            "example": {
                "Customer_Age": 45,
                "Gender": "M",
                "Dependent_count": 3,
                "Education_Level": "Graduate",
                "Marital_Status": "Married",
                "Income_Category": "$60K - $80K",
                "Card_Category": "Blue",
                "Months_on_book": 36,
                "Total_Relationship_Count": 5,
                "Months_Inactive_12_mon": 1,
                "Contacts_Count_12_mon": 2,
                "Credit_Limit": 8500.0,
                "Total_Revolving_Bal": 1200.0,
                "Avg_Open_To_Buy": 7300.0,
                "Total_Amt_Chng_Q4_Q1": 0.76,
                "Total_Trans_Amt": 4500.0,
                "Total_Trans_Ct": 65,
                "Total_Ct_Chng_Q4_Q1": 0.7,
                "Avg_Utilization_Ratio": 0.14
            }
        }


class PredictionResponse(BaseModel):
    churn_prediction: Literal['Churn', 'Stay'] = Field(..., example='Stay')
    churn_probability: float = Field(..., example=0.12, description="Probability of churn (0-1)")
    risk_level: Literal['Low', 'Medium', 'High'] = Field(..., example='Low')
    model_used: str = Field(..., example='RF')


class BatchPredictionRequest(BaseModel):
    customers: list[CustomerInput]


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str | None = None
