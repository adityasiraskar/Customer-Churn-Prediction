"""
scripts/test_api.py

Quick manual test against a running FastAPI instance.

Usage:
    1. uvicorn app.main:app --reload --port 8000
    2. python scripts/test_api.py
"""

import requests

BASE_URL = "http://127.0.0.1:8000"

sample_customer = {
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

# A "high risk" looking customer: low activity, low transaction count, inactive months
high_risk_customer = {
    **sample_customer,
    "Months_Inactive_12_mon": 5,
    "Contacts_Count_12_mon": 6,
    "Total_Trans_Amt": 900.0,
    "Total_Trans_Ct": 15,
    "Total_Relationship_Count": 1,
}


def main():
    print("1) Health check")
    r = requests.get(f"{BASE_URL}/health")
    print(r.status_code, r.json())

    print("\n2) Single prediction (typical customer)")
    r = requests.post(f"{BASE_URL}/predict", json=sample_customer)
    print(r.status_code, r.json())

    print("\n3) Single prediction (high-risk-looking customer)")
    r = requests.post(f"{BASE_URL}/predict", json=high_risk_customer)
    print(r.status_code, r.json())

    print("\n4) Batch prediction")
    r = requests.post(
        f"{BASE_URL}/predict/batch",
        json={"customers": [sample_customer, high_risk_customer]}
    )
    print(r.status_code, r.json())


if __name__ == "__main__":
    main()
