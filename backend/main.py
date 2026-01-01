from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Load BOTH models and regions
tree_model = joblib.load("models/tree_model.joblib")
logistic_model = joblib.load("models/logistic_model.joblib")
VALID_REGIONS = joblib.load("models/regions.joblib")

class InputData(BaseModel):
    gdp_per_capita: float
    infant_mortality: float
    fertility: float
    unemployment: float
    internet_users: float
    region: str

@app.get("/regions")
async def get_regions():
    return {"regions": VALID_REGIONS}

@app.post("/predict")
async def predict(data: InputData):
    try:
        df = pd.DataFrame([data.dict()])

        # Tree / Random Forest prediction
        tree_pred = tree_model.predict(df)[0]
        tree_proba = tree_model.predict_proba(df)[0]
        tree_conf = float(max(tree_proba))

        # Logistic Regression prediction
        log_pred = logistic_model.predict(df)[0]
        log_proba = logistic_model.predict_proba(df)[0]
        log_conf = float(max(log_proba))

        return {
            "logistic_regression": str(log_pred),
            "decision_tree": str(tree_pred),
            "model_confidence": {
                "logistic": log_conf,
                "tree": tree_conf
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))