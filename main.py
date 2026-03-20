
from fastapi import FastAPI
from src.predict import run

app = FastAPI()

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/predict")
def predict(data:dict):
    prob, risk = run(data)
    return {"readmission_probability":prob,"risk_level":risk}
