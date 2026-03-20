
import joblib, pandas as pd

def run(input_dict):
    model = joblib.load("models/model.pkl")
    df = pd.DataFrame([input_dict])
    prob = model.predict_proba(df)[0][1]
    risk = "High" if prob>0.7 else "Medium" if prob>0.4 else "Low"
    return prob, risk
