
# 🏥 30-Day Readmission Risk Orchestrator (Advanced)

## Features
- End-to-end ML pipeline
- Random Forest + XGBoost
- Optuna tuning
- SHAP explainability
- FastAPI backend
- Streamlit dashboard
- Docker ready
- Logging system

## Run
pip install -r requirements.txt
python run_pipeline.py

## API
uvicorn api.main:app --reload

## Dashboard
streamlit run dashboard/app.py
