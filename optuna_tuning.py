
import optuna
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        "n_estimators": trial.suggest_int("n_estimators",50,200),
        "max_depth": trial.suggest_int("max_depth",3,10),
        "learning_rate": trial.suggest_float("learning_rate",0.01,0.3)
    }
    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", **params)
    model.fit(X_train,y_train)
    preds = model.predict_proba(X_val)[:,1]
    return roc_auc_score(y_val,preds)
