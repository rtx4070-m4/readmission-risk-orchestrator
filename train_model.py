
import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def train_models(X_train,y_train):
    rf = RandomForestClassifier(n_estimators=200)
    rf.fit(X_train,y_train)

    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_train,y_train)

    return rf, xgb

def save(model,path):
    joblib.dump(model,path)
