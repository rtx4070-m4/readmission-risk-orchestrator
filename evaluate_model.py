
from sklearn.metrics import classification_report, roc_auc_score

def evaluate(model,X,y):
    preds = model.predict(X)
    prob = model.predict_proba(X)[:,1]
    return {
        "report": classification_report(y,preds),
        "roc_auc": roc_auc_score(y,prob)
    }
