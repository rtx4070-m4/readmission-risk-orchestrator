
import shap

def get_shap(model,X):
    explainer = shap.TreeExplainer(model)
    return explainer.shap_values(X)
