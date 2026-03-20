
import pandas as pd
from src.data_preprocessing import preprocess, split
from src.train_model import train_models, save
from src.evaluate_model import evaluate

df = pd.read_csv("data/raw/data.csv")
df = preprocess(df)

X_train,X_test,y_train,y_test = split(df,"target")

rf,xgb = train_models(X_train,y_train)

print("RF:",evaluate(rf,X_test,y_test))
print("XGB:",evaluate(xgb,X_test,y_test))

save(xgb,"models/model.pkl")
