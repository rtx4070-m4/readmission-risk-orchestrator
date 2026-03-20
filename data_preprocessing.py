
import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess(df):
    df = df.copy()
    df = df.fillna(df.median(numeric_only=True))
    df = pd.get_dummies(df, drop_first=True)
    return df

def split(df, target):
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X,y,test_size=0.2,random_state=42)
