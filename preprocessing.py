import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    df = df.drop_duplicates()

    categorical_cols = [
        'Shipping_Type',
        'Traffic_Level',
        'Weather_Condition'
    ]

    for col in categorical_cols:
        df[col] = df[col].astype('category').cat.codes

    return df
