import pandas as pd

def load_iris_dataset(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    # Drop rows with missing values
    df = df.dropna()
    # Encode target variable
    df['species'] = df['species'].astype('category').cat.codes
    return df
