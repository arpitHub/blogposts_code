import pytest
from pathlib import Path
from src.preprocessing import load_iris_dataset, preprocess_data

data_dir = Path(__file__).parent.parent / 'data'  # Navigate up to the project root
iris_path = data_dir / 'iris.csv'

@pytest.fixture
def test_load_iris_dataset():
    df = load_iris_dataset(iris_path)
    assert not df.empty

def test_preprocess_data():
    df = load_iris_dataset(iris_path)
    preprocessed_df = preprocess_data(df)
    assert preprocessed_df.isna().sum().sum() == 0
    assert 'species' in preprocessed_df.columns

def test_missing_values():
    df = load_iris_dataset(iris_path)
    assert not df.isnull().values.any(), "Dataset contains missing values"

def test_no_duplicates():
    df = load_iris_dataset(iris_path)
    assert not df.duplicated().any(), "Dataset contains duplicate records"


def test_column_datatypes():
    df = load_iris_dataset(iris_path)
    expected_datatypes = {
        'sepal length (cm)': 'float64',
        'sepal width (cm)': 'float64',
        'petal length (cm)': 'float64',
        'petal width (cm)': 'float64',
        'species': 'object'
    }
    for col, dtype in expected_datatypes.items():
        assert df[col].dtype == dtype, f"Unexpected datatype for column {col}"
