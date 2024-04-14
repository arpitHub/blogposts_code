import pytest
from pathlib import Path
from ..src.preprocessing import load_iris_dataset,preprocess_data
from ..src.model import train_and_evaluate_model

data_dir = Path(__file__).parent.parent / 'data'  # Navigate up to the project root
iris_path = data_dir / 'iris.csv'

@pytest.fixture
def preprocessed_iris_data():
    df = load_iris_dataset(iris_data)
    return preprocess_data(df)

def test_train_and_evaluate_model(preprocessed_iris_data):
    model, accuracy = train_and_evaluate_model(preprocessed_iris_data)
    assert accuracy > 0.8
