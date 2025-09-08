from data.imdb_loader import load_data
import pytest
import yaml

@pytest.fixture
def config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)

def test_data_shapes(config):
    X_train, X_test, y_train, y_test = load_data(config)
    assert X_train.shape[1] == config['maxlen']
    assert len(X_train) == len(y_train)
