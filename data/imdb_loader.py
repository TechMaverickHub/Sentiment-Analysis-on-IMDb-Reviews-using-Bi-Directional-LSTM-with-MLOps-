from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import yaml

def load_data(config):
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=config['max_features'])
    X_train = pad_sequences(X_train, maxlen=config['maxlen'])
    X_test = pad_sequences(X_test, maxlen=config['maxlen'])
    return X_train, X_test, y_train, y_test
