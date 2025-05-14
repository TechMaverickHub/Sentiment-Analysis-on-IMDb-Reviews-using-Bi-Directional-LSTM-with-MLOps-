from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense

def create_model(config, embedding_dim, lstm_units, dropout):
    model = Sequential([
        Embedding(config['max_features'], embedding_dim, input_length=config['maxlen']),
        Bidirectional(LSTM(lstm_units)),
        Dropout(dropout),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


