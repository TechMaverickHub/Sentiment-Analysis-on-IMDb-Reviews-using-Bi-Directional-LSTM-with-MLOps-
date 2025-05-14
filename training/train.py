import mlflow
import mlflow.tensorflow
from tensorflow.keras.callbacks import EarlyStopping
from models.bilstm_model import create_model
from data.imdb_loader import load_data

def train(config, best_params):
    mlflow.tensorflow.autolog()
    with mlflow.start_run():
        X_train, X_test, y_train, y_test = load_data(config)
        model = create_model(config, best_params['embedding_dim'], best_params['lstm_units'], best_params['dropout'])

        model.fit(
            X_train, y_train,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            validation_split=0.2,
            callbacks=[EarlyStopping(patience=2)],
            verbose=2
        )

        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        mlflow.log_metric("test_loss", loss)
        mlflow.log_metric("test_accuracy", acc)

        model.save(config['model_path'], save_format="keras")

