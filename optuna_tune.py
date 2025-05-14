import optuna
import yaml
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from data.imdb_loader import load_data
from models.bilstm_model import create_model

# Load config
with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

# Load data
X_train, X_test, y_train, y_test = load_data(config)

# Enable MLflow autologging
mlflow.tensorflow.autolog()

def objective(trial):
    # Suggest hyperparameters
    embedding_dim = trial.suggest_categorical("embedding_dim", [64, 128, 256])
    lstm_units = trial.suggest_categorical("lstm_units", [32, 64, 128])
    dropout = trial.suggest_float("dropout", 0.2, 0.5)

    with mlflow.start_run(nested=True):
        # Create model with trial-specific params
        model = create_model(config, embedding_dim=embedding_dim, lstm_units=lstm_units, dropout=dropout)

        # Compile
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Fit
        history = model.fit(
            X_train, y_train,
            epochs=5,
            batch_size=config['batch_size'],
            validation_split=0.2,
            verbose=0
        )

        val_accuracy = history.history['val_accuracy'][-1]
        mlflow.log_params({
            'embedding_dim': embedding_dim,
            'lstm_units': lstm_units,
            'dropout': dropout
        })
        mlflow.log_metric('val_accuracy', val_accuracy)

        return val_accuracy

# Run Optuna study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

# Save best result
print("Best trial:")
print(study.best_trial.params)

with open("config/best_params.yaml", "w") as f:
    yaml.dump(study.best_trial.params, f)
