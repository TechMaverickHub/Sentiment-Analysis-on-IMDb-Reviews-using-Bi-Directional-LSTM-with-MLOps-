import yaml
from training.train import train

if __name__ == "__main__":
    with open("config/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Load best hyperparameters
    with open("config/best_params.yaml") as f:
        best_params = yaml.safe_load(f)
    train(config, best_params)
