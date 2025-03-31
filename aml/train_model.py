import os
import pickle
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from azureml.core import Workspace, Experiment

# Connect to Azure ML Workspace
ws = Workspace(
    subscription_id="63c6a983-a039-496b-b764-06a48da78e33",
    resource_group="mlops-rg",
    workspace_name="mlops-workspace"
)

def train_and_register_model():
    # Create experiment and log run
    experiment = Experiment(ws, "iris-classifier")
    run = experiment.start_logging()

    # Train model
    data = load_iris()
    X, y = data.data, data.target
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    # Save model
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Log accuracy
    run.log("accuracy", model.score(X, y))
    run.complete()

    # Register model
    run.register_model(model_name="iris-model", model_path="outputs/model.pkl")
    print("✅ Model registered successfully.")

# Make sure script runs
if __name__ == "__main__":
    train_and_register_model()
