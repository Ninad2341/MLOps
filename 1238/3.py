import mlflow, mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Use local folder ./mlruns (file-based)
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("iris-registry-demo")

# Data
X, y = load_iris(return_X_y=True, as_frame=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

with mlflow.start_run() as run:
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(Xtr, ytr)

    # metric
    acc = float(accuracy_score(yte, clf.predict(Xte)))
    mlflow.log_metric("accuracy", acc)

    # log model
    mlflow.sklearn.log_model(clf, "model")

    # register model
    model_uri = f"runs:/{run.info.run_id}/model"
    reg = mlflow.register_model(model_uri, "iris_rf")

# tag version (since stages may break on file store)
client = MlflowClient(tracking_uri="file:./mlruns")
client.set_model_version_tag("iris_rf", reg.version, "stage", "Production")

print(f"Registered iris_rf v{reg.version}. Accuracy={acc}.")
