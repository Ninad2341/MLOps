import mlflow, mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)
import numpy as np

mlflow.set_experiment("iris-mlflow-demo")

# Load data
X, y = load_iris(return_X_y=True, as_frame=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)

with mlflow.start_run():

    # Hyperparameters
    n_estimators = 100
    max_depth = 5

    # Train model
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    clf.fit(Xtr, ytr)
    preds = clf.predict(Xte)

    acc = accuracy_score(yte, preds)
    prec = precision_score(yte, preds)    
    rec = recall_score(yte, preds)       
    f1 = f1_score(yte, preds)          

    cm = confusion_matrix(yte, preds)

    # Log parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # Log metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    # Log confusion matrix as artifact
    np.savetxt("confusion_matrix.csv", cm, delimiter=",")
    mlflow.log_artifact("confusion_matrix.csv")

    # Log model
    mlflow.sklearn.log_model(clf, "model")
