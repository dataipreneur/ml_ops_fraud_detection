import os
os.environ['LOKY_MAX_CPU_COUNT'] = '16'

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, precision_recall_curve, auc
import joblib
import logging
import sys
import mlflow
from mlflow.sklearn import log_model
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(data_path):
    """
    Load data from a given path.
    """
    try:
        data = pd.read_csv(data_path)
        logging.info(f"Data loaded successfully from {data_path}")
        return data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        sys.exit(1)

def preprocess_data(data):
    """
    Preprocess the data, returning scaled features and labels.
    """
    X = data.drop('Class', axis=1)
    y = data['Class']

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def train_model(X, y, test_size=0.3):
    """
    Train the model and save the test set.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
    model.fit(X_train, y_train)

    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and return metrics.
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Compute metrics
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_test, y_pred)
    metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
    metrics['classification_report'] = classification_report(y_test, y_pred)

    # Plot and save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='g')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    confusion_matrix_path = "assets/confusion_matrix.png"
    plt.savefig(confusion_matrix_path)
    mlflow.log_artifact(confusion_matrix_path)

    # Plot and save precision-recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    auprc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, label=f'AUPRC = {auprc:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc='best')
    precision_recall_curve_path = "assets/precision_recall_curve.png"
    plt.savefig(precision_recall_curve_path)
    mlflow.log_artifact(precision_recall_curve_path)

    metrics['auprc'] = auprc

    return metrics

def save_model(model, model_save_path):
    """
    Save the model to a given path.
    """
    try:
        joblib.dump(model, model_save_path)
        logging.info(f"Model saved to {model_save_path}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")

if __name__ == '__main__':
    # Set MLflow tracking URI if provided
    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', None)
    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)
        logging.info(f"MLflow tracking URI set to: {mlflow_uri}")

    mlflow.set_experiment("fraud_detection")

    data_path = os.getenv('DATA_PATH', 'data/creditcard.csv')
    simulated_data_path = os.getenv('SIMULATED_DATA_PATH', 'data/simulated_data.csv')
    model_save_path = os.getenv('MODEL_SAVE_PATH', 'model/saved_models/model.pkl')

    # Load the actual dataset
    actual_data = load_data(data_path)
    X_actual, y_actual = preprocess_data(actual_data)

    # Load the simulated dataset
    simulated_data = load_data(simulated_data_path)
    X_simulated, y_simulated = preprocess_data(simulated_data)

    with mlflow.start_run():
        # Train the model on the actual data
        model, _, _ = train_model(X_actual, y_actual)

        # Log model and parameters
        params = model.get_params()
        mlflow.log_params(params)

        # Evaluate model on simulated data
        metrics = evaluate_model(model, X_simulated, y_simulated)

        # Log metrics
        mlflow.log_metric("accuracy", metrics['accuracy'])
        mlflow.log_metric("roc_auc", metrics['roc_auc'])
        mlflow.log_metric("auprc", metrics['auprc'])

        # Log model with signature and input example
        input_example = X_simulated[:1]  # Use first row as example
        model_info = log_model(
            model,
            "model",
            input_example=input_example,
            registered_model_name="fraud_detection"
        )

        # Save the model locally
        save_model(model, model_save_path)

        logging.info(f"Model training completed with accuracy: {metrics['accuracy']}, ROC AUC: {metrics['roc_auc']}, AUPRC: {metrics['auprc']}")
        logging.info(metrics['classification_report'])

        # Promote model to Production stage
        from mlflow.tracking import MlflowClient
        client = MlflowClient()

        # Get the latest version
        model_name = "fraud_detection"
        latest_versions = client.get_latest_versions(model_name, stages=["None"])
        if latest_versions:
            latest_version = latest_versions[0].version
            # Transition to Production
            client.transition_model_version_stage(
                name=model_name,
                version=latest_version,
                stage="Production",
                archive_existing_versions=True
            )
            logging.info(f"Model version {latest_version} promoted to Production stage")