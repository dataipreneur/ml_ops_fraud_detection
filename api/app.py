from flask import Flask, request, jsonify, render_template
import mlflow
import pandas as pd
import os
import logging
import joblib
from mlflow.pyfunc import load_model
from flask_basicauth import BasicAuth

app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = os.environ.get('AUTH_USERNAME', 'admin')
app.config['BASIC_AUTH_PASSWORD'] = os.environ.get('AUTH_PASSWORD', 'password')

basic_auth = BasicAuth(app)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Environment variables and default values
MODEL_URI = os.getenv('MODEL_URI', 'models:/fraud_detection/Production')
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', None)
SERVER_PORT = os.getenv('PORT', '8000')
DEBUG_MODE = os.getenv('DEBUG', 'False').lower() == 'true'

# Set MLflow tracking URI if provided
if MLFLOW_TRACKING_URI:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logging.info(f"MLflow tracking URI set to: {MLFLOW_TRACKING_URI}")

# Load the model
try:
    # Check if MODEL_URI is a pickle file or MLflow model
    if MODEL_URI.endswith('.pkl'):
        model = joblib.load(MODEL_URI)
        logging.info(f"Model loaded successfully from joblib file: {MODEL_URI}")
    else:
        model = load_model(MODEL_URI)
        logging.info(f"Model loaded successfully from MLflow Model Registry: {MODEL_URI}")
except Exception as e:
    logging.error(f"Error loading model from {MODEL_URI}: {e}")
    logging.warning("Attempting to load from fallback pickle file...")
    try:
        model = joblib.load('./model/saved_models/model.pkl')
        logging.info("Model loaded successfully from fallback pickle file")
    except Exception as fallback_error:
        logging.error(f"Fallback also failed: {fallback_error}")
        model = None

@app.route('/')
@basic_auth.required
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@basic_auth.required
def predict():
    """Endpoint to make fraud detection predictions."""
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500

    data = request.form.to_dict()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        # Input validation
        required_fields = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                           'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                           'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({'error': f'Missing required fields: {missing_fields}'}), 400

        # Convert string values to floats
        data_numeric = {k: float(v) for k, v in data.items()}

        df = pd.DataFrame([data_numeric])
        prediction = model.predict(df)[0]  # Class prediction (0 or 1)
        logging.info(f"Prediction: {prediction}")
        is_fraud = bool(prediction)

        # Log prediction and input data for monitoring
        logging.info(f"Prediction: {prediction}, Is Fraud: {is_fraud}")

        return jsonify({'prediction': int(prediction), 'is_fraud': is_fraud})
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': 'An error occurred during prediction'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(SERVER_PORT), debug=DEBUG_MODE)