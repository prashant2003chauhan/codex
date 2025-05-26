
# Flask
from flask import Flask, render_template, request
# Data manipulation
import pandas as pd
# Matrices manipulation
import numpy as np
# Script logging
import logging
import os
# ML model
import joblib
# JSON manipulation
import json
# Utilities
import sys
import os
from pathlib import Path
import re

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='template')

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name_')

# Get base directory
BASE_DIR = Path(__file__).resolve().parent

# Function to load model and schema paths from environment variables
MODEL_PATH = os.environ.get('MODEL_PATH', BASE_DIR / 'bin' / 'xgboostModel.pkl')
SCHEMA_PATH = os.environ.get('SCHEMA_PATH', BASE_DIR / 'data' / 'columns_set.json')

# Function to predict loan approval
def ValuePredictor(data: pd.DataFrame) -> int:
    """
    Predict loan approval using the XGBoost model.
    Args:
        data: Input DataFrame with features.
    Returns:
        Prediction result (0 or 1).
    """
    try:
        logger.info(f"Loading model from {MODEL_PATH}")
        loaded_model = joblib.load(MODEL_PATH)
        result = loaded_model.predict(data)
        logger.info("Prediction successful")
        return result[0]
    except Exception as e:
        logger.error(f"Model prediction failed: {str(e)}")
        raise Exception(f"Prediction error: {str(e)}")

# Home page route
@app.route('/')
def home():
    logger.info("Serving home page")
    return render_template('index.html')

# Prediction route
@app.route('/prediction', methods=['POST'])
def predict():
    logger.info("Received prediction request")
    if request.method == 'POST':
        try:
            # Sanitize and validate form inputs
            name = re.sub(r'[<>]', '', request.form.get('name', '').strip())
            gender = request.form.get('gender')
            education = request.form.get('education')
            self_employed = request.form.get('self_employed')
            marital_status = request.form.get('marital_status')
            dependents = request.form.get('dependents')
            applicant_income = request.form.get('applicant_income')
            coapplicant_income = request.form.get('coapplicant_income')
            loan_amount = request.form.get('loan_amount')
            loan_term = request.form.get('loan_term')
            credit_history = request.form.get('credit_history')
            property_area = request.form.get('property_area')

            # Validate required fields
            if not all([name, gender, education, self_employed, marital_status, dependents,
                        applicant_income, coapplicant_income, loan_amount, loan_term,
                        credit_history, property_area]):
                logger.warning("Missing form fields")
                return render_template('error.html', prediction="Please fill all required fields.")

            # Convert numerical inputs to floats
            try:
                applicant_income = float(applicant_income)
                coapplicant_income = float(coapplicant_income)
                loan_amount = float(loan_amount)
                loan_term = float(loan_term)
            except ValueError:
                logger.warning("Invalid numerical input")
                return render_template('error.html', prediction="Please enter valid numerical values.")

            # Load JSON schema
            logger.info(f"Loading schema from {SCHEMA_PATH}")
            try:
                with open(SCHEMA_PATH, 'r') as f:
                    cols = json.load(f)
                schema_cols = cols['data_columns'].copy()  # Create a copy to avoid modifying original
            except (FileNotFoundError, json.JSONDecodeError) as e:
                logger.error(f"Schema loading failed: {str(e)}")
                return render_template('error.html', prediction="Server error: Unable to load schema.")

            # Initialize all schema columns to 0
            for key in schema_cols:
                schema_cols[key] = 0

            # Parse categorical columns
            try:
                col = f'Dependents_{dependents}'
                if col in schema_cols:
                    schema_cols[col] = 1
            except Exception as e:
                logger.warning(f"Dependents parsing error: {str(e)}")

            try:
                col = f'Property_Area_{property_area}'
                if col in schema_cols:
                    schema_cols[col] = 1
            except Exception as e:
                logger.warning(f"Property area parsing error: {str(e)}")

            # Assign numerical and categorical values
            schema_cols['ApplicantIncome'] = applicant_income
            schema_cols['CoapplicantIncome'] = coapplicant_income
            schema_cols['LoanAmount'] = loan_amount
            schema_cols['Loan_Amount_Term'] = loan_term
            schema_cols['Gender_Male'] = 1 if gender == 'Male' else 0
            schema_cols['Married_Yes'] = 1 if marital_status == 'Yes' else 0
            schema_cols['Education_Not Graduate'] = 1 if education == 'Not Graduate' else 0
            schema_cols['Self_Employed_Yes'] = 1 if self_employed == 'Yes' else 0
            schema_cols['Credit_History_1.0'] = 1 if credit_history == '1.0' else 0

            # Create DataFrame
            try:
                df = pd.DataFrame(
                    data={k: [v] for k, v in schema_cols.items()},
                    dtype=np.float64
                )
                logger.info(f"DataFrame created with dtypes:\n{df.dtypes}")
            except Exception as e:
                logger.error(f"DataFrame creation failed: {str(e)}")
                return render_template('error.html', prediction="Server error: Invalid data format.")

            # Make prediction
            try:
                result = ValuePredictor(data=df)
            except Exception as e:
                logger.error(f"Prediction failed: {str(e)}")
                return render_template('error.html', prediction=f"Prediction error: {str(e)}")

            # Format prediction message
            prediction = (
                f"Dear Mr/Mrs/Ms {name}, your loan is approved!" if int(result) == 1
                else f"Sorry Mr/Mrs/Ms {name}, your loan is rejected!"
            )
            logger.info(f"Prediction result: {prediction}")

            return render_template('prediction.html', prediction=prediction)

        except Exception as e:
            logger.error(f"Unexpected error in prediction: {str(e)}")
            return render_template('error.html', prediction="An unexpected error occurred. Please try again.")
    else:
        logger.warning("Invalid request method")
        return render_template('error.html', prediction="Invalid request. Please use the form to submit.")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))  # Render uses dynamic ports
    logger.info(f"Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)