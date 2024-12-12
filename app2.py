import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the model and label encoder

with open("xgb_best_model_2.pkl", 'rb') as model_file:
    model = pickle.load(model_file)

with open("label_encoder_2.pkl", 'rb') as encoder_file:
    label_encoder = pickle.load(encoder_file)

# Define required columns for batch predictions
required_columns = ['index', 'step', 'type', 'amount', 'oldbalanceOrg', 'newBalanceOrig', 'oldbalanceDest', 'newbalanceDest']

# Streamlit App
st.set_page_config(
    page_title="Fraud Detection App",  # Name displayed on the browser tab
    page_icon="🔍",                   # Optional: Emoji or icon for the tab
    layout="wide"                     # Optional: Sets layout to wide mode
)
st.title("Financial Fraud Detection App")
st.write("This app predicts whether a transaction is fraudulent using an XGBoost model.")


# Input Options
st.header("Input Options")
input_option = st.radio(
    "Choose how you want to input data:",
    ("Single Transaction", "Upload CSV File")
)

# Function to predict a single transaction
def predict_single_transaction(index, step, transaction_type, amount, oldbalanceOrg, newBalanceOrig, oldbalanceDest, newbalanceDest):
    try:
        # Clean the transaction type
        transaction_type = transaction_type.strip().upper()

        # Validate transaction type
        if transaction_type not in label_encoder.classes_:
            st.error(f"Invalid transaction type '{transaction_type}'")
            return None

        # Prepare input data
        input_data = np.array([
            [index, step, label_encoder.transform([transaction_type])[0], float(amount), 
             float(oldbalanceOrg), float(newBalanceOrig), float(oldbalanceDest), float(newbalanceDest)]
        ], dtype=np.float32)  # Ensure data type is compatible with XGBoost

        # Validate input shape
        if input_data.shape[1] != model.n_features_in_:
            raise ValueError(f"Model expects {model.n_features_in_} features, but {input_data.shape[1]} provided.")

        # Make prediction
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        return {
            "is_fraudulent": bool(prediction[0]),
            "fraud_probability": prediction_proba[0][1]
        }
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

# Single Transaction Input
if input_option == "Single Transaction":
    st.subheader("Enter Transaction Details")
    transaction_type = st.selectbox("Transaction Type", options=label_encoder.classes_)
    amount = st.number_input("Transaction Amount", min_value=0.0, step=0.01)
    step = st.number_input("Step", min_value=1, step=1)
    oldbalanceOrg = st.number_input("Original Account Balance", min_value=0.0, step=0.01)
    newBalanceOrig = st.number_input("New Account Balance", min_value=0.0, step=0.01)
    oldbalanceDest = st.number_input("Destination Original Balance", min_value=0.0, step=0.01)
    newbalanceDest = st.number_input("Destination New Balance", min_value=0.0, step=0.01)

    if st.button("Predict Single Transaction"):
        result = predict_single_transaction(0, step, transaction_type, amount, oldbalanceOrg, newBalanceOrig, oldbalanceDest, newbalanceDest)
        if result:
            st.write("### Prediction Result")
            st.write("Fraudulent Transaction" if result["is_fraudulent"] else "Legitimate Transaction")
            st.write(f"Fraud Probability: {result['fraud_probability']:.2%}")

# Batch Predictions via CSV
elif input_option == "Upload CSV File":
    st.subheader("Upload a CSV File")
    uploaded_file = st.file_uploader("Choose a file", type=["csv"])

    if uploaded_file is not None:
        try:
            # Load the uploaded CSV
            data = pd.read_csv(uploaded_file)
            st.write("Uploaded Dataset:")
            st.dataframe(data, use_container_width=True)

            # Validate columns
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                st.error(f"Missing columns in uploaded file: {', '.join(missing_columns)}")
                st.stop()

            # Clean and encode transaction types
            data['type'] = data['type'].apply(lambda x: x.strip().upper() if isinstance(x, str) else None)
            if data['type'].isnull().any():
                st.error("The 'type' column contains invalid values.")
                st.stop()

            data['type'] = label_encoder.transform(data['type'])

            # Prepare input data
            input_data = data[required_columns].values.astype(np.float32)

            # Validate input shape
            if input_data.shape[1] != model.n_features_in_:
                raise ValueError(f"Model expects {model.n_features_in_} features, but {input_data.shape[1]} provided.")

            # Make predictions
            predictions = model.predict(input_data)
            probabilities = model.predict_proba(input_data)[:, 1]  # Probability of fraud (class 1)

            # Add predictions to the DataFrame
            data['is_fraudulent'] = predictions
            data['fraud_probability'] = probabilities

            # Display results
            st.write("### Prediction Results")
            st.dataframe(data[['type', 'amount', 'is_fraudulent', 'fraud_probability']])

            # Allow download of results
            st.download_button(
                label="Download Results as CSV",
                data=data.to_csv(index=False),
                file_name="fraud_detection_results.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    # Footer with CSV Format Example
    st.write("### CSV Format Example")
    example_data = pd.DataFrame({
        'index': [0, 0],
        'step': [1, 2],
        'type': ["PAYMENT", "CASH_OUT"],
        'amount': [1000.0, 500.0],
        'oldbalanceOrg': [1500.0, 700.0],
        'newBalanceOrig': [500.0, 200.0],
        'oldbalanceDest': [0.0, 300.0],
        'newbalanceDest': [1000.0, 800.0]
    })
    st.write(example_data)
