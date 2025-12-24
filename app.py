import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOAD ASSETS (Cached for speed) ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('models/final_champion_tuned.joblib')
        model_columns = joblib.load('models/model_columns.joblib')
        return model, model_columns
    except FileNotFoundError:
        st.error("Model files not found. Please ensure 'final_champion_tuned.joblib' and 'model_columns.joblib' are in the 'models/' folder.")
        return None, None

model, model_columns = load_assets()

# --- PREPROCESSING FUNCTION ---
def preprocess_input(input_dict, model_columns):
    """
    Converts user input dictionary into the exact One-Hot Encoded DataFrame 
    the model expects.
    """
    # Create a DataFrame from the single input row
    df = pd.DataFrame([input_dict])
    
    # Define categorical columns that need dummy encoding
    # (These must match the logic used in your training phase)
    categorical_cols = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod'
    ]
    
    # One-Hot Encode
    df_encoded = pd.get_dummies(df, columns=categorical_cols)
    
    # ALIGNMENT STEP: Ensure all columns from training exist here
    # 1. Add missing columns with 0
    for col in model_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
            
    # 2. Reorder columns to match model exactly
    df_final = df_encoded[model_columns]
    
    return df_final

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predict Churn"])

# --- PAGE 1: HOME ---
if page == "Home":
    st.title("📊 Telco Customer Retention Dashboard")
    st.markdown("""
    Welcome to the **Customer Churn Prediction System**.
    
    This tool uses advanced Machine Learning (Random Forest) to assess the risk of customer attrition.
    
    ### 🎯 Objectives
    * **Identify** high-risk customers before they cancel.
    * **Understand** the key drivers of dissatisfaction.
    * **Act** by recommending targeted retention strategies.
    
    ### 📈 Model Performance
    * **Accuracy:** ~80%
    * **Key Drivers:** Contract Type, Tenure, Monthly Charges.
    
    Navigate to the **Predict Churn** tab to analyze a specific customer.
    """)
    
    # Optional: You could load and display the ROI plot image here if saved
    # st.image("reports/figures/roi_curve.png")

# --- PAGE 2: PREDICTION ---
elif page == "Predict Churn":
    st.title("🔮 Single Customer Analysis")
    st.write("Enter customer details below to generate a risk profile.")
    
    # --- INPUT FORM ---
    with st.form("churn_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Demographics")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (Months)", 0, 72, 12)

        with col2:
            st.subheader("Services")
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            # Logic: If no phone, multiple lines is usually 'No phone service'
            multiline = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            
        with col3:
            st.subheader("Account & Billing")
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            payment = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=500.0)

        # Hidden inputs/Defaults for things we might not have in the form but need
        # For simplicity, we just use the ones above.
        
        submitted = st.form_submit_button("Analyze Risk")

    # --- PREDICTION LOGIC ---
    if submitted:
        # 1. Collect Input
        input_data = {
            'gender': gender,
            'SeniorCitizen': senior,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone,
            'MultipleLines': multiline,
            'InternetService': internet,
            'OnlineSecurity': security,
            'OnlineBackup': backup,
            'DeviceProtection': "No", # Simplified for UI, defaulting others
            'TechSupport': "No",
            'StreamingTV': "No",
            'StreamingMovies': "No",
            'Contract': contract,
            'PaperlessBilling': paperless,
            'PaymentMethod': payment,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }
        
        # 2. Process
        try:
            X_input = preprocess_input(input_data, model_columns)
            
            # 3. Predict
            prediction = model.predict(X_input)[0]
            probability = model.predict_proba(X_input)[0][1]
            
            # 4. Display Results
            st.divider()
            st.subheader("Risk Analysis Result")
            
            c1, c2 = st.columns([1, 2])
            
            with c1:
                # Gauge-like display
                if probability < 0.3:
                    st.success(f"Low Risk: {probability:.1%}")
                    risk_color = "green"
                elif probability < 0.6:
                    st.warning(f"Medium Risk: {probability:.1%}")
                    risk_color = "orange"
                else:
                    st.error(f"High Risk: {probability:.1%}")
                    risk_color = "red"
            
            with c2:
                st.info(f"The model predicts this customer **{'WILL CHURN' if prediction == 1 else 'WILL STAY'}**.")
                
                # Simple Rules-Based "Why" (Approximation of SHAP for speed)
                st.write("**Top Risk Factors Detected:**")
                reasons = []
                if contract == "Month-to-month":
                    reasons.append("⚠️ **Contract:** Month-to-month contracts have the highest churn rate.")
                if internet == "Fiber optic":
                    reasons.append("⚠️ **Internet:** Fiber optic users often face technical issues driving churn.")
                if payment == "Electronic check":
                    reasons.append("⚠️ **Payment:** Electronic check users are historically more likely to leave.")
                if tenure < 12:
                    reasons.append("⚠️ **Tenure:** New customers (< 1 year) are in the 'danger zone'.")
                
                if not reasons:
                    st.write("✅ Customer appears stable. No major risk flags detected.")
                else:
                    for r in reasons:
                        st.write(r)

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")