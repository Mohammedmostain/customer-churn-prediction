import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Telco Churn Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    try:
        # Paths based on your structure: models/final_champion_tuned.joblib
        model = joblib.load('models/final_champion_tuned.joblib')
        model_cols = joblib.load('models/model_columns.joblib')
        return model, model_cols
    except FileNotFoundError:
        st.error("⚠️ Model files not found. Please check 'models/' directory.")
        return None, None

@st.cache_data
def load_test_data():
    try:
        # Paths based on your structure: data/processed/X_test.csv
        X_test = pd.read_csv('data/processed/X_test.csv')
        y_test = pd.read_csv('data/processed/y_test.csv')
        return X_test, y_test
    except FileNotFoundError:
        st.warning("⚠️ Test data not found in 'data/processed/'. Analytics tab will be disabled.")
        return None, None

model, model_columns = load_assets()
X_test, y_test = load_test_data()

# --- HELPER FUNCTIONS ---

def preprocess_input(input_dict, model_columns):
    """Prepares a single row of input for prediction."""
    df = pd.DataFrame([input_dict])
    # Define categorical columns (Same as training)
    categorical_cols = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod'
    ]
    # Only get dummies for columns that actually exist in input
    df_encoded = pd.get_dummies(df, columns=[c for c in categorical_cols if c in df.columns])
    
    # Align columns: Add missing with 0, drop extra, enforce order
    df_final = df_encoded.reindex(columns=model_columns, fill_value=0)
    return df_final

def batch_preprocess(df, model_columns):
    """Prepares an uploaded CSV for prediction."""
    # Drop ID columns if they exist to avoid noise
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
        
    # One-Hot Encode
    df_encoded = pd.get_dummies(df)
    
    # Align columns
    df_final = df_encoded.reindex(columns=model_columns, fill_value=0)
    return df_final

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Home", 
    "Single Prediction", 
    "Batch Prediction", 
    "Analytics Dashboard", 
    "Model Performance"
])

st.sidebar.info("v2.0 | Deployment Ready")

# ==========================================
# PAGE 1: HOME
# ==========================================
if page == "Home":
    st.title("📊 Telco Customer Retention Hub")
    st.markdown("""
    ### Welcome to the Churn Command Center
    
    This application bridges the gap between machine learning and business action.
    
    #### What can you do here?
    * **Single Prediction:** specific analysis for a customer on the phone.
    * **Batch Prediction:** Upload a CSV of new customers to get a lead list.
    * **Analytics:** Explore trends in your customer base.
    * **Performance:** Audit the model's technical accuracy.
    """)
    
    # Mock KPIs for the Home Page
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Model Accuracy", "80.4%", "+2.1%")
    with c2:
        st.metric("Revenue Saved (Est)", "$12,450", "Last Month")
    with c3:
        st.metric("Active Model", "Random Forest v2", "Production")

# ==========================================
# PAGE 2: SINGLE PREDICTION
# ==========================================
elif page == "Single Prediction":
    st.title("🔮 Single Customer Analysis")
    
    with st.form("churn_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (Months)", 0, 72, 12)
        with col2:
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            multiline = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        with col3:
            payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            monthly_charges = st.number_input("Monthly Charges ($)", 50.0)
            total_charges = st.number_input("Total Charges ($)", 500.0)
            
        submitted = st.form_submit_button("Analyze Risk")

    if submitted:
        input_data = {
            'gender': gender, 'SeniorCitizen': senior, 'Partner': partner, 'Dependents': dependents,
            'tenure': tenure, 'PhoneService': phone, 'MultipleLines': multiline, 'InternetService': internet,
            'OnlineSecurity': security, 'OnlineBackup': "No", 'DeviceProtection': "No", 'TechSupport': "No",
            'StreamingTV': "No", 'StreamingMovies': "No", 'Contract': contract, 'PaperlessBilling': paperless,
            'PaymentMethod': payment, 'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges
        }
        
        if model:
            X_input = preprocess_input(input_data, model_columns)
            prob = model.predict_proba(X_input)[0][1]
            
            st.divider()
            c1, c2 = st.columns([1, 2])
            with c1:
                if prob < 0.3:
                    st.success(f"Low Risk: {prob:.1%}")
                elif prob < 0.6:
                    st.warning(f"Medium Risk: {prob:.1%}")
                else:
                    st.error(f"High Risk: {prob:.1%}")
            with c2:
                st.write(f"This customer has a **{prob:.1%}** probability of churning.")
                if contract == "Month-to-month": st.write("⚠️ Risk Factor: Month-to-month contract")
                if internet == "Fiber optic": st.write("⚠️ Risk Factor: Fiber Optic service")
        else:
            st.error("Model not loaded.")

# ==========================================
# PAGE 3: BATCH PREDICTION
# ==========================================
elif page == "Batch Prediction":
    st.title("📂 Batch Prediction")
    st.write("Upload a CSV file containing customer data to predict churn for multiple users at once.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file and model:
        df_upload = pd.read_csv(uploaded_file)
        st.write(f"Uploaded data: {df_upload.shape[0]} rows")
        
        if st.button("Run Predictions"):
            with st.spinner("Processing..."):
                try:
                    X_batch = batch_preprocess(df_upload, model_columns)
                    
                    # Predict
                    probs = model.predict_proba(X_batch)[:, 1]
                    preds = model.predict(X_batch)
                    
                    # Append results
                    results_df = df_upload.copy()
                    results_df['Churn_Probability'] = np.round(probs, 4)
                    results_df['Predicted_Churn'] = preds
                    results_df['Risk_Level'] = pd.cut(
                        results_df['Churn_Probability'], 
                        bins=[-0.1, 0.3, 0.7, 1.1], 
                        labels=["Low", "Medium", "High"]
                    )
                    
                    st.success("Predictions Complete!")
                    st.dataframe(results_df.head())
                    
                    # Download
                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download Results CSV",
                        csv,
                        "churn_predictions.csv",
                        "text/csv",
                        key='download-csv'
                    )
                    
                    # Quick Stats
                    st.bar_chart(results_df['Risk_Level'].value_counts())
                    
                except Exception as e:
                    st.error(f"Error processing batch: {e}")

# ==========================================
# PAGE 4: ANALYTICS DASHBOARD
# ==========================================
elif page == "Analytics Dashboard":
    st.title("📈 Customer Analytics Dashboard")
    
    if X_test is not None:
        df_analysis = X_test.copy()
        # Flatten y_test in case it's a DataFrame
        df_analysis['Actual_Churn'] = y_test.values.ravel() if hasattr(y_test, 'values') else y_test
        
        # --- KPI ROW ---
        kpi1, kpi2, kpi3 = st.columns(3)
        churn_rate = df_analysis['Actual_Churn'].mean()
        avg_charge = df_analysis['MonthlyCharges'].mean()
        
        kpi1.metric("Test Set Churn Rate", f"{churn_rate:.1%}")
        kpi2.metric("Avg Monthly Charge", f"${avg_charge:.2f}")
        kpi3.metric("Total Customers Evaluated", f"{len(df_analysis)}")
        
        st.divider()
        
        # --- CHARTS ---
        tab1, tab2 = st.tabs(["Churn Drivers", "Financial Impact"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Churn by Senior Citizen")
                # Group data for Plotly
                chart_data = df_analysis.groupby('SeniorCitizen')['Actual_Churn'].mean().reset_index()
                chart_data['SeniorCitizen'] = chart_data['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
                
                fig = px.bar(chart_data, x='SeniorCitizen', y='Actual_Churn', 
                             title="Churn Rate: Seniors vs Non-Seniors",
                             labels={'Actual_Churn': 'Churn Rate'})
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                st.subheader("Tenure Distribution")
                fig2 = px.histogram(df_analysis, x="tenure", color="Actual_Churn", 
                                    nbins=20, title="Tenure vs Churn (0=Stay, 1=Churn)",
                                    barmode='overlay', opacity=0.7)
                st.plotly_chart(fig2, use_container_width=True)

        with tab2:
            st.subheader("Monthly Charges vs Churn")
            fig3 = px.box(df_analysis, x="Actual_Churn", y="MonthlyCharges", 
                          color="Actual_Churn", 
                          title="Do Higher Charges Lead to Churn?")
            st.plotly_chart(fig3, use_container_width=True)

    else:
        st.warning("Data not available for dashboard.")

# ==========================================
# PAGE 5: MODEL PERFORMANCE
# ==========================================
elif page == "Model Performance":
    st.title("⚙️ Model Performance Specs")
    
    if X_test is not None and y_test is not None and model:
        y_test_flat = y_test.values.ravel() if hasattr(y_test, 'values') else y_test
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        tab1, tab2, tab3 = st.tabs(["Metrics", "Plots", "Features"])
        
        with tab1:
            # Classification Report
            report = classification_report(y_test_flat, y_pred, output_dict=True)
            df_report = pd.DataFrame(report).transpose()
            st.dataframe(df_report.style.highlight_max(axis=0))
            
            c1, c2 = st.columns(2)
            c1.metric("Accuracy", f"{report['accuracy']:.2%}")
            c2.metric("Recall (Churners Found)", f"{report['1']['recall']:.2%}")
            
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test_flat, y_pred)
                fig_cm = plt.figure(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.ylabel('Actual')
                plt.xlabel('Predicted')
                st.pyplot(fig_cm)
            
            with col2:
                st.subheader("ROC Curve")
                fpr, tpr, _ = roc_curve(y_test_flat, y_prob)
                roc_auc = auc(fpr, tpr)
                
                fig_roc = plt.figure(figsize=(5, 4))
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.legend(loc="lower right")
                st.pyplot(fig_roc)

        with tab3:
            st.subheader("Feature Importance")
            if hasattr(model, 'feature_importances_'):
                imp = model.feature_importances_
                feats = X_test.columns
                df_imp = pd.DataFrame({'Feature': feats, 'Importance': imp}).sort_values('Importance', ascending=False).head(10)
                
                fig_imp = px.bar(df_imp, x='Importance', y='Feature', orientation='h', title="Top 10 Drivers of Churn")
                fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_imp)
            else:
                st.info("This model type does not support feature importance.")
    else:
        st.error("Test data or model not loaded properly.")