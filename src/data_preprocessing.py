import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, RANDOM_STATE, TEST_SIZE, TARGET_VARIABLE

def load_data(file_name):
    """
    Loads data from the raw data directory.
    """
    file_path = os.path.join(RAW_DATA_DIR, file_name)
    if not os.path.exists(file_path):
        # Fallback for local testing if config paths aren't set up on machine
        file_path = file_name 
        
    print(f"Loading data from {file_path}...")
    return pd.read_csv(file_path)

def clean_data(df):
    """
    Performs basic data cleaning:
    1. Drops customerID.
    2. Converts TotalCharges to numeric (handling errors).
    3. Fills missing values (specifically TotalCharges for 0 tenure).
    4. Drops duplicates.
    """
    df = df.copy()
    
    # Drop identifier
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    
    # Fix TotalCharges: Coerce errors to NaN, then fill with 0
    # (Errors usually happen for tenure=0 where TotalCharges is ' ')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(0, inplace=True)
    
    # Remove duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    if len(df) < initial_rows:
        print(f"Removed {initial_rows - len(df)} duplicate rows.")
        
    return df

def handle_outliers(df, columns, factor=3.0):
    """
    Caps outliers using the IQR method.
    Using factor=3.0 (extreme outliers) is safer for Churn data 
    than standard 1.5, to preserve valuable high-spender signals.
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - (factor * IQR)
            upper_bound = Q3 + (factor * IQR)
            
            # Cap values
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
            
    return df

def encode_data(df):
    """
    1. Encodes target variable (Churn) to 0/1.
    2. One-hot encodes categorical variables (drop_first=True to avoid collinearity).
    """
    df = df.copy()
    
    # Encode Target
    if TARGET_VARIABLE in df.columns:
        df[TARGET_VARIABLE] = df[TARGET_VARIABLE].map({'Yes': 1, 'No': 0})
    
    # Identify categorical columns (excluding target)
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # One-Hot Encoding
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    return df

def split_and_save_data(df):
    """
    Splits data into Train/Test sets with stratification and saves them.
    """
    X = df.drop(columns=[TARGET_VARIABLE])
    y = df[TARGET_VARIABLE]
    
    # Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE, 
        stratify=y
    )
    
    # Ensure processed directory exists
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # Save files
    print("Saving processed data...")
    X_train.to_csv(os.path.join(PROCESSED_DATA_DIR, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(PROCESSED_DATA_DIR, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(PROCESSED_DATA_DIR, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(PROCESSED_DATA_DIR, 'y_test.csv'), index=False)
    
    print(f"Data saved to {PROCESSED_DATA_DIR}")
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def run_preprocessing_pipeline(file_name='WA_Fn-UseC_-Telco-Customer-Churn.csv'):
    """
    Orchestrates the entire preprocessing pipeline.
    """
    # 1. Load
    df = load_data(file_name)
    
    # 2. Clean
    df = clean_data(df)
    
    # 3. Outliers (Optional but requested)
    # We apply it to numerical cols. Note: Be cautious with TotalCharges in Churn prediction.
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df = handle_outliers(df, num_cols)
    
    # 4. Encode
    df = encode_data(df)
    
    # 5. Split & Save
    return split_and_save_data(df)

if __name__ == "__main__":
    # Allow running as a standalone script
    run_preprocessing_pipeline()