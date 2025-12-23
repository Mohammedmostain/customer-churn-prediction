import pandas as pd
import numpy as np

def create_tenure_features(df):
    """
    Creates features based on customer tenure.
    Rationale: Customer behavior changes drastically over time. Cohorts (groups) 
    capture these lifecycle stages better than a single number for tree models.
    """
    df = df.copy()
    
    # 1. Tenure Groups (Binning)
    # 0-12: High risk (Acquisition phase)
    # 12-24: Retention phase
    # 24-48: Loyal
    # 48+: Very Loyal
    labels = ['0-12 Months', '12-24 Months', '24-48 Months', 'Over 48 Months']
    df['TenureGroup'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72], labels=labels, include_lowest=True)
    
    # 2. Is New Customer (Binary)
    # Explicit flag for the most dangerous period (first 6 months)
    df['IsNewCustomer'] = (df['tenure'] <= 6).astype(int)
    
    return df

def create_service_features(df):
    """
    Aggregates service usage to create 'bundle' features.
    Rationale: Customers with more services are 'stickier' (higher switching costs).
    """
    df = df.copy()
    
    # List of all additional value-added services
    services = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    # 1. Total Services Count
    # We count how many times 'Yes' appears in the service columns
    # We treat 'InternetService' separately as it's the core platform
    df['TotalServices'] = df[services].apply(lambda x: x == 'Yes').sum(axis=1)
    
    # 2. Has Internet and Phone (Dual Play)
    df['HasDualPlay'] = ((df['PhoneService'] == 'Yes') & (df['InternetService'] != 'No')).astype(int)
    
    # 3. Streaming Bundle
    df['HasStreamingBundle'] = ((df['StreamingTV'] == 'Yes') & (df['StreamingMovies'] == 'Yes')).astype(int)
    
    # 4. Security Bundle (High stickiness potential)
    df['HasSecurityBundle'] = ((df['OnlineSecurity'] == 'Yes') & 
                               (df['OnlineBackup'] == 'Yes') & 
                               (df['DeviceProtection'] == 'Yes')).astype(int)
    
    return df

def create_financial_features(df):
    """
    Creates value-based features.
    Rationale: Understanding the economic value of the customer and their price sensitivity.
    """
    df = df.copy()
    
    # 1. Average Monthly Spend (Historical)
    # MonthlyCharges is the *current* price. Total/Tenure gives the *average* price over time.
    # If Avg < Current, they might have had a price hike recently (churn risk).
    # Handle division by zero for tenure=0
    df['AvgMonthlySpend'] = df.apply(lambda x: x['TotalCharges'] / x['tenure'] if x['tenure'] > 0 else x['MonthlyCharges'], axis=1)
    
    # 2. Price Hike Indicator
    # If Current Monthly Charge is significantly higher than Average, it suggests a price increase.
    df['PriceHikeRatio'] = df['MonthlyCharges'] / df['AvgMonthlySpend']
    
    # 3. Auto-Payment Flag
    # Automated payments usually indicate higher retention.
    auto_methods = ['Bank transfer (automatic)', 'Credit card (automatic)']
    df['IsAutoPayment'] = df['PaymentMethod'].isin(auto_methods).astype(int)
    
    return df

def create_interaction_features(df):
    """
    Creates interactions between key categorical variables.
    Rationale: Specific combinations (e.g., Month-to-Month + Electronic Check) are toxic.
    """
    df = df.copy()
    
    # 1. Contract + Payment Method
    df['Contract_Payment'] = df['Contract'] + '_' + df['PaymentMethod']
    
    # 2. Senior + Internet Type
    # Seniors might struggle with Fiber Optic technical issues
    df['Senior_Internet'] = df['SeniorCitizen'].astype(str) + '_' + df['InternetService']
    
    return df

def apply_feature_engineering(df):
    """
    Master pipeline function to apply all engineering steps.
    """
    df = create_tenure_features(df)
    df = create_service_features(df)
    df = create_financial_features(df)
    df = create_interaction_features(df)
    return df