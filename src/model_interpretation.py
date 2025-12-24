import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

def get_feature_importance(model, feature_names):
    """
    Extracts global feature importance (Gini importance) from the model.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # Fallback for linear models (Logistic Regression)
        importances = np.abs(model.coef_[0])
    else:
        raise ValueError("Model does not have 'feature_importances_' or 'coef_' attribute.")

    feature_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    return feature_imp

def plot_feature_importance(importance_df, top_n=10):
    """
    Plots the top N most important features.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(top_n), palette='viridis')
    plt.title(f'Top {top_n} Feature Importance (Gini)')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

def calculate_shap_values(model, X_sample):
    """
    Calculates SHAP values using TreeExplainer.
    Returns the explainer and the shap_values object.
    """
    # TreeExplainer is best for Random Forest / XGBoost
    explainer = shap.TreeExplainer(model)
    
    # Check if X_sample is a DataFrame and extract values if needed to be safe
    # (Though TreeExplainer often handles DataFrames well, explicit checks help)
    shap_values = explainer(X_sample)
    
    return explainer, shap_values

def plot_shap_summary(shap_values, X_sample):
    """
    Plots the standard SHAP summary plot (beeswarm), handling DataFrames correctly.
    """
    # 1. Handle SHAP values
    if isinstance(shap_values, list):
        shap_vals_to_plot = shap_values[1] # Binary classification positive class
    elif hasattr(shap_values, "values"):
        # For newer SHAP Explanation objects
        # If 3D (samples, features, classes), take positive class
        if len(shap_values.shape) == 3:
             shap_vals_to_plot = shap_values.values[:, :, 1]
        else:
             shap_vals_to_plot = shap_values.values
    else:
        shap_vals_to_plot = shap_values

    # 2. Handle X Features (CRITICAL FIX FOR TYPE ERROR)
    if hasattr(X_sample, 'columns'):
        # Convert Pandas Index to a simple list to avoid "integer scalar array" errors
        feature_names_list = X_sample.columns.tolist()
        # Convert DataFrame to numpy array
        X_vals_matrix = X_sample.values
    else:
        # Fallback if it's already numpy
        feature_names_list = None
        X_vals_matrix = X_sample

    plt.figure() 
    shap.summary_plot(
        shap_vals_to_plot, 
        X_vals_matrix,       # Pass raw numpy array
        feature_names=feature_names_list, # Pass explicit list of names
        show=False
    )
    plt.title("SHAP Summary Plot")
    plt.tight_layout()
    plt.show()

def plot_shap_dependence(shap_values, X_sample, feature_name, interaction_feature=None):
    """
    Robust SHAP dependence plotter.
    Handles:
    1. Binary classification lists (extracts the positive class).
    2. SHAP Explanation objects (extracts .values).
    3. Pandas DataFrames (converts to Numpy).
    """
    
    # --- STEP 1: Process SHAP Values ---
    # Case A: If it's a list (common in binary classification: [Class0, Class1])
    if isinstance(shap_values, list):
        # We assume index 1 is the positive class (Churn)
        shap_vals_matrix = shap_values[1]
    # Case B: If it's a newer SHAP Explanation object
    elif hasattr(shap_values, "values"):
        # If the explanation object has 2 dimensions (samples, features) it's fine.
        # If it has 3 (samples, features, classes), slice it.
        if len(shap_values.shape) == 3:
             shap_vals_matrix = shap_values.values[:, :, 1] # Take positive class
        else:
             shap_vals_matrix = shap_values.values
    # Case C: It's already a numpy array
    else:
        shap_vals_matrix = shap_values

    # --- STEP 2: Process X Features ---
    if hasattr(X_sample, "values"):
        X_vals_matrix = X_sample.values
        feature_names_list = X_sample.columns.tolist()
    else:
        X_vals_matrix = X_sample
        feature_names_list = None

    # --- STEP 3: Safety Checks ---
    # 1. Check if feature exists
    if feature_names_list and feature_name not in feature_names_list:
        raise ValueError(f"Feature '{feature_name}' not found in X_sample columns.")

    # 2. Check dimensions match
    if shap_vals_matrix.shape[0] != X_vals_matrix.shape[0]:
        raise ValueError(
            f"Dimension Mismatch! \n"
            f"SHAP rows: {shap_vals_matrix.shape[0]} \n"
            f"X rows: {X_vals_matrix.shape[0]} \n"
            "Ensure 'shap_values' and 'X_sample' come from the exact same dataset rows."
        )

    # --- STEP 4: Plot ---
    shap.dependence_plot(
        feature_name, 
        shap_vals_matrix, 
        X_vals_matrix, 
        feature_names=feature_names_list,
        interaction_index=interaction_feature,
        show=False
    )
    plt.title(f'SHAP Dependence: {feature_name}')
    plt.tight_layout()
    plt.show()

def explain_single_prediction(explainer, single_row_df):
    """
    Generates a force plot for a single observation.
    Robustly handles binary classification dimensions.
    """
    shap.initjs()
    
    # 1. Calculate SHAP values (returns an Explanation object)
    shap_object = explainer(single_row_df)
    
    # 2. Extract Base Value and SHAP Values based on dimensions
    # The shape is usually (n_samples, n_features) or (n_samples, n_features, n_classes)
    
    # Case A: Binary Classification with 2 outputs (Random Forest often does this)
    if len(shap_object.values.shape) == 3:
        # We want the Positive Class (Index 1 -> Churn)
        # base_values shape: (1, 2) -> we take [0, 1]
        base_value = shap_object.base_values[0, 1]
        # values shape: (1, features, 2) -> we take [0, :, 1]
        shap_values = shap_object.values[0, :, 1]
    
    # Case B: Regression or Binary with 1 output (XGBoost often does this)
    else:
        base_value = shap_object.base_values[0]
        shap_values = shap_object.values[0]

    # 3. Generate Force Plot with EXPLICIT arguments to satisfy v0.20+ requirements
    # Signature: force(base_value, shap_values, features, feature_names)
    return shap.plots.force(
        base_value,
        shap_values,
        single_row_df.iloc[0].values,
        feature_names=single_row_df.columns.tolist(),
        matplotlib=True  # Use Matplotlib for a static, robust plot image
    )