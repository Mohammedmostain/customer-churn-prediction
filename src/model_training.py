import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve
)
from src.config import RANDOM_STATE, MODELS_DIR

# ==========================================
# 1. TRAINING FUNCTIONS
# ==========================================

def train_baseline_model(X_train, y_train, class_weight=None):
    """
    Trains a Logistic Regression model as a baseline.
    """
    model = LogisticRegression(
        random_state=RANDOM_STATE, 
        class_weight=class_weight, 
        solver='liblinear',
        max_iter=1000
    )
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, class_weight='balanced'):
    """
    Trains a Random Forest model with robust initial parameters.
    """
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features='sqrt',
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    return rf_model

def train_xgboost(X_train, y_train, scale_pos_weight=1):
    """
    Trains an XGBoost model.
    """
    xgb_model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    return xgb_model

# ==========================================
# 2. EVALUATION FUNCTIONS
# ==========================================

def evaluate_model_cv(model, X_train, y_train, cv_folds=5):
    """
    Performs 5-fold Stratified Cross-Validation.
    """
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
    return {metric: np.mean(val) for metric, val in scores.items()}

def evaluate_model_test(model, X_test, y_test):
    """
    Evaluates the trained model on the held-out test set.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob)
    }
    return metrics, y_pred, y_prob

def plot_model_performance(y_test, y_pred, y_prob, model_name="Model"):
    """
    Generates Confusion Matrix and ROC Curve plots.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title(f'{model_name} - Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    
    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    axes[1].plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_prob):.3f}')
    axes[1].plot([0, 1], [0, 1], 'k--', label='Random Chance')
    axes[1].set_title(f'{model_name} - ROC Curve')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()

def get_feature_importance(model, feature_names):
    """
    Extracts feature importance from Random Forest or XGBoost.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    return pd.DataFrame({
        'Feature': [feature_names[i] for i in indices],
        'Importance': importances[indices]
    })

# ==========================================
# 3. SAVING FUNCTIONS
# ==========================================

def save_model(model, filename):
    """
    Saves the trained model to the models/ directory.
    """
    # Ensure directory exists
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    file_path = os.path.join(MODELS_DIR, filename)
    joblib.dump(model, file_path)
    print(f"Model saved successfully to: {file_path}")