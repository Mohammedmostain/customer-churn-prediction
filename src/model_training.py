import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, classification_report
)
from imblearn.pipeline import Pipeline as ImbPipeline
from src.config import RANDOM_STATE

def train_baseline_model(X_train, y_train, class_weight=None):
    """
    Trains a Logistic Regression model as a baseline.
    """
    # Initialize model
    # solver='liblinear' is good for smaller datasets and binary classification
    model = LogisticRegression(
        random_state=RANDOM_STATE, 
        class_weight=class_weight, 
        solver='liblinear',
        max_iter=1000
    )
    
    # Fit model
    model.fit(X_train, y_train)
    
    return model

def evaluate_model_cv(model, X_train, y_train, cv_folds=5):
    """
    Performs 5-fold Stratified Cross-Validation to get robust metrics.
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
    
    # Summarize results
    results = {metric: np.mean(val) for metric, val in scores.items()}
    
    return results

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
    Generates standard evaluation plots: Confusion Matrix and ROC Curve.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title(f'{model_name} - Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    axes[0].set_xticklabels(['Retained', 'Churned'])
    axes[0].set_yticklabels(['Retained', 'Churned'])
    
    # 2. ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    axes[1].plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_prob):.3f}')
    axes[1].plot([0, 1], [0, 1], 'k--', label='Random Chance')
    axes[1].set_title(f'{model_name} - ROC Curve')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].legend(loc='lower right')
    
    plt.tight_layout()
    plt.show()