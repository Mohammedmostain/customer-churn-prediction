import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve, auc
)

def get_comprehensive_metrics(y_true, y_pred, y_prob):
    """
    Returns a dictionary of all key metrics.
    """
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred),
        'AUC-ROC': roc_auc_score(y_true, y_prob)
    }

def plot_confusion_matrix_custom(y_true, y_pred, title="Confusion Matrix"):
    """
    Plots a professional heatmap of the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = [f"{value:0.0f}" for value in cm.flatten()]
    group_percentages = [f"{value:.2%}" for value in cm.flatten()/np.sum(cm)]
    
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=False)
    plt.title(title, fontsize=14)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

def plot_roc_pr_curves(y_true, y_prob, model_name="Model"):
    """
    Plots both ROC and Precision-Recall curves side-by-side.
    PR Curves are crucial for imbalanced datasets like Churn.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0].set_title(f'{model_name} - ROC Curve')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].legend(loc="lower right")
    
    # 2. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    
    axes[1].plot(recall, precision, color='green', lw=2, label=f'PR AUC = {pr_auc:.2f}')
    axes[1].set_title(f'{model_name} - Precision-Recall Curve')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].legend(loc="lower left")
    
    plt.tight_layout()
    plt.show()

def find_optimal_threshold(y_true, y_prob):
    """
    Finds the threshold that maximizes the F1-Score.
    Standard is 0.5, but for Churn, we might lower it to catch more churners.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    
    # Locate best threshold
    ix = np.argmax(f1_scores)
    best_thresh = thresholds[ix]
    best_f1 = f1_scores[ix]
    
    print(f"Best Threshold: {best_thresh:.4f}")
    print(f"Best F1 Score: {best_f1:.4f}")
    
    # Plotting the trade-off
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precision[:-1], label='Precision', linestyle='--')
    plt.plot(thresholds, recall[:-1], label='Recall', linestyle='--')
    plt.plot(thresholds, f1_scores[:-1], label='F1 Score', linewidth=2)
    plt.axvline(best_thresh, color='red', linestyle=':', label=f'Optimal ({best_thresh:.2f})')
    plt.title('Threshold Optimization: Precision vs Recall vs F1')
    plt.xlabel('Probability Threshold')
    plt.ylabel('Score')
    plt.legend()
    plt.show()
    
    return best_thresh

def analyze_misclassifications(model, X_test, y_test, y_pred):
    """
    Returns a DataFrame of False Negatives (Churners we missed).
    This helps in qualitative error analysis.
    """
    X_test_copy = X_test.copy()
    X_test_copy['Actual'] = y_test
    X_test_copy['Predicted'] = y_pred
    
    # False Negatives: Actual = 1 (Yes), Predicted = 0 (No)
    fn_df = X_test_copy[(X_test_copy['Actual'] == 1) & (X_test_copy['Predicted'] == 0)]
    
    return fn_df