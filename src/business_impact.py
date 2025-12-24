import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_business_impact(y_true, y_pred, y_prob, 
                              avg_revenue=65, 
                              marketing_cost=10, 
                              offer_cost=50, 
                              probability_acceptance=0.5):
    """
    Calculates the financial impact of the model vs. a baseline (doing nothing).
    
    Parameters:
    - marketing_cost: Cost to contact a customer (email/call).
    - offer_cost: Value of the incentive (discount) given to predicted churners.
    - probability_acceptance: % of churners who stay if given an offer.
    """
    
    # 1. Confusion Matrix Components
    tp = np.sum((y_true == 1) & (y_pred == 1)) # Correctly identified churners
    fp = np.sum((y_true == 0) & (y_pred == 1)) # Loyal customers we annoyed/discounted
    tn = np.sum((y_true == 0) & (y_pred == 0)) # Loyal customers left alone
    fn = np.sum((y_true == 1) & (y_pred == 0)) # Missed churners (lost revenue)
    
    # 2. Baseline Scenario: Do Nothing (Let them churn)
    # Loss = Total Churners * 12 months of Lost Revenue (CLV impact approx)
    # We assume a churner loses us ~12 months of future revenue on average
    clv_loss_baseline = (tp + fn) * avg_revenue * 12 
    total_cost_baseline = clv_loss_baseline
    
    # 3. Model Scenario: Intervene on Predicted Churners (TP + FP)
    
    # Cost A: Intervention Costs (Marketing + Offer)
    # We spend money on everyone predicted as churn (TP + FP)
    n_targeted = tp + fp
    intervention_spend = n_targeted * (marketing_cost + offer_cost)
    
    # Cost B: Remaining Churn Loss
    # We save some TPs, but not all. Some reject the offer.
    # Saved Churners = TP * probability_acceptance
    saved_churners = int(tp * probability_acceptance)
    remaining_churners = (tp - saved_churners) + fn # Unsaved TPs + Missed FNs
    
    clv_loss_model = remaining_churners * avg_revenue * 12
    
    total_cost_model = intervention_spend + clv_loss_model
    
    # 4. Results
    savings = total_cost_baseline - total_cost_model
    roi = (savings / intervention_spend) * 100 if intervention_spend > 0 else 0
    
    return {
        "baseline_loss": total_cost_baseline,
        "model_loss": total_cost_model,
        "savings": savings,
        "roi_percentage": roi,
        "churners_saved": saved_churners,
        "customers_targeted": n_targeted,
        "intervention_spend": intervention_spend
    }

def plot_roi_curve(y_true, y_prob, avg_revenue=65, offer_cost=50):
    """
    Plots Profit/ROI across different classification thresholds.
    Helps decide if we should target top 10%, 20%, or 50% of risk.
    """
    thresholds = np.linspace(0.1, 0.9, 50)
    profits = []
    
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        metrics = calculate_business_impact(y_true, y_pred_t, y_prob, 
                                            avg_revenue=avg_revenue, 
                                            offer_cost=offer_cost)
        profits.append(metrics['savings'])
        
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, profits, marker='o', color='green')
    plt.title("Expected Savings by Probability Threshold")
    plt.xlabel("Churn Probability Threshold")
    plt.ylabel("Estimated Annual Savings ($)")
    plt.axvline(x=0.5, color='gray', linestyle='--', label='Default Threshold (0.5)')
    
    # Highlight max profit
    max_profit_idx = np.argmax(profits)
    plt.plot(thresholds[max_profit_idx], profits[max_profit_idx], 'ro', label='Optimal Threshold')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()