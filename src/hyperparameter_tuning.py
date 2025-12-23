import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from src.config import RANDOM_STATE

def perform_hyperparameter_tuning(X_train, y_train, method='random'):
    """
    Performs hyperparameter tuning for Random Forest.
    
    Args:
        method (str): 'random' for RandomizedSearchCV (faster), 'grid' for GridSearchCV (thorough).
    """
    print(f"Starting {method.capitalize()} Search...")
    start_time = time.time()
    
    # 1. Define the Parameter Grid
    # These ranges cover the most important "knobs" for Random Forest
    param_dist = {
        'n_estimators': [100, 200, 300, 500],        # Number of trees
        'max_depth': [10, 15, 20, 25, None],         # Max depth of tree (None = full depth)
        'min_samples_split': [2, 5, 10],             # Min samples to split a node
        'min_samples_leaf': [1, 2, 4, 8],            # Min samples in a leaf (controls smoothing)
        'max_features': ['sqrt', 'log2'],            # Features to consider per split
        'class_weight': ['balanced', 'balanced_subsample'] # Handle imbalance
    }
    
    # 2. Setup the Base Model
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    
    # 3. Setup Cross-Validation
    # Stratified ensures each fold has 27% churners, just like the real data
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    
    # 4. Configure the Search Strategy
    if method == 'grid':
        search = GridSearchCV(
            estimator=rf,
            param_grid=param_dist,
            scoring='f1',             # Optimize for F1 Score
            cv=cv,
            n_jobs=-1,                # Use all CPU cores
            verbose=2
        )
    else:
        # RandomizedSearch is usually better for Capstones (95% performance in 5% time)
        search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_dist,
            n_iter=50,                # Try 50 random combinations
            scoring='f1',             # Optimize for F1 Score
            cv=cv,
            n_jobs=-1,
            verbose=2,
            random_state=RANDOM_STATE
        )
        
    # 5. Execute Search
    search.fit(X_train, y_train)
    
    elapsed_time = (time.time() - start_time) / 60
    print(f"Tuning Completed in {elapsed_time:.1f} minutes.")
    print(f"Best F1 Score: {search.best_score_:.4f}")
    print(f"Best Params: {search.best_params_}")
    
    return search.best_estimator_, search.cv_results_