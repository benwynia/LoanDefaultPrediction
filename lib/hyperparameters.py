#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def optimize(model, param_grid, method, X, y, cv=5, scoring='accuracy', n_iter=10, verbose=2, n_jobs=-1, random_state=42):
    start_time = time.time()
    
    """
    DOCUMENTATION: information on defining a param_grid to use with GridSearchCV is found here: 
    -- https://scikit-learn.org/stable/modules/grid_search.html
    -- Copied param values from: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html
    """

    if method == "GridSearchCV":
        print("Starting GridSearchCV...")
        grid_search = GridSearchCV(model, param_grid, scoring=scoring, cv=cv, n_jobs=n_jobs, verbose=verbose)
        grid_search.fit(X, y)
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        best_estimator = grid_search.best_estimator_
    
    elif method == "RandomizedSearchCV":
        print("Starting RandomizedSearchCV...")
        random_search = RandomizedSearchCV(model, param_grid, n_iter=n_iter, scoring=scoring, cv=cv, n_jobs=n_jobs, verbose=verbose, random_state=random_state)
        random_search.fit(X, y)
        best_params = random_search.best_params_
        best_score = random_search.best_score_
        best_estimator = random_search.best_estimator_
    
    else:
        return 'Invalid method provided to function'

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"Optimization completed. Total runtime: {elapsed_time:.2f} seconds")
    print(f"Best parameters: {best_params}")
    
    return best_params, best_score, best_estimator, elapsed_time


# In[2]:


if __name__ == '__main__':
    print("Hello World")

