# =============================================
# Model Training Script
# Tourism Package Prediction - MLOps Pipeline
# =============================================
# This script loads train/test data from GitHub Actions
# workflow artifacts, tunes multiple models using
# GridSearchCV, logs all parameters, evaluates
# performance, and saves the best model locally.
# The pipeline will commit the model to the repository.

# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
# Model algorithms (as per rubric: Decision Tree, Bagging, Random Forest,
# AdaBoost, Gradient Boosting, XGBoost)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
# Hyperparameter tuning
from sklearn.model_selection import GridSearchCV
# Evaluation metrics
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report)
# Model serialization
import joblib
# File system
import os

# =============================================
# LOAD TRAIN/TEST DATA FROM WORKFLOW ARTIFACTS
# =============================================
# These CSV files are downloaded by the GitHub Actions
# download-artifact step into the current working directory.

Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").values.ravel()
ytest = pd.read_csv("ytest.csv").values.ravel()

print(f" Data loaded from workflow artifacts.")
print(f"   Train: {Xtrain.shape}, Test: {Xtest.shape}")
print(f"   Features: {list(Xtrain.columns)}")

# =============================================
# MODEL DEFINITION & HYPERPARAMETER GRIDS
# =============================================
# Rubric: Decision Tree, Bagging, Random Forest,
# AdaBoost, Gradient Boosting, and XGBoost

pipelines = {
    'DecisionTree': (
        make_pipeline(StandardScaler(), DecisionTreeClassifier(random_state=42)),
        {
            'decisiontreeclassifier__max_depth': [5, 10, 15],
            'decisiontreeclassifier__min_samples_split': [2, 5, 10],
            'decisiontreeclassifier__criterion': ['gini', 'entropy']
        }
    ),
    'Bagging': (
        make_pipeline(StandardScaler(), BaggingClassifier(random_state=42)),
        {
            'baggingclassifier__n_estimators': [50, 100],
            'baggingclassifier__max_samples': [0.7, 1.0],
        }
    ),
    'RandomForest': (
        make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42)),
        {
            'randomforestclassifier__n_estimators': [100, 150],
            'randomforestclassifier__max_depth': [10, 15, 20],
            'randomforestclassifier__min_samples_split': [2, 5]
        }
    ),
    'AdaBoost': (
        make_pipeline(StandardScaler(), AdaBoostClassifier(random_state=42, algorithm='SAMME')),
        {
            'adaboostclassifier__n_estimators': [50, 100],
            'adaboostclassifier__learning_rate': [0.05, 0.1, 0.5]
        }
    ),
    'GradientBoosting': (
        make_pipeline(StandardScaler(), GradientBoostingClassifier(random_state=42)),
        {
            'gradientboostingclassifier__n_estimators': [100, 150],
            'gradientboostingclassifier__max_depth': [5, 7],
            'gradientboostingclassifier__learning_rate': [0.05, 0.1],
            'gradientboostingclassifier__subsample': [0.8, 1.0]
        }
    ),
}

# =============================================
# MODEL TUNING & PARAMETER LOGGING
# =============================================

best_overall_score = 0.0
best_overall_model = None
best_overall_name = ""

for name, (pipeline, param_grid) in pipelines.items():
    print(f"\n🔧 Tuning {name}...")
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=3, n_jobs=-1, scoring='accuracy', verbose=1
    )
    grid_search.fit(Xtrain, ytrain)

    # Log all tuned parameters & CV scores
    results = grid_search.cv_results_
    print(f"\n  All tuned parameter combinations for {name}:")
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        print(f"      Params: {param_set} | Mean CV Accuracy: {mean_score:.4f}")

    print(f"\n   {name} Best CV Accuracy: {grid_search.best_score_:.4f}")
    print(f"   {name} Best Params: {grid_search.best_params_}")

    if grid_search.best_score_ > best_overall_score:
        best_overall_score = grid_search.best_score_
        best_overall_model = grid_search.best_estimator_
        best_overall_name = name

# =============================================
# BEST MODEL SUMMARY
# =============================================

best_model = best_overall_model
print(f"\n{'='*60}")
print(f" BEST MODEL SELECTED: {best_overall_name}")
print(f"{'='*60}")
print(f"   Best CV Accuracy: {best_overall_score:.4f}")

# =============================================
# MODEL EVALUATION
# =============================================

y_pred_train = best_model.predict(Xtrain)
y_pred_test = best_model.predict(Xtest)
y_prob_test = best_model.predict_proba(Xtest)[:, 1]

train_accuracy = accuracy_score(ytrain, y_pred_train)
test_accuracy = accuracy_score(ytest, y_pred_test)
test_precision = precision_score(ytest, y_pred_test)
test_recall = recall_score(ytest, y_pred_test)
test_f1 = f1_score(ytest, y_pred_test)
test_auc = roc_auc_score(ytest, y_prob_test)

print(f"\n Model Evaluation Metrics ({best_overall_name}):")
print(f"   Train Accuracy:  {train_accuracy:.4f}")
print(f"   Test Accuracy:   {test_accuracy:.4f}")
print(f"   Test Precision:  {test_precision:.4f}")
print(f"   Test Recall:     {test_recall:.4f}")
print(f"   Test F1-Score:   {test_f1:.4f}")
print(f"   Test AUC-ROC:    {test_auc:.4f}")
print(f"\n  Classification Report:")
print(classification_report(ytest, y_pred_test,
                            target_names=["Not Purchased (0)", "Purchased (1)"]))

# =============================================
# SAVE BEST MODEL LOCALLY
# =============================================
# The pipeline will commit this file to the repository
# so the Streamlit app can load it directly.

model_dir = os.path.join("tourism_package_prediction", "model")
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "best_tourism_model.joblib")
joblib.dump(best_model, model_path)

print(f"\n Best model saved to: {model_path}")
print("   The pipeline will commit this model to the repository.")
