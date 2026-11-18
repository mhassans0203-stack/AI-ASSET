"""
Loan Application Approval Prediction using K-Nearest Neighbors (KNN)
Dataset: Kaggle Loan Prediction Dataset (altruistdelhite04/loan-prediction-problem-dataset)
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

def run_pipeline():
    print("=" * 70)
    print("STEP 1: DATA LOADING & INITIAL EXPLORATION")
    print("=" * 70)
    
    csv_path = "train.csv"
    if not os.path.exists(csv_path):
        import urllib.request
        url = "https://raw.githubusercontent.com/shrikant-temburwar/Loan-Prediction-Dataset/master/train.csv"
        print(f"Downloading dataset from: {url}")
        urllib.request.urlretrieve(url, csv_path)
    
    df = pd.read_csv(csv_path)
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nFirst 5 rows of dataset:")
    print(df.head())
    
    print("\nDataset Information:")
    print(df.info())
    
    print("\nMissing values before imputation:")
    missing = df.isnull().sum()
    print(missing[missing > 0])
    
    print("\n" + "=" * 70)
    print("STEP 2: DATA PREPROCESSING")
    print("=" * 70)
    
    # Drop Loan_ID as it's an identifier, not a predictive feature
    data = df.drop(columns=['Loan_ID'])
    
    num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    cat_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']
    
    # Compute and fill missing values
    imputation_values = {}
    print("Imputing missing values:")
    for col in num_cols:
        med = data[col].median()
        imputation_values[col] = float(med)
        data[col] = data[col].fillna(med)
        print(f" - Numerical '{col}': filled missing with median = {med}")
        
    for col in cat_cols:
        mode_val = data[col].mode()[0]
        imputation_values[col] = mode_val if isinstance(mode_val, str) else float(mode_val)
        data[col] = data[col].fillna(mode_val)
        print(f" - Categorical '{col}': filled missing with mode = '{mode_val}'")
        
    print(f"\nRemaining missing values after imputation: {data.isnull().sum().sum()}")
    
    # Categorical Encoding
    encoding_dict = {
        'Gender': {'Female': 0, 'Male': 1},
        'Married': {'No': 0, 'Yes': 1},
        'Dependents': {'0': 0, '1': 1, '2': 2, '3+': 3},
        'Education': {'Graduate': 1, 'Not Graduate': 0},
        'Self_Employed': {'No': 0, 'Yes': 1},
        'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
        'Loan_Status': {'N': 0, 'Y': 1}
    }
    
    for col, mapping in encoding_dict.items():
        data[col] = data[col].map(mapping)
        print(f" - Encoded '{col}': {mapping}")
        
    # Ensure float types for Credit_History and Dependents are numeric
    data['Credit_History'] = data['Credit_History'].astype(float)
    data['Dependents'] = data['Dependents'].astype(int)
    
    X = data.drop(columns=['Loan_Status'])
    y = data['Loan_Status']
    feature_names = list(X.columns)
    print(f"\nFeatures ({len(feature_names)}): {feature_names}")
    
    # Train / Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\nTraining set size: {X_train.shape[0]} samples")
    print(f"Testing set size:  {X_test.shape[0]} samples")
    
    # Standardization using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("StandardScaler applied successfully to training and testing sets.")
    
    print("\n" + "=" * 70)
    print("STEP 3: TRAIN DEFAULT KNN CLASSIFIER")
    print("=" * 70)
    
    # Default KNeighborsClassifier (n_neighbors=5, metric='minkowski', p=2)
    knn_default = KNeighborsClassifier()
    knn_default.fit(X_train_scaled, y_train)
    
    y_pred_default = knn_default.predict(X_test_scaled)
    acc_default = accuracy_score(y_test, y_pred_default)
    cm_default = confusion_matrix(y_test, y_pred_default)
    
    print(f"Default KNN (k={knn_default.n_neighbors}) Test Accuracy: {acc_default * 100:.2f}%")
    print("\nDefault KNN Confusion Matrix:")
    print(cm_default)
    print("\nDefault KNN Classification Report:")
    print(classification_report(y_test, y_pred_default, target_names=['Rejected (0)', 'Approved (1)']))
    
    print("=" * 70)
    print("STEP 4: EXPERIMENT WITH DIFFERENT VALUES OF K (e.g., 3, 5, 7, ...)")
    print("=" * 70)
    
    k_experiments = [1, 3, 5, 7, 9, 11, 15, 17, 21, 25]
    k_results = []
    
    print(f"{'k':<5} | {'Train Accuracy':<16} | {'Test Accuracy':<15} | {'CV (5-Fold) Mean':<18}")
    print("-" * 60)
    for k in k_experiments:
        model_k = KNeighborsClassifier(n_neighbors=k)
        model_k.fit(X_train_scaled, y_train)
        train_acc = accuracy_score(y_train, model_k.predict(X_train_scaled))
        test_acc = accuracy_score(y_test, model_k.predict(X_test_scaled))
        cv_scores = cross_val_score(model_k, X_train_scaled, y_train, cv=5, scoring='accuracy')
        cv_mean = cv_scores.mean()
        k_results.append({
            'k': k,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'cv_mean': cv_mean,
            'cv_std': cv_scores.std()
        })
        print(f"{k:<5} | {train_acc*100:>13.2f}% | {test_acc*100:>12.2f}% | {cv_mean*100:>15.2f}% (+/- {cv_scores.std()*100:.1f}%)")
        
    print("\nAnalysis of k values:")
    print("- At small k (e.g. k=1): Model overfits the training data (high train acc, lower test acc).")
    print("- At moderate k (e.g. k=5 to 17): Model stabilizes, capturing true neighborhood patterns.")
    print("- At large k (>25): Smoothing leads to slight underfitting as distant points dilute local signal.")
    
    print("\n" + "=" * 70)
    print("STEP 5: HYPERPARAMETER TUNING (n_neighbors)")
    print("=" * 70)
    
    param_grid = {
        'n_neighbors': list(range(1, 31)),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    
    grid = GridSearchCV(
        KNeighborsClassifier(),
        param_grid={'n_neighbors': list(range(1, 31))},
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid.fit(X_train_scaled, y_train)
    
    best_k = grid.best_params_['n_neighbors']
    print(f"Optimal n_neighbors found via 5-Fold Cross Validation: {best_k}")
    print(f"Best Cross-Validation Score: {grid.best_score_ * 100:.2f}%")
    
    knn_tuned = grid.best_estimator_
    y_pred_tuned = knn_tuned.predict(X_test_scaled)
    acc_tuned = accuracy_score(y_test, y_pred_tuned)
    cm_tuned = confusion_matrix(y_test, y_pred_tuned)
    
    print("\n" + "=" * 70)
    print("STEP 6: MODEL EVALUATION & COMPARISON")
    print("=" * 70)
    
    print(f"{'Metric':<25} | {'Default KNN (k=5)':<20} | {'Tuned KNN (k=' + str(best_k) + ')':<20}")
    print("-" * 70)
    print(f"{'Test Accuracy':<25} | {acc_default*100:>17.2f}% | {acc_tuned*100:>17.2f}%")
    
    tn_def, fp_def, fn_def, tp_def = cm_default.ravel()
    tn_tun, fp_tun, fn_tun, tp_tun = cm_tuned.ravel()
    
    print(f"{'True Negatives (Rejected)':<25} | {tn_def:>19} | {tn_tun:>19}")
    print(f"{'False Positives (Type I)':<25} | {fp_def:>19} | {fp_tun:>19}")
    print(f"{'False Negatives (Type II)':<25} | {fn_def:>19} | {fn_tun:>19}")
    print(f"{'True Positives (Approved)':<25} | {tp_def:>19} | {tp_tun:>19}")
    
    print("\nTuned KNN Classification Report:")
    print(classification_report(y_test, y_pred_tuned, target_names=['Rejected (0)', 'Approved (1)']))
    
    print("=" * 70)
    print("STEP 7: GENERATING & SAVING VISUALIZATIONS")
    print("=" * 70)
    
    os.makedirs("visualizations", exist_ok=True)
    os.makedirs("backend/model", exist_ok=True)
    
    # Plot 1: Side-by-side Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.heatmap(cm_default, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False,
                xticklabels=['Rejected', 'Approved'], yticklabels=['Rejected', 'Approved'],
                annot_kws={'size': 14, 'weight': 'bold'})
    axes[0].set_title(f'Default KNN (k=5)\nAccuracy: {acc_default*100:.2f}%', fontsize=13, pad=10)
    axes[0].set_xlabel('Predicted Label', fontsize=11)
    axes[0].set_ylabel('True Label', fontsize=11)
    
    sns.heatmap(cm_tuned, annot=True, fmt='d', cmap='Greens', ax=axes[1], cbar=False,
                xticklabels=['Rejected', 'Approved'], yticklabels=['Rejected', 'Approved'],
                annot_kws={'size': 14, 'weight': 'bold'})
    axes[1].set_title(f'Tuned KNN (k={best_k})\nAccuracy: {acc_tuned*100:.2f}%', fontsize=13, pad=10)
    axes[1].set_xlabel('Predicted Label', fontsize=11)
    axes[1].set_ylabel('True Label', fontsize=11)
    
    plt.suptitle('Confusion Matrix Comparison: Default KNN vs Tuned KNN', fontsize=15, y=1.02)
    plt.tight_layout()
    cm_plot_path = "visualizations/confusion_matrix_comparison.png"
    plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix plot saved to: {cm_plot_path}")
    
    # Plot 2: K-values vs Accuracy curve
    k_range = list(range(1, 31))
    cv_means = []
    test_scores = []
    
    for k in k_range:
        m = KNeighborsClassifier(n_neighbors=k)
        cv_s = cross_val_score(m, X_train_scaled, y_train, cv=5, scoring='accuracy')
        cv_means.append(cv_s.mean())
        m.fit(X_train_scaled, y_train)
        test_scores.append(accuracy_score(y_test, m.predict(X_test_scaled)))
        
    plt.figure(figsize=(11, 5))
    plt.plot(k_range, cv_means, marker='o', color='#2563eb', label='5-Fold CV Accuracy (Train)', linewidth=2)
    plt.plot(k_range, test_scores, marker='s', color='#10b981', label='Test Set Accuracy', linewidth=2, linestyle='--')
    plt.axvline(x=best_k, color='#ef4444', linestyle=':', linewidth=2, label=f'Optimal k = {best_k}')
    plt.title('KNN Performance vs Number of Neighbors (k)', fontsize=14, pad=12)
    plt.xlabel('Number of Neighbors (k)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xticks(range(1, 31, 2))
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    k_plot_path = "visualizations/knn_k_accuracy_tuning.png"
    plt.savefig(k_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"K-tuning plot saved to: {k_plot_path}")
    
    print("\n" + "=" * 70)
    print("STEP 8: SAVING MODEL & PREPROCESSING METADATA FOR WEB APPLICATION")
    print("=" * 70)
    
    model_export_path = "backend/model/knn_model.joblib"
    scaler_export_path = "backend/model/scaler.joblib"
    meta_export_path = "backend/model/model_meta.json"
    
    joblib.dump(knn_tuned, model_export_path)
    joblib.dump(scaler, scaler_export_path)
    
    # Also save in root for convenience
    joblib.dump(knn_tuned, "knn_model.joblib")
    joblib.dump(scaler, "scaler.joblib")
    
    metadata = {
        "model_name": "K-Nearest Neighbors Classifier (Tuned)",
        "optimal_k": best_k,
        "default_k": 5,
        "accuracy_default": float(acc_default),
        "accuracy_tuned": float(acc_tuned),
        "features": feature_names,
        "feature_count": len(feature_names),
        "imputation_values": imputation_values,
        "encoding_dict": encoding_dict,
        "confusion_matrix_default": cm_default.tolist(),
        "confusion_matrix_tuned": cm_tuned.tolist(),
        "k_experiments": k_results
    }
    
    with open(meta_export_path, "w") as f:
        json.dump(metadata, f, indent=4)
        
    with open("model_meta.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Tuned model saved to: {model_export_path}")
    print(f"Scaler saved to:       {scaler_export_path}")
    print(f"Metadata saved to:     {meta_export_path}")
    print("=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == '__main__':
    run_pipeline()

