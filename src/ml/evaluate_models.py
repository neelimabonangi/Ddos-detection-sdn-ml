#!/usr/bin/env python3
"""
ML Model Evaluation for DDoS Detection in SDN

This script evaluates the performance of trained machine learning models
for DDoS attack detection in Software-Defined Networks (SDN).

It loads the trained models, evaluates them on the test dataset,
and generates detailed performance reports and visualizations.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    classification_report
)
import joblib

def load_models():
    """
    Load trained ML models.
    
    Returns:
        dict: Dictionary of loaded models
    """
    models_dir = 'models'
    if not os.path.exists(models_dir):
        print(f"Error: Models directory '{models_dir}' not found.")
        print("Please run train_models.py first.")
        sys.exit(1)
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl') and f != 'scaler.pkl']
    
    if not model_files:
        print("Error: No model files found in the models directory.")
        print("Please run train_models.py first.")
        sys.exit(1)
    
    models = {}
    for model_file in model_files:
        model_name = model_file.replace('.pkl', '').replace('_', ' ').title()
        model_path = os.path.join(models_dir, model_file)
        models[model_name] = joblib.load(model_path)
        print(f"Loaded model: {model_name}")
    
    return models

def load_test_data():
    """
    Load and preprocess test data.
    
    Returns:
        tuple: X_test, y_test
    """
    # Load the dataset
    data_path = 'data/ddos_dataset.csv'
    if not os.path.exists(data_path):
        print(f"Error: Dataset file '{data_path}' not found.")
        print("Please run train_models.py first to generate the dataset.")
        sys.exit(1)
    
    data = pd.read_csv(data_path)
    
    # Split features and target
    X = data.drop('label', axis=1)
    y = data['label']
    
    # Load the scaler
    scaler_path = 'models/scaler.pkl'
    if not os.path.exists(scaler_path):
        print(f"Error: Scaler file '{scaler_path}' not found.")
        print("Please run train_models.py first.")
        sys.exit(1)
    
    scaler = joblib.load(scaler_path)
    
    # Use a fixed test set (last 20% of data)
    test_size = int(len(data) * 0.2)
    X_test = X.iloc[-test_size:]
    y_test = y.iloc[-test_size:]
    
    # Scale the features
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Test data loaded: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
    return X_test_scaled, y_test, X_test.columns.tolist()

def evaluate_models(models, X_test, y_test):
    """
    Evaluate multiple models on the test dataset.
    
    Args:
        models (dict): Dictionary of trained models
        X_test: Test features
        y_test: Test labels
        
    Returns:
        dict: Dictionary of evaluation results
    """
    results = {}
    
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        
        # Measure inference time
        start_time = time.time()
        y_pred = model.predict(X_test)
        inference_time = time.time() - start_time
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Calculate ROC curve and AUC
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            # For models without predict_proba, use decision_function if available
            if hasattr(model, "decision_function"):
                y_prob = model.decision_function(X_test)
            else:
                # Fall back to binary predictions
                y_prob = y_pred
        
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        # Calculate precision-recall curve
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall_curve, precision_curve)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Store results
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'confusion_matrix': cm,
            'classification_report': report,
            'inference_time': inference_time,
            'fpr': fpr,
            'tpr': tpr,
            'precision_curve': precision_curve,
            'recall_curve': recall_curve
        }
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"PR AUC: {pr_auc:.4f}")
        print(f"Inference Time: {inference_time:.4f} seconds")
    
    return results

def plot_results(results):
    """
    Generate and save visualizations of model performance.
    
    Args:
        results (dict): Dictionary of evaluation results
    """
    os.makedirs('results', exist_ok=True)
    
    # Prepare data for bar charts
    models = list(results.keys())
    accuracy = [results[m]['accuracy'] for m in models]
    precision = [results[m]['precision'] for m in models]
    recall = [results[m]['recall'] for m in models]
    f1 = [results[m]['f1'] for m in models]
    roc_auc = [results[m]['roc_auc'] for m in models]
    inference_time = [results[m]['inference_time'] for m in models]
    
    # Plot accuracy, precision, recall, f1
    plt.figure(figsize=(12, 8))
    x = np.arange(len(models))
    width = 0.2
    
    plt.bar(x - width*1.5, accuracy, width, label='Accuracy')
    plt.bar(x - width/2, precision, width, label='Precision')
    plt.bar(x + width/2, recall, width, label='Recall')
    plt.bar(x + width*1.5, f1, width, label='F1 Score')
    
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x, models, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/model_performance.png')
    
    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    for model in models:
        plt.plot(results[model]['fpr'], results[model]['tpr'], 
                 label=f'{model} (AUC = {results[model]["roc_auc"]:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.savefig('results/roc_curves.png')
    
    # Plot Precision-Recall curves
    plt.figure(figsize=(10, 8))
    for model in models:
        plt.plot(results[model]['recall_curve'], results[model]['precision_curve'], 
                 label=f'{model} (AUC = {results[model]["pr_auc"]:.4f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc='lower left')
    plt.savefig('results/pr_curves.png')
    
    # Plot inference time
    plt.figure(figsize=(10, 6))
    plt.bar(models, inference_time, color='skyblue')
    plt.xlabel('Models')
    plt.ylabel('Inference Time (seconds)')
    plt.title('Model Inference Time Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/inference_time.png')
    
    # Plot confusion matrices
    for model in models:
        plt.figure(figsize=(8, 6))
        cm = results[model]['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Attack'],
                   yticklabels=['Normal', 'Attack'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'{model} Confusion Matrix')
        plt.tight_layout()
        plt.savefig(f'results/confusion_matrix_{model.lower().replace(" ", "_")}.png')
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'Model': models,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC': roc_auc,
        'Inference Time (s)': inference_time
    })
    
    results_df.to_csv('results/model_evaluation.csv', index=False)
    
    print("\nResults visualizations saved to 'results/' directory")

def analyze_feature_importance(models, feature_names):
    """
    Analyze and visualize feature importance for tree-based models.
    
    Args:
        models (dict): Dictionary of trained models
        feature_names (list): List of feature names
    """
    # Check which models have feature_importances_ attribute
    for name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            print(f"\nAnalyzing feature importance for {name}...")
            
            # Get feature importances
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Plot feature importances
            plt.figure(figsize=(12, 8))
            plt.title(f'Feature Importances ({name})')
            plt.bar(range(len(indices)), importances[indices], align='center')
            plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
            plt.tight_layout()
            plt.savefig(f'results/feature_importances_{name.lower().replace(" ", "_")}.png')
            
            # Save feature importances to CSV
            importance_df = pd.DataFrame({
                'Feature': [feature_names[i] for i in indices],
                'Importance': importances[indices]
            })
            importance_df.to_csv(f'results/feature_importances_{name.lower().replace(" ", "_")}.csv', index=False)
            
            # Print top 10 features
            print("Top 10 most important features:")
            for i in range(min(10, len(indices))):
                print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

def main():
    """Main function to evaluate models."""
    print("Starting DDoS Detection ML Model Evaluation")
    
    # Load models
    models = load_models()
    
    # Load test data
    X_test, y_test, feature_names = load_test_data()
    
    # Evaluate models
    results = evaluate_models(models, X_test, y_test)
    
    # Plot results
    plot_results(results)
    
    # Analyze feature importance for applicable models
    analyze_feature_importance(models, feature_names)
    
    print("\nEvaluation complete. Results saved in 'results/' directory.")

if __name__ == "__main__":
    main()