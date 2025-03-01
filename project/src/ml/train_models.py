#!/usr/bin/env python3
"""
ML Model Training for DDoS Detection in SDN

This script trains multiple machine learning models for DDoS attack detection
in Software-Defined Networks (SDN) using a dataset with 23 features.

Models implemented:
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Decision Tree
- Naive Bayes

The script performs data preprocessing, model training, evaluation, and saves
the trained models for use with the POX controller.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Create directories if they don't exist
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Constants
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.25  # 25% of the remaining 80% = 20% of total

def load_data():
    """
    Load the dataset from CSV file or create a synthetic dataset if file doesn't exist.
    
    Returns:
        pandas.DataFrame: The loaded or created dataset
    """
    data_path = 'data/ddos_dataset.csv'
    
    # Check if the dataset exists
    if os.path.exists(data_path):
        print(f"Loading dataset from {data_path}")
        return pd.read_csv(data_path)
    
    # If dataset doesn't exist, create a synthetic one for demonstration
    print("Dataset not found. Creating synthetic dataset for demonstration...")
    
    # Number of samples
    n_samples = 104345
    
    # Create feature names based on the image
    feature_names = [
        'dt', 'switch', 'src', 'dst', 'pkt_count', 'byte_count', 'duration',
        'duration_nsec', 'total_duration', 'flows', 'pkt_rate', 'pair_flow',
        'protocol', 'port_no', 'tx_bytes', 'rx_bytes', 'tx_kbps', 'rx_kbps',
        'tot_kbps', 'label', 'packetins', 'byte_per_flow', 'packet_per_flow'
    ]
    
    # Generate synthetic data
    np.random.seed(RANDOM_STATE)
    
    # Create a DataFrame with random values
    data = pd.DataFrame()
    
    # Generate features with appropriate distributions
    data['dt'] = np.random.uniform(0, 100, n_samples)
    data['switch'] = np.random.randint(1, 10, n_samples)
    data['src'] = np.random.randint(1, 255, n_samples)
    data['dst'] = np.random.randint(1, 255, n_samples)
    data['pkt_count'] = np.random.exponential(100, n_samples)
    data['byte_count'] = data['pkt_count'] * np.random.uniform(50, 1500, n_samples)
    data['duration'] = np.random.exponential(10, n_samples)
    data['duration_nsec'] = np.random.uniform(0, 1000000, n_samples)
    data['total_duration'] = data['duration'] + data['duration_nsec'] / 1e9
    data['flows'] = np.random.poisson(5, n_samples)
    data['pkt_rate'] = data['pkt_count'] / np.maximum(data['total_duration'], 0.001)
    data['pair_flow'] = np.random.poisson(3, n_samples)
    data['protocol'] = np.random.choice([6, 17], n_samples)  # TCP=6, UDP=17
    data['port_no'] = np.random.choice([80, 443, 53, 22, 25], n_samples)
    data['tx_bytes'] = np.random.exponential(10000, n_samples)
    data['rx_bytes'] = np.random.exponential(8000, n_samples)
    data['tx_kbps'] = data['tx_bytes'] * 8 / 1000 / np.maximum(data['total_duration'], 0.001)
    data['rx_kbps'] = data['rx_bytes'] * 8 / 1000 / np.maximum(data['total_duration'], 0.001)
    data['tot_kbps'] = data['tx_kbps'] + data['rx_kbps']
    data['packetins'] = np.random.poisson(10, n_samples)
    data['byte_per_flow'] = data['byte_count'] / np.maximum(data['flows'], 1)
    data['packet_per_flow'] = data['pkt_count'] / np.maximum(data['flows'], 1)
    
    # Create attack patterns
    # Normal traffic: 65% of data
    # Attack traffic: 35% of data
    attack_mask = np.random.choice([0, 1], n_samples, p=[0.65, 0.35])
    
    # Modify features for attack traffic to create patterns
    data.loc[attack_mask == 1, 'pkt_rate'] *= 5
    data.loc[attack_mask == 1, 'flows'] *= 3
    data.loc[attack_mask == 1, 'tx_bytes'] *= 4
    data.loc[attack_mask == 1, 'rx_bytes'] *= 0.5
    data.loc[attack_mask == 1, 'tx_kbps'] *= 4
    data.loc[attack_mask == 1, 'packet_per_flow'] *= 2
    
    # Add label column (0 = normal, 1 = attack)
    data['label'] = attack_mask
    
    # Save the synthetic dataset
    data.to_csv(data_path, index=False)
    print(f"Synthetic dataset created and saved to {data_path}")
    
    return data

def preprocess_data(data):
    """
    Preprocess the dataset for machine learning.
    
    Args:
        data (pandas.DataFrame): The raw dataset
        
    Returns:
        tuple: X_train, X_val, X_test, y_train, y_val, y_test, scaler
    """
    print("Preprocessing data...")
    
    # Separate features and target
    X = data.drop('label', axis=1)
    y = data['label']
    
    # First split: training + validation vs test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # Second split: training vs validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=VALIDATION_SIZE, 
        random_state=RANDOM_STATE, stratify=y_train_val
    )
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler for later use
    joblib.dump(scaler, 'models/scaler.pkl')
    
    print(f"Data split: {X_train.shape[0]} training, {X_val.shape[0]} validation, {X_test.shape[0]} test samples")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler

def train_and_evaluate_models(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Train and evaluate multiple machine learning models.
    
    Args:
        X_train, X_val, X_test: Training, validation, and test features
        y_train, y_val, y_test: Training, validation, and test labels
        
    Returns:
        dict: Dictionary of trained models and their performance metrics
    """
    print("Training and evaluating models...")
    
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        'SVM': SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_STATE),
        'Naive Bayes': GaussianNB()
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        start_time = time.time()
        
        # Train the model
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Predictions on validation set
        y_val_pred = model.predict(X_val)
        
        # Calculate metrics
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_precision = precision_score(y_val, y_val_pred)
        val_recall = recall_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred)
        
        print(f"{name} - Validation Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")
        print(f"{name} - Training Time: {training_time:.2f} seconds")
        
        # Save model
        joblib.dump(model, f'models/{name.lower().replace(" ", "_")}.pkl')
        
        # Store results
        results[name] = {
            'model': model,
            'val_accuracy': val_accuracy,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1': val_f1,
            'training_time': training_time
        }
    
    # Evaluate best model on test set
    best_model_name = max(results, key=lambda k: results[k]['val_accuracy'])
    best_model = results[best_model_name]['model']
    
    print(f"\nEvaluating best model ({best_model_name}) on test set...")
    y_test_pred = best_model.predict(X_test)
    
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    
    print(f"Test Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    
    # Plot and save confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Attack'], 
                yticklabels=['Normal', 'Attack'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{best_model_name} Confusion Matrix')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png')
    
    # Feature importance for Random Forest
    if 'Random Forest' in results:
        rf_model = results['Random Forest']['model']
        feature_names = [
            'dt', 'switch', 'src', 'dst', 'pkt_count', 'byte_count', 'duration',
            'duration_nsec', 'total_duration', 'flows', 'pkt_rate', 'pair_flow',
            'protocol', 'port_no', 'tx_bytes', 'rx_bytes', 'tx_kbps', 'rx_kbps',
            'tot_kbps', 'packetins', 'byte_per_flow', 'packet_per_flow'
        ]
        
        # Get feature importances
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Plot feature importances
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importances (Random Forest)')
        plt.bar(range(len(indices)), importances[indices], align='center')
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig('results/feature_importances.png')
        
        # Save feature importances to CSV
        importance_df = pd.DataFrame({
            'Feature': [feature_names[i] for i in indices],
            'Importance': importances[indices]
        })
        importance_df.to_csv('results/feature_importances.csv', index=False)
    
    # Save test results
    test_results = {
        'accuracy': test_accuracy,
        'precision': test_precision,
        'recall': test_recall,
        'f1': test_f1,
        'confusion_matrix': cm.tolist()
    }
    
    # Save all results
    all_results = {name: {k: v for k, v in model_results.items() if k != 'model'} 
                  for name, model_results in results.items()}
    all_results['test_results'] = test_results
    
    # Convert to DataFrame for easier viewing
    results_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Validation Accuracy': [results[m]['val_accuracy'] for m in results],
        'Validation Precision': [results[m]['val_precision'] for m in results],
        'Validation Recall': [results[m]['val_recall'] for m in results],
        'Validation F1': [results[m]['val_f1'] for m in results],
        'Training Time (s)': [results[m]['training_time'] for m in results]
    })
    
    # Save results to CSV
    results_df.to_csv('results/model_comparison.csv', index=False)
    
    # Plot model comparison
    plt.figure(figsize=(12, 8))
    results_df.set_index('Model')[['Validation Accuracy', 'Validation Precision', 
                                  'Validation Recall', 'Validation F1']].plot(kind='bar')
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('results/model_comparison.png')
    
    return results

def main():
    """Main function to run the training pipeline."""
    print("Starting DDoS Detection ML Training Pipeline")
    
    # Load data
    data = load_data()
    print(f"Dataset shape: {data.shape}")
    
    # Display class distribution
    class_counts = data['label'].value_counts()
    print("\nClass distribution:")
    for label, count in class_counts.items():
        print(f"Class {label}: {count} samples ({count/len(data)*100:.2f}%)")
    
    # Preprocess data
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = preprocess_data(data)
    
    # Train and evaluate models
    results = train_and_evaluate_models(X_train, X_val, X_test, y_train, y_val, y_test)
    
    print("\nTraining complete. Models saved in 'models/' directory.")
    print("Results saved in 'results/' directory.")

if __name__ == "__main__":
    main()