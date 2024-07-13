import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from model_training import GRUTFTModel

def load_and_preprocess_data(file_path):
    return X_val, y_val, scaler

def plot_learning_curves(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curves')
    plt.legend()
    plt.show()

def plot_time_series(actual, predicted):
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual')
    plt.plot(predicted, label='Predicted')
    plt.title('Actual vs. Predicted Time Series')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

def plot_residuals(actual, predicted):
    residuals = actual - predicted
    plt.figure(figsize=(10, 6))
    plt.plot(residuals, marker='o', linestyle='None')
    plt.axhline(y=0, color='r', linestyle='-')
    plt.title('Residual Plot')
    plt.xlabel('Index')
    plt.ylabel('Residual')
    plt.show()
    
def evaluate_model(model, criterion, X_val_tensor, y_val_tensor):
    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_val_tensor), y_val_tensor)
        print(f"Validation Loss: {val_loss.item()}")

        y_pred = model(X_val_tensor).detach().numpy()
        y_true = y_val_tensor.numpy()

        acc = accuracy_score(y_true, np.round(y_pred))
        precision = precision_score(y_true, np.round(y_pred))
        recall = recall_score(y_true, np.round(y_pred))
        f1 = f1_score(y_true, np.round(y_pred))

        print(f"Accuracy: {acc}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")

        cm = confusion_matrix(y_true, np.round(y_pred))
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        plt.show()

        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()

        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.show()

def eval():
    file_path = "EURUSD_Candlestick_1_Hour_BID_01.01.2004-30.03.2024.csv"
    X_val, y_val, scaler = load_and_preprocess_data(file_path)

    X_val_tensor = torch.Tensor(X_val)
    y_val_tensor = torch.Tensor(y_val.values)

    input_size = X_val_tensor.shape[2]
    model = GRUTFTModel(input_size, output_size)

    model.load_state_dict(torch.load('best_model.pth'))

    criterion = nn.MSELoss()

    evaluate_model(model, criterion, X_val_tensor, y_val_tensor)

if __name__ == "__main__":
    eval()
