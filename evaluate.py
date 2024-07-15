import numpy as np
import torch
import torch.nn as nn
from data_preprocessing import load_and_preprocess_data
from model_training import GRUTFTModel
from eval_metrics import evaluate_model

def create_sequences(data, target, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        targets.append(target.iloc[i + seq_length])
    return np.array(sequences), np.array(targets)

def eval():
    file_path = "EURUSD_Candlestick_1_Hour_BID_01.01.2004-30.03.2024.csv"
    X_train, X_val, y_train, y_val, scaler = load_and_preprocess_data(file_path)

    sequence_length = 24

    X_val_seq, y_val_seq = create_sequences(X_val, y_val, sequence_length)

    X_val_tensor = torch.Tensor(X_val_seq)
    y_val_tensor = torch.Tensor(y_val_seq).unsqueeze(1)

    input_size = X_val_tensor.shape[2]
    output_size = 1
    model = GRUTFTModel(input_size, output_size)

    model.load_state_dict(torch.load('best_model.pth'))

    criterion = nn.MSELoss()

    evaluate_model(model, criterion, X_val_tensor, y_val_tensor)

if __name__ == "__main__":
    eval()
