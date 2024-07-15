import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from data_preprocessing import load_and_preprocess_data
from model_training import GRUTFTModel
from eval_metrics import plot_learning_curves

def create_sequences(data, target, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        targets.append(target.iloc[i + seq_length])
    return np.array(sequences), np.array(targets)

def main():
    file_path = "EURUSD_Candlestick_1_Hour_BID_01.01.2004-30.03.2024.csv"
    X_train, X_val, y_train, y_val, scaler = load_and_preprocess_data(file_path)

    sequence_length = 24

    X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, sequence_length)

    X_train_tensor = torch.Tensor(X_train_seq)
    X_val_tensor = torch.Tensor(X_val_seq)
    y_train_tensor = torch.Tensor(y_train_seq).unsqueeze(1)
    y_val_tensor = torch.Tensor(y_val_seq).unsqueeze(1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    input_size = X_train_tensor.shape[2]
    output_size = 1
    model = GRUTFTModel(input_size, output_size)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)

    epochs = 2
    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    patience, trials = 5, 1

    for epoch in range(epochs):
        model.train()
        epoch_train_losses = []

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            epoch_train_losses.append(loss.item())

        train_losses.append(np.mean(epoch_train_losses))

        model.eval()
        with torch.no_grad():
            val_loss = np.mean([criterion(model(X_batch), y_batch).item() for X_batch, y_batch in val_loader])
            val_losses.append(val_loss)
            print(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {val_loss}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                trials = 0
                torch.save(model.state_dict(), 'best_model.pth')
            else:
                trials += 1
                if trials >= patience:
                    print("Early stopping triggered")
                    break

    plot_learning_curves(train_losses, val_losses)

if __name__ == "__main__":
    main()
