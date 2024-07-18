# Machine-Learning Forex Prediction

## Overview
This project aims to predict Forex market movements using machine learning techniques. The model is trained on historical EUR/USD exchange rate data (static .csv) and utilizes a GRU (Gated Recurrent Unit) with Temporal Fusion Transformer (TFT) architecture to make future predictions (next hour prices based on previous 24 hours prices).

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Example Usage](#example-usage)
- [Technologies Used](#technologies-used)

## Features
1. **Data Preprocessing:**
    - Loading and Cleaning: Load data, handle missing values, and ensure quality.
    - Scaling: Normalize the data to ensure that features are on a similar scale.
    - Date-Time Features: Extract date-time features to capture temporal patterns.

2. **Sequence Creation:**
     - Time Series Windowing: Create sequences of fixed length to serve as input for the model.
     - Sliding Window Approach: To ensure the model captures temporal dependencies.

3. **Feature Engineering:**
     - Technical Indicators: Aside from OHLCV, dataset is enhanced with technical indicators commonly used in trading strategies.

4. **Model Architecture:**
     - GRU-TFT: Combine Gated Recurrent Unit (GRU) and Temporal Fusion Transformer (TFT) for robust time series forecasting.
     - Attention Mechanisms: To weigh the importance of different time steps in the input sequence.

5. **Model Training:**
     - Loss Function: Mean Squared Error (MSE) as the loss function to measure prediction accuracy.
     - Optimization: Adam optimizer with weight decay for efficient training.
     - Early Stopping: Implement early stopping to avoid overfitting and save the best model.

6. **Evaluation Metrics:**
     - Mean Squared Error (MSE): Measures the average squared difference between actual and predicted values.
     - Mean Absolute Error (MAE): Measures the average absolute difference between actual and predicted values.
     - R-squared (R²): Indicates the proportion of the variance in the dependent variable that is predictable from the independent variables.

7. **Visualizations:**
     - Learning Curves: Plot training and validation loss over epochs.
     - True vs Predicted Values: Scatter plot to compare the true and predicted values.
     - Residuals Plot: Scatter plot of residuals vs predicted values to diagnose potential issues in the model.
     - Residuals Distribution: Histogram of residuals to check their distribution and identify any patterns.

## Installation
1. **Clone the repository**
    ```sh
    git clone https://github.com/lkjearl/ML-Forex-Prediction
    cd ml-forex-prediction
    ```
2. **Install the required dependencies**
    ```sh
    pip install -r requirements.txt
    ```

3. **Download dataset**
    - Ensure you have the dataset in the project directory (I used mine from duskacopy).
    - Modify *file_path* in **main()** and **eval()** to your file name.
## Usage
1. **Train model**
    ```sh
    python main.py
    ```

2. **Evaluate model and generate plots**
    ```sh
    python eval_metrics.py
    ```

## Example Usage
For a step-by-step example, you may refer to **'example_usage.ipynb'** jupyter notebook from this repository.

## Technologies Used

- Python
- Pandas
- NumPy
- PyTorch
- scikit-learn
- Matplotlib
- Seaborn