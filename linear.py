import torch
from torch import nn
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from torch.optim import Adam
import matplotlib.pyplot as plt

SEED = 1234
NUM_SAMPLES = 50
torch.manual_seed(SEED)
TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15
INPUT_DIM = X_train.shape[1]
OUTPUT_DIM = y_train.shape[1]
LEARNING_RATE = 1e-1
NUM_EPOCHS = 100

def generate_data(num_samples):
    '''Generate data for linear regression. 
    theto 0 and theta1 being defined here'''
    X = np.array(range(num_samples))
    random_noise = np.random.uniform(-10,20, size=num_samples)
    y = 3.5*X + random_noise # add some noise
    return X, y
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super(LinearRegression, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
    def forward(self, x_in):
        y_pred = self.fc1(x_in)
        return y_pred
if __name__ == "__main__":
    X, y = generate_data(num_samples=NUM_SAMPLES)
    data = np.vstack([X, y]).T
    print(data[:5])

    df = pd.DataFrame(data, columns=["X","y"])

    X = df[["X"]].values
    y = df[["y"]].values
    X_train, X_, y_train, y_ = train_test_split(X, y, train_size = TRAIN_SIZE)
    print (f"train: {len(X_train)} ({(len(X_train) / len(X)):.2f})\n"
       f"remaining: {len(X_)} ({(len(X_) / len(X)):.2f})")
    # Split (test)
    X_val, X_test, y_val, y_test = train_test_split(X_, y_, train_size=0.5)
    print(f"train: {len(X_train)} ({len(X_train)/len(X):.2f})\n"
      f"val: {len(X_val)} ({len(X_val)/len(X):.2f})\n"
      f"test: {len(X_test)} ({len(X_test)/len(X):.2f})")
    # Standardize the data (mean = 0 and std = 1) using training data
    X_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train)

    X_train = X_scaler.transform(X_train)
    y_train = y_scaler.transform(y_train).ravel().reshape(-1, 1)
    X_val = X_scaler.transform(X_val)
    y_val = y_scaler.transform(y_val).ravel().reshape(-1, 1)
    X_test = X_scaler.transform(X_test)
    y_test = y_scaler.transform(y_test).ravel().reshape(-1, 1)

    # Check (means should be ~0 and std should be ~1)
    print (f"mean: {np.mean(X_test, axis=0)[0]:.1f}, std: {np.std(X_test, axis=0)[0]:.1f}")
    print (f"mean: {np.mean(y_test, axis=0)[0]:.1f}, std: {np.std(y_test, axis=0)[0]:.1f}")

    model = LinearRegression(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM)
    print(model.named_parameters)

    loss_fn = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr = LEARNING_RATE)

    # Convert data to tensors
    X_train = torch.Tensor(X_train)
    y_train = torch.Tensor(y_train)
    X_val = torch.Tensor(X_val)
    y_val = torch.Tensor(y_val)
    X_test = torch.Tensor(X_test)
    y_test = torch.Tensor(y_test)

    # Training
    for epoch in range(NUM_EPOCHS):
        # Forward pass
        y_pred = model(X_train)

        # Loss
        loss = loss_fn(y_pred, y_train)

        # Zero all gradients
        optimizer.zero_grad()

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        if epoch%20==0:
            print (f"Epoch: {epoch} | loss: {loss:.2f}")
    # Predictions
    pred_train = model(X_train)
    pred_test = model(X_test)
    # Performance
    train_error = loss_fn(pred_train, y_train)
    test_error = loss_fn(pred_test, y_test)
    print(f"train_error: {train_error:.2f}")
    print(f"test_error: {test_error:.2f}")
