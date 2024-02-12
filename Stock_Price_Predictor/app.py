import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# Retrieve stock price data
stock_data = yf.download("TATAMOTORS.NS", period="max")
stock_data.reset_index(inplace=True)
stock_data.to_csv("tatamotors.csv")
stock_data.plot(x='Open',y='Close',kind='line')
# Select features for training (Open, High, Low, Volume)
features = ['Open']

# Extract selected features and target variable (Close price)
X = stock_data[features].values
y = stock_data['Close'].values

# Scale the features using Min-Max scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert numpy arrays to PyTorch tensors
X_train_tensor = torch.Tensor(X_train).unsqueeze(2)
y_train_tensor = torch.Tensor(y_train).unsqueeze(1)
X_test_tensor = torch.Tensor(X_test).unsqueeze(2)
y_test_tensor = torch.Tensor(y_test).unsqueeze(1)

class StockPriceRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(StockPriceRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # Initialize hidden state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # Initialize cell state
        lstm_out, _ = self.lstm(x, (h0, c0))
        out = self.fc(lstm_out[:, -1, :])  # Use the last output of the sequence
        return out

# Initialize the model with more layers and increased complexity
model = StockPriceRNN(input_size=len(features), hidden_size=64, num_layers=20, output_size=1, dropout=0.2)


# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Create DataLoader objects for batching
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Training loop
for epoch in range(1000):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/1000], Loss: {loss.item():.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_loss = criterion(test_outputs, y_test_tensor)
    print(f'Test Loss: {test_loss.item():.4f}')

# Preprocess the new opening price (151.00)
new_opening_price = 916.10

scaled_opening_price = scaler.transform(np.array([[new_opening_price]]))

# Convert the scaled opening price to a PyTorch tensor
opening_price_tensor = torch.Tensor(scaled_opening_price).unsqueeze(1)

# Make a prediction using the trained model
model.eval()
with torch.no_grad():
    predicted_closing_price = model(opening_price_tensor)
    print("Predicted closing price:", predicted_closing_price.item())