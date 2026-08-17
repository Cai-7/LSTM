import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
np.random.seed(42)
torch.manual_seed(42)

t = np.linspace(0, 100, 1000)
data = np.sin(t) + np.random.normal(0, 0.1, 1000)
data = data.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

def create_sequences(data, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i+look_back])
        y.append(data[i+look_back])
    return np.array(X), np.array(y)

look_back = 20
X, y = create_sequences(data_scaled, look_back)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_f = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_i = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_g = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_o = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x_t, h_prev, c_prev):
        combined = torch.cat((h_prev, x_t), dim=1)

        f_t = torch.sigmoid(self.W_f(combined))
        i_t = torch.sigmoid(self.W_i(combined))
        g_t = torch.tanh(self.W_g(combined))
        o_t = torch.sigmoid(self.W_o(combined))

        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t

class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = LSTMCell(input_size, hidden_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
        c_t = torch.zeros(batch_size, self.hidden_size).to(x.device)

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            h_t, c_t = self.cell(x_t, h_t, c_t)
            outputs.append(h_t)

        outputs = torch.stack(outputs, dim=1)
        return outputs, (h_t, c_t)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = CustomLSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTMModel(input_size=1, hidden_size=50, output_size=1).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

num_epochs = 100
train_losses = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")

model.eval()
with torch.no_grad():
    train_predict = model(X_train).cpu().numpy()
    test_predict = model(X_test).cpu().numpy()

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_train_actual = scaler.inverse_transform(y_train.cpu().numpy())
y_test_actual = scaler.inverse_transform(y_test.cpu().numpy())

plt.figure(figsize=(12, 6))

plt.plot(scaler.inverse_transform(data_scaled), label='Original Data', alpha=0.5)

train_plot = np.empty_like(data_scaled)
train_plot[:, :] = np.nan
train_plot[look_back:look_back + len(train_predict)] = train_predict
plt.plot(train_plot, label='Train Prediction')

test_plot = np.empty_like(data_scaled)
test_plot[:, :] = np.nan
test_start = look_back + len(train_predict)
test_plot[test_start:test_start + len(test_predict)] = test_predict
plt.plot(test_plot, label='Test Prediction')

plt.legend()
plt.title("LSTM (From Scratch) - Sine Wave Prediction")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.show()
