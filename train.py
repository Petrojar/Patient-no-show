import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Načítanie dát
data = pd.read_csv("KaggleV2-May-2016.csv")
data.columns = data.columns.str.strip()  # odstráni medzery

# Preklad názvov ak treba
data = data.rename(columns={
    'Hipertension': 'Hypertension',
    'Handcap': 'Handcap',
    'No-show': 'No-show',
    'SMS_received': 'SMS_received'
})

# Filtrácia a výber relevantných stĺpcov
data = data[['Age', 'Scholarship', 'Hypertension', 'Diabetes', 'Alcoholism', 'Handcap', 'SMS_received', 'No-show']]
data = data[(data['Age'] > 0) & (data['Age'] < 115)]
data['No-show'] = data['No-show'].map({'Yes': 1, 'No': 0})

X = data.drop(columns='No-show').values
y = data['No-show'].values

# Škálovanie
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "scaler.pkl")

# Rozdelenie dát
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# PyTorch dataset
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)

# Model
class SimpleModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

model = SimpleModel(X_train.shape[1])
criterion = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

# Tréning
for epoch in range(20):
    model.train()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Uloženie modelu
torch.save(model.state_dict(), "model.pt")
print("Model a scaler uložený.")
