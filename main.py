import joblib
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel


# ===== Model (rovnaký ako v train.py) =====
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

# ===== FastAPI setup =====
app = FastAPI()

# ===== Načítanie modelu a scaleru =====
scaler = joblib.load("scaler.pkl")
model = SimpleModel(input_dim=7)  # 7 vstupných prvkov (prispôsob podľa tvojho modelu)
model.load_state_dict(torch.load("model.pt", map_location=torch.device("cpu")))
model.eval()

# ===== Pydantic pre JSON API =====
class InputData(BaseModel):
    age: float
    scholarship: int
    hypertension: int
    diabetes: int
    alcoholism: int
    handcap: int
    sms_received: int

# ===== Hlavný endpoint pre HTML form =====
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <form action="/predict_form" method="post">
        Age: <input type="number" name="age"><br>
        Scholarship: <input type="number" name="scholarship"><br>
        Hypertension: <input type="number" name="hypertension"><br>
        Diabetes: <input type="number" name="diabetes"><br>
        Alcoholism: <input type="number" name="alcoholism"><br>
        Handcap: <input type="number" name="handcap"><br>
        SMS Received: <input type="number" name="sms_received"><br>
        <input type="submit">
    </form>
    """

# ===== POST cez formulár =====
@app.post("/predict_form")
async def predict_form(
    age: float = Form(...),
    scholarship: int = Form(...),
    hypertension: int = Form(...),
    diabetes: int = Form(...),
    alcoholism: int = Form(...),
    handcap: int = Form(...),
    sms_received: int = Form(...)
):
    input_data = np.array([[age, scholarship, hypertension, diabetes, alcoholism, handcap, sms_received]])
    scaled = scaler.transform(input_data)
    with torch.no_grad():
        prediction = model(torch.tensor(scaled, dtype=torch.float32)).item()
    return {"probability_of_no_show": prediction}

# ===== POST cez JSON API =====
@app.post("/predict")
async def predict_json(data: InputData):
    input_array = np.array([[data.age, data.scholarship, data.hypertension, data.diabetes,
                             data.alcoholism, data.handcap, data.sms_received]])
    scaled = scaler.transform(input_array)
    with torch.no_grad():
        prediction = model(torch.tensor(scaled, dtype=torch.float32)).item()
    return {"probability_of_no_show": prediction}
