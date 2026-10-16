import joblib
import numpy as np
import torch
import torch.nn as nn

from fastapi import FastAPI
from pydantic import BaseModel, Field

app=FastAPI(title="Predictive Maintenance API")
device = torch.device("cpu")

class LSTMModel(nn.Module):
    def __init__(self, input_dim=21, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1]).squeeze(1)


model = LSTMModel()
import os

MODEL_PATH = os.getenv("MODEL_PATH", "model.pt")
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

import os
import joblib

SCALER_PATH = os.getenv("SCALER_PATH", "scaler.pkl")
scaler = joblib.load(SCALER_PATH)

class PredictionRequest(BaseModel):
    window: list[list[float]]

@app.get("/")
def home():
    return {"message": "API is running"}


@app.post("/predict")
def predict(request: PredictionRequest):
    window = np.asarray(request.window, dtype=np.float32)

    if window.shape != (20, 21):
        return {"error": "Expected 20 cycles and 21 sensors"}

    scaled = scaler.transform(window)
    X = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits = model(X)
        probability = torch.sigmoid(logits).item()

    threshold = 0.1
    prediction = int(probability >= threshold)

    return {
        "risk_probability": probability,
        "prediction": prediction,
        "status": "at_risk" if prediction else "not_at_risk"
    }