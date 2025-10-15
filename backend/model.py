# model.py
import numpy as np
import torch
import torch.nn as nn

class SimpleGestureClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(63, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # e.g., 4 gestures
        )

    def forward(self, x):
        return self.fc(x)

# Load model (here we just use random weights)
model = SimpleGestureClassifier()
model.eval()

GESTURE_LABELS = ["More", "Food", "Play", "None"]

def predict(landmarks):
    arr = np.array(landmarks).flatten().astype(np.float32)
    x = torch.tensor(arr).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=1).item()
    return GESTURE_LABELS[pred]
