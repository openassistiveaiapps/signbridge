# model.py
import numpy as np
import torch
import torch.nn as nn

# ---------------- Gesture labels ----------------
GESTURE_LABELS = ["More", "Food", "Play", "None"]

# ---------------- Neural Network ----------------
class SimpleGestureClassifier(nn.Module):
    def __init__(self, input_size=63, hidden_size=64, output_size=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.fc(x)

# ---------------- Initialize model ----------------
model = SimpleGestureClassifier()
model.eval()  # inference mode

# ---------------- Prediction Function ----------------
def predict(hand_landmarks):
    """
    Predict gesture for a single hand:
    - Normalizes landmarks (center + scale)
    - Uses the model if possible
    - Returns a label from GESTURE_LABELS
    """
    # Flatten and normalize landmarks
    arr = np.array(hand_landmarks, dtype=np.float32)  # Nx3
    # center by wrist
    wrist = arr[0, :2]
    arr[:, 0] -= wrist[0]
    arr[:, 1] -= wrist[1]

    # scale normalization by max distance from wrist
    max_dist = np.max(np.linalg.norm(arr[:, :2], axis=1))
    if max_dist > 0:
        arr[:, :2] /= max_dist

    x = torch.tensor(arr.flatten(), dtype=torch.float32).unsqueeze(0)

    try:
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            return GESTURE_LABELS[pred_idx]
    except Exception as e:
        # fallback: simple heuristic based on finger spread
        return fallback_heuristic(hand_landmarks)

# ---------------- Simple heuristic fallback ----------------
def fallback_heuristic(hand_landmarks):
    """
    Baseline gesture prediction in case model fails
    """
    tips = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
    wrist = hand_landmarks[0]
    folded_count = 0
    for tip in tips[1:]:
        if hand_landmarks[tip][1] > wrist[1]:
            folded_count += 1
    if folded_count >= 4:
        return "None"
    return "More"  # default guess
