from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from datetime import datetime, timezone
import math

app = FastAPI()

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Request Model ----------------
class GestureRequest(BaseModel):
    landmarks: List[List[List[float]]]  # hands -> points -> [x, y, z]

# ---------------- Helper Functions ----------------
def distance(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

def heuristic_detect_hand(hand_landmarks):
    lm = [p if len(p) == 3 else [p[0], p[1], 0] for p in hand_landmarks]
    wrist = lm[0]
    tips = [4, 8, 12, 16, 20]
    pips = [2, 6, 10, 14, 18]
    palm_size = max(1e-6, distance(wrist, lm[9]))

    folded = []
    for tip_idx, pip_idx in zip(tips, pips):
        tip_d = distance(lm[tip_idx], wrist)
        pip_d = distance(lm[pip_idx], wrist)
        folded.append(tip_d < 0.75 * pip_d)  # stricter fold detection

    thumb_index_dist = distance(lm[4], lm[8])

    # Priority order:
    if sum(folded) == 5:
        return "FIST"
    if not folded[0] and sum(folded[1:]) == 4:
        return "RAISE_HAND"
    if not folded[1] and folded[2] and folded[3] and folded[4]:
        return "POINT"
    if not folded[1] and not folded[2] and folded[3] and folded[4]:
        return "PEACE"
    if thumb_index_dist < 0.2 * palm_size:
        return "OK"
    if sum(folded[1:]) == 0 and not folded[0]:
        return "OPEN"

    return "NONE"


def combine_gestures(per_hand):
    """Combine multiple hands into one gesture"""
    if len(per_hand) == 0:
        return "NONE"
    if all(g == "RAISE_HAND" for g in per_hand):
        return "MORE"
    if any(g == "RAISE_HAND" for g in per_hand):
        return "SINGLE"
    if all(g == "FIST" for g in per_hand):
        return "STOP"
    return per_hand[0]  # fallback to first hand gesture

# ---------------- Endpoints ----------------
@app.get("/health")
async def health_check():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}

@app.post("/api/gesture")
async def receive_gesture(req: GestureRequest):
    hands = req.landmarks
    per_hand_gestures = [heuristic_detect_hand(hand) for hand in hands]
    combined = combine_gestures(per_hand_gestures)

    print(f"👋 Per-hand gestures: {per_hand_gestures}, Combined: {combined}")

    return {
        "per_hand_gestures": per_hand_gestures,
        "predicted_gesture": combined,
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }
