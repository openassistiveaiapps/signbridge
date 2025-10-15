from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import math
import time

app = FastAPI()

# Allow frontend to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok"}

class LandmarksRequest(BaseModel):
    landmarks: List[List[List[float]]]  # hands -> points -> [x, y, z]

# --- Cache to reduce repeated computations ---
gesture_cache = {}
CACHE_TTL = 0.2  # seconds

# --- Helper functions ---
def distance(p1, p2):
    """Euclidean distance between two 3D points."""
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(p1, p2)]))

def angle_between(p1, p2, p3):
    """Compute angle at p2 formed by p1-p2-p3 in degrees."""
    a = [p1[i]-p2[i] for i in range(3)]
    b = [p3[i]-p2[i] for i in range(3)]
    dot = sum([a[i]*b[i] for i in range(3)])
    norm_a = math.sqrt(sum([a[i]**2 for i in range(3)]))
    norm_b = math.sqrt(sum([b[i]**2 for i in range(3)]))
    if norm_a * norm_b == 0:
        return 0.0
    cos_theta = max(min(dot/(norm_a*norm_b), 1.0), -1.0)
    return math.degrees(math.acos(cos_theta))

def detect_gesture(hand_landmarks):
    """
    Enhanced gesture detection using distances and angles.
    Returns one of: Open, Fist, ThumbsUp, Peace, Pointing, OK, Unknown
    """
    # Ensure landmarks are 3D tuples (x,y,z). Frontend may send 2D [x,y].
    def to3(p):
        if len(p) == 3:
            return tuple(p)
        return (p[0], p[1], 0.0)

    lm = [to3(p) for p in hand_landmarks]

    # Indices per MediaPipe/Handpose: 0=wrist, 1..4 thumb, 5..8 index, 9..12 middle, 13..16 ring, 17..20 pinky
    tips = [4, 8, 12, 16, 20]
    pips = [2, 6, 10, 14, 18]

    # Palm size: use distance wrist (0) to middle MCP (9) as reference
    palm_size = max(1e-6, distance(lm[0], lm[9]))

    # Heuristic: a finger is folded if tip is significantly closer to corresponding PIP/MCP than expected
    folded = []
    for tip_idx, pip_idx in zip(tips, pips):
        tip_d = distance(lm[tip_idx], lm[0])
        pip_d = distance(lm[pip_idx], lm[0])
        # If tip is much closer to wrist than pip is, finger is folded
        folded.append(tip_d < 0.85 * pip_d)

    # Debugging: if landmarks look degenerate, return Unknown
    if any(math.isnan(c) for pt in lm for c in pt):
        return "Unknown"

    # Gesture rules (order matters)
    # Fist: most fingers folded
    if sum(folded) >= 4:
        return "Fist"

    # Thumbs Up: thumb extended (not folded) and most others folded
    if not folded[0] and sum(folded[1:]) >= 3:
        return "ThumbsUp"

    # Peace: index and middle extended, others folded
    if (not folded[1]) and (not folded[2]) and folded[3] and folded[4]:
        return "Peace"

    # Pointing: index extended, others folded
    if (not folded[1]) and folded[2] and folded[3] and folded[4]:
        return "Pointing"

    # OK sign: thumb tip close to index tip
    if distance(lm[4], lm[8]) < 0.25 * palm_size and (not folded[1] or not folded[0]):
        return "OK"

    # Open: most fingers extended
    if sum(folded[1:]) == 0:
        return "Open"

    return "Unknown"

@app.post("/detect_gesture")
async def detect_gesture_endpoint(req: LandmarksRequest):
    now = time.time()
    results = []

    for hand in req.landmarks:
        # Check cache
        cache_key = tuple(tuple(map(tuple, hand)))
        if cache_key in gesture_cache:
            cached_time, cached_result = gesture_cache[cache_key]
            if now - cached_time < CACHE_TTL:
                results.append(cached_result)
                continue

        gesture = detect_gesture(hand)
        gesture_cache[cache_key] = (now, gesture)
        results.append(gesture)

    if len(results) == 1:
        return {"gesture": results[0]}
    return {"gesture": results}
