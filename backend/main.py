from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import base64
import io
import cv2
import numpy as np
from pydantic import BaseModel
from PIL import Image
import mediapipe as mp

app = FastAPI()

# Allow frontend to access API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "SignBridge API is live 🚀"}

@app.get("/suggestions")
def get_suggestions():
    return {"suggestions": ["More", "Yes", "No"]}

class FrameData(BaseModel):
    image_base64: str

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

def classify_gesture(landmarks):
    # Basic heuristic: Check distance between wrist and middle fingertip
    wrist = landmarks.landmark[0]
    middle_tip = landmarks.landmark[12]
    dist = abs(middle_tip.y - wrist.y)

    # Simple threshold — tune this
    if dist > 0.15:
        return "More"
    else:
        return "All Done"

@app.post("/detect")
def detect_gesture(frame: FrameData):
    img_bytes = base64.b64decode(frame.image_base64)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_np = np.array(img)

    result = hands.process(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    if not result.multi_hand_landmarks:
        return {"gesture": "None", "suggestions": ["Yes", "No", "More"]}

    gesture_name = classify_gesture(result.multi_hand_landmarks[0])

    # Map gesture to suggestions
    suggestions_map = {
        "More": ["More", "Yes", "No"],
        "All Done": ["All Done", "No", "More"],
    }

    return {"gesture": gesture_name, "suggestions": suggestions_map.get(gesture_name, ["Yes", "No", "More"])}