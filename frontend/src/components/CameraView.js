import React, { useRef, useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";
import * as handpose from "@tensorflow-models/handpose";

export default function CameraView() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const modelRef = useRef(null);
  const lastSentTime = useRef(0);
  const [perHandGestures, setPerHandGestures] = useState([]);
  const [combinedGesture, setCombinedGesture] = useState("Waiting...");
  const [timer, setTimer] = useState(30); // 30 second session

  const API_URL = "http://localhost:8000/api/gesture";

  // ---------------- Setup Camera ----------------
  const setupCamera = async () => {
    const video = videoRef.current;
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 },
      audio: false,
    });
    video.srcObject = stream;
    return new Promise((resolve) => {
      video.onloadedmetadata = () => {
        video.play();
        resolve(video);
      };
    });
  };

  // ---------------- Draw Hands ----------------
  const drawHand = (predictions, ctx, gestures) => {
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx.strokeStyle = "#00FF00";
    ctx.lineWidth = 2;
    ctx.font = "18px Arial";

    predictions.forEach((prediction, index) => {
      const landmarks = prediction.landmarks;
      landmarks.forEach(([x, y]) => {
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, 2 * Math.PI);
        ctx.fillStyle = "rgba(0, 255, 0, 0.6)";
        ctx.fill();
      });

      const [x, y] = landmarks[0];
      ctx.fillStyle = "red";
      ctx.fillText(gestures[index] || "NONE", x - 30, y - 20);
    });
  };

  // ---------------- Call Backend ----------------
  const sendGestureToAPI = async (landmarksArr) => {
    const now = Date.now();
    if (now - lastSentTime.current < 5000) return; // throttle every 5s
    lastSentTime.current = now;

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ landmarks: landmarksArr }),
      });
      if (!response.ok) {
        console.error("FastAPI error:", response.statusText);
        return;
      }
      const data = await response.json();
      setPerHandGestures(data.per_hand_gestures);
      setCombinedGesture(data.predicted_gesture);
    } catch (err) {
      console.error("Error calling FastAPI:", err);
    }
  };

  // ---------------- Main Detection Loop ----------------
  let startTime = null;
  const detectHands = async (timestamp) => {
    if (!startTime) startTime = timestamp;
    const elapsed = Math.floor((timestamp - startTime) / 1000);
    const remaining = Math.max(30 - elapsed, 0);
    setTimer(remaining);

    if (!modelRef.current || !videoRef.current) {
      if (remaining > 0) requestAnimationFrame(detectHands);
      return;
    }

    const video = videoRef.current;
    if (video.videoWidth === 0 || video.videoHeight === 0) {
      if (remaining > 0) requestAnimationFrame(detectHands);
      return;
    }

    const predictions = await modelRef.current.estimateHands(video);
    const ctx = canvasRef.current.getContext("2d");

    if (predictions.length > 0) {
      const landmarksArr = predictions.map((p) =>
        p.landmarks.map((l) => [...l, 0])
      );
      drawHand(predictions, ctx, perHandGestures);
      await sendGestureToAPI(landmarksArr);
    } else {
      setPerHandGestures([]);
      setCombinedGesture("Waiting...");
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    }

    if (remaining > 0) requestAnimationFrame(detectHands);
  };

  // ---------------- Init ----------------
  useEffect(() => {
    const init = async () => {
      try {
        await tf.setBackend("webgl");
        await tf.ready();
        modelRef.current = await handpose.load();
        await setupCamera();
        requestAnimationFrame(detectHands);
      } catch (err) {
        console.error("Initialization error:", err);
      }
    };
    init();
  }, []);

  // ---------------- Render ----------------
  return (
    <div style={{ position: "relative", width: "640px", height: "480px" }}>
      <video
        ref={videoRef}
        style={{
          position: "absolute",
          width: "640px",
          height: "480px",
          transform: "scaleX(-1)",
        }}
      />
      <canvas
        ref={canvasRef}
        width="640"
        height="480"
        style={{
          position: "absolute",
          width: "640px",
          height: "480px",
          transform: "scaleX(-1)",
        }}
      />
      <div
        style={{
          position: "absolute",
          bottom: 10,
          left: 10,
          backgroundColor: "rgba(0,0,0,0.5)",
          color: "white",
          padding: "5px 10px",
          borderRadius: "5px",
          fontSize: "18px",
        }}
      >
        <div>Timer: {timer}s</div>
        <div>Combined: {combinedGesture}</div>
        <div>
          Per Hand:{" "}
          {perHandGestures.length > 0
            ? perHandGestures.join(" | ")
            : "Waiting..."}
        </div>
      </div>
    </div>
  );
}
