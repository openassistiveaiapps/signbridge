import { useRef, useEffect } from "react";
import * as handpose from "@tensorflow-models/handpose";

export default function CameraView() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const modelRef = useRef(null);
  const processingRef = useRef(false); 
  const captureActiveRef = useRef(true); 
  const lastPredictionsRef = useRef([]); 
  const lastGestureRef = useRef("None"); // store last gesture

  useEffect(() => {
    let animationId;
    let stream;
    let lastApiCallTime = 0;
    const API_INTERVAL = 500; 
    const CAPTURE_DURATION = 30000; // 30 seconds

    const loadModelAndCamera = async () => {
      modelRef.current = await handpose.load();
      console.log("✅ Handpose model loaded");

      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 320, height: 240 },
        });
        videoRef.current.srcObject = stream;
      } catch (err) {
        console.error("⚠️ Cannot access camera:", err);
        return;
      }

      videoRef.current.onloadedmetadata = () => {
        const canvas = canvasRef.current;
        canvas.width = videoRef.current.videoWidth;
        canvas.height = videoRef.current.videoHeight;

        videoRef.current.play();
        requestAnimationFrame(processFrame);
      };

      // Stop detection and API updates after 30 seconds
      setTimeout(() => {
        captureActiveRef.current = false;
        console.log("⏹️ Stopped hand detection and API calls after 30 seconds");
      }, CAPTURE_DURATION);
    };

    const processFrame = async () => {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");

      if (!videoRef.current) {
        animationId = requestAnimationFrame(processFrame);
        return;
      }

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);

      if (captureActiveRef.current && modelRef.current && !processingRef.current) {
        processingRef.current = true;

        const predictions = await modelRef.current.estimateHands(
          videoRef.current,
          true
        );

        if (predictions.length > 0) {
          lastPredictionsRef.current = predictions;

          predictions.forEach((hand) => {
            hand.landmarks.forEach(([x, y]) => {
              ctx.beginPath();
              ctx.arc(x, y, 5, 0, 2 * Math.PI);
              ctx.fillStyle = "red";
              ctx.fill();
            });
          });

          const now = Date.now();
          if (now - lastApiCallTime > API_INTERVAL) {
            lastApiCallTime = now;

            try {
              const response = await fetch("http://localhost:8000/detect_gesture", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                  landmarks: predictions.map((p) => p.landmarks),
                }),
              });
              const data = await response.json();
              lastGestureRef.current = data.gesture || lastGestureRef.current;
              const gestureEl = document.getElementById("gesture-text");
              if (gestureEl) {
                if (Array.isArray(data.gesture)) {
                  gestureEl.textContent = `Gestures: ${data.gesture.join(", ")}`;
                } else {
                  gestureEl.textContent = `Gesture: ${data.gesture}`;
                }
              }
            } catch (err) {
              console.error("⚠️ Backend API error:", err);
            }
          }
        }

        processingRef.current = false;
      } else {
        // Draw last landmarks
        lastPredictionsRef.current.forEach((hand) => {
          hand.landmarks.forEach(([x, y]) => {
            ctx.beginPath();
            ctx.arc(x, y, 5, 0, 2 * Math.PI);
            ctx.fillStyle = "red";
            ctx.fill();
          });
        });

        // Freeze gesture text after capture stops
        const gestureEl = document.getElementById("gesture-text");
        if (gestureEl)
          gestureEl.textContent = `Gesture: ${lastGestureRef.current}`;
      }

      animationId = requestAnimationFrame(processFrame);
    };

    loadModelAndCamera();

    return () => {
      if (animationId) cancelAnimationFrame(animationId);
      if (stream) stream.getTracks().forEach((track) => track.stop());
    };
  }, []);

  return (
    <div>
      <video ref={videoRef} style={{ display: "none" }} />
      <canvas
        ref={canvasRef}
        style={{ border: "1px solid black", width: 320, height: 240 }}
      />
      <div
        id="gesture-text"
        style={{ fontSize: "20px", marginTop: "10px", fontWeight: "bold" }}
      ></div>
    </div>
  );
}
