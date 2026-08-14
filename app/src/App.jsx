import { useCallback, useEffect, useRef, useState } from "react";
import { FaceDetector, FilesetResolver, GestureRecognizer } from "@mediapipe/tasks-vision";
import styles from "./App.module.css";

const EMOJI = {
  happy: "😄", sad: "😢", angry: "😠", surprised: "😲",
  confused: "🤔", disgusted: "🤢", fearful: "😨", neutral: "😐",
};

const EMOTION_COLORS = {
  happy:     "#FFCB2E",
  sad:       "#2FA8FF",
  angry:     "#FF3D7F",
  surprised: "#FF8A3D",
  confused:  "#8C6BFF",
  disgusted: "#A6E22E",
  fearful:   "#FF8A3D",
  neutral:   "#FF3D7F",
};

const AUTO_INTERVAL_MS = 5000;
const HISTORY_MAX = 10;
const MAX_OBJECTS = 10;
const OBJECT_UNCERTAIN_THRESHOLD = 0.45;
const PERFORMANCE_MODES = {
  balanced: { label: "Balanced", face: true, gesture: true, object: true, faceMs: 100, gestureMs: 150, objectMs: 300 },
  fast: { label: "Fast", face: true, gesture: true, object: false, faceMs: 80, gestureMs: 140, objectMs: 999999 },
  object: { label: "Object Demo", face: false, gesture: false, object: true, faceMs: 999999, gestureMs: 999999, objectMs: 180 },
};
const FALLBACK_MODELS = ["llama3.2-vision", "llava:7b", "llava:13b", "minicpm-v"];
const ALLOWED_GESTURES = new Set([
  "Thumb_Up",
  "Thumb_Down",
  "Victory",
  "Open_Palm",
  "Closed_Fist",
  "Pointing_Up",
  "ILoveYou",
]);
const GESTURE_LABELS = {
  Thumb_Up: "👍 Thumbs Up",
  Thumb_Down: "👎 Thumbs Down",
  Victory: "✌️ Victory",
  Open_Palm: "👋 Open Palm",
  Closed_Fist: "✊ Fist",
  Pointing_Up: "☝️ Pointing Up",
  ILoveYou: "🤟 I Love You",
  Finger_Heart: "🫰 Finger Heart",
};
const GESTURE_MIN_SCORE = 0.72;
const GESTURE_CONFIRM_FRAMES = 3;
const GESTURE_CLEAR_FRAMES = 4;
const GESTURE_REACTION_HOLD_MS = 2000;

async function safeJson(response) {
  const text = await response.text();
  if (!text) throw new Error(`Empty response from ${response.url} (${response.status} ${response.statusText})`);
  try {
    return JSON.parse(text);
  } catch {
    throw new Error(`Invalid JSON from ${response.url} (${response.status}): ${text.slice(0, 160)}`);
  }
}

function timeLabel() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function distance2d(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

function angleDeg(a, b, c) {
  const abx = a.x - b.x;
  const aby = a.y - b.y;
  const cbx = c.x - b.x;
  const cby = c.y - b.y;
  const dot = abx * cbx + aby * cby;
  const mag1 = Math.sqrt(abx * abx + aby * aby);
  const mag2 = Math.sqrt(cbx * cbx + cby * cby);
  if (!mag1 || !mag2) return 180;
  const cos = clamp(dot / (mag1 * mag2), -1, 1);
  return (Math.acos(cos) * 180) / Math.PI;
}

function isFingerHeartGesture(landmarks) {
  if (!Array.isArray(landmarks) || landmarks.length < 21) return false;
  const handSize = Math.max(distance2d(landmarks[0], landmarks[9]), 0.001);
  const wrist = landmarks[0];

  const indexTip = landmarks[8];
  const middleTip = landmarks[12];
  const ringTip = landmarks[16];
  const pinkyTip = landmarks[20];

  const indexPip = landmarks[6];
  const middlePip = landmarks[10];
  const ringPip = landmarks[14];
  const pinkyPip = landmarks[18];

  const indexExtended = distance2d(indexTip, wrist) > distance2d(indexPip, wrist) * 1.05;
  const indexNotFullyStraight = angleDeg(landmarks[5], landmarks[6], landmarks[8]) > 120;
  const middleCurled = distance2d(middleTip, wrist) < distance2d(middlePip, wrist) * 1.12;
  const ringCurled = distance2d(ringTip, wrist) < distance2d(ringPip, wrist) * 1.12;
  const pinkyCurled = distance2d(pinkyTip, wrist) < distance2d(pinkyPip, wrist) * 1.15;
  const thumbBent = angleDeg(landmarks[2], landmarks[3], landmarks[4]) < 170;

  const thumbToIndex = Math.min(
    distance2d(landmarks[4], landmarks[6]),
    distance2d(landmarks[4], landmarks[7]),
    distance2d(landmarks[4], landmarks[8])
  );
  const thumbNearIndex = thumbToIndex < handSize * 0.72;
  const curlCount = [middleCurled, ringCurled, pinkyCurled].filter(Boolean).length;

  return indexExtended && indexNotFullyStraight && thumbBent && thumbNearIndex && curlCount >= 2;
}

function isMostlyBlackFrame(canvas) {
  const ctx = canvas.getContext("2d");
  // Sample a small region for cheap brightness check
  const sampleW = Math.min(32, canvas.width);
  const sampleH = Math.min(32, canvas.height);
  if (sampleW <= 0 || sampleH <= 0) return true;
  const { data } = ctx.getImageData(0, 0, sampleW, sampleH);
  let sum = 0;
  const pixels = sampleW * sampleH;
  for (let i = 0; i < data.length; i += 4) {
    // perceived luminance approximation
    sum += data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114;
  }
  const avg = sum / pixels;
  return avg < 12;
}

async function captureStableFrame(video, maxAttempts = 3) {
  for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
    if (video.readyState < 2 || !video.videoWidth || !video.videoHeight) {
      await new Promise((r) => setTimeout(r, 80));
      continue;
    }
    const maxSide = 640;
    const scale = Math.min(1, maxSide / Math.max(video.videoWidth, video.videoHeight));
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
    canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
    if (!isMostlyBlackFrame(canvas)) {
      return canvas.toDataURL("image/jpeg", 0.7);
    }
    await new Promise((r) => setTimeout(r, 80));
  }
  return null;
}

export default function App() {
  const videoRef = useRef(null);
  const overlayRef = useRef(null);
  const detectorRef = useRef(null);
  const gestureRef = useRef(null);
  const objectDetectorRef = useRef(null);
  const rafRef = useRef(null);
  const lastDetectTsRef = useRef(0);
  const lastGestureTsRef = useRef(0);
  const lastObjectTsRef = useRef(0);
  const lastFrameTsRef = useRef(0);
  const boxesRef = useRef([]);
  const handLandmarksRef = useRef([]);
  const objectBoxesRef = useRef([]);
  const analyzingRef = useRef(false);
  const autoTimerRef = useRef(null);
  const lastFxGestureRef = useRef(null);
  const lastFxTsRef = useRef(0);
  const reactionGestureRef = useRef(null);
  const reactionGestureStartTsRef = useRef(0);
  const gestureCandidateRef = useRef(null);
  const gestureCandidateFramesRef = useRef(0);
  const gestureMissFramesRef = useRef(0);
  const confirmedGestureRef = useRef(null);

  const [status, setStatus] = useState({ text: "Checking Ollama…", state: "normal" });
  const [models, setModels] = useState(FALLBACK_MODELS);
  const [selectedModel, setSelectedModel] = useState(FALLBACK_MODELS[0]);
  const [analyzing, setAnalyzing] = useState(false);
  const [autoOn, setAutoOn] = useState(false);
  const [faceBoxesOn, setFaceBoxesOn] = useState(true);
  const [gestureOn, setGestureOn] = useState(true);
  const [objectOn, setObjectOn] = useState(true);
  const [performanceMode, setPerformanceMode] = useState("balanced");
  const [detectorInfo, setDetectorInfo] = useState("Detector: loading…");
  const [faceCount, setFaceCount] = useState(0);
  const [currentGesture, setCurrentGesture] = useState("None");
  const [detectionFps, setDetectionFps] = useState(0);
  const [analysisLatencyMs, setAnalysisLatencyMs] = useState(0);

  const [currentSnapshot, setCurrentSnapshot] = useState(null);
  const [currentEmotion, setCurrentEmotion] = useState(null);
  const [currentDesc, setCurrentDesc] = useState(null);
  const [currentConfidence, setCurrentConfidence] = useState(0);
  const [personResults, setPersonResults] = useState([]);
  const [objects, setObjects] = useState([]);
  const [moodColor, setMoodColor] = useState("#FF3D7F");
  const [toastMsg, setToastMsg] = useState("");
  const [toastVisible, setToastVisible] = useState(false);

  const toastTimerRef = useRef(null);
  const feedFrameRef = useRef(null);

  const showToast = useCallback((msg) => {
    setToastMsg(msg);
    setToastVisible(true);
    clearTimeout(toastTimerRef.current);
    toastTimerRef.current = setTimeout(() => setToastVisible(false), 1600);
  }, []);

  const burstConfetti = useCallback((color) => {
    const el = feedFrameRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const colors = [color, "#FFCB2E", "#2FA8FF", "#A6E22E"];
    for (let i = 0; i < 16; i++) {
      const piece = document.createElement("div");
      piece.style.cssText = `position:fixed;width:8px;height:8px;border-radius:2px;z-index:9999;pointer-events:none;background:${colors[i % colors.length]};left:${rect.left + rect.width / 2}px;top:${rect.top + rect.height / 2}px`;
      document.body.appendChild(piece);
      const angle = Math.random() * Math.PI * 2;
      const dist = 60 + Math.random() * 140;
      const dx = Math.cos(angle) * dist;
      const dy = Math.sin(angle) * dist - 40;
      piece.animate(
        [
          { transform: "translate(0,0) rotate(0deg)", opacity: 1 },
          { transform: `translate(${dx}px,${dy + 160}px) rotate(${Math.random() * 360}deg)`, opacity: 0 },
        ],
        { duration: 900 + Math.random() * 400, easing: "cubic-bezier(.2,.8,.3,1)" }
      );
      setTimeout(() => piece.remove(), 1400);
    }
  }, []);
  const [history, setHistory] = useState([]);
  const [gestureFx, setGestureFx] = useState(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [historyDetail, setHistoryDetail] = useState(null);

  useEffect(() => {
    const mode = PERFORMANCE_MODES[performanceMode] || PERFORMANCE_MODES.balanced;
    setFaceBoxesOn(mode.face);
    setGestureOn(mode.gesture);
    setObjectOn(mode.object);
  }, [performanceMode]);

  useEffect(() => {
    let activeStream = null;

    const startCamera = async () => {
      try {
        const preferredConstraints = {
          video: {
            width: { ideal: 1920, min: 1280 },
            height: { ideal: 1080, min: 720 },
            frameRate: { ideal: 30, max: 60 },
            facingMode: "user",
          },
          audio: false,
        };
        const fallbackConstraints = {
          video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
          audio: false,
        };

        let stream;
        try {
          stream = await navigator.mediaDevices.getUserMedia(preferredConstraints);
        } catch {
          stream = await navigator.mediaDevices.getUserMedia(fallbackConstraints);
        }
        activeStream = stream;

        const [track] = stream.getVideoTracks();
        if (track?.applyConstraints) {
          try {
            await track.applyConstraints({
              advanced: [
                { focusMode: "continuous" },
                { exposureMode: "continuous" },
                { whiteBalanceMode: "continuous" },
              ],
            });
          } catch {
            // Ignore unsupported camera controls.
          }
        }

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        setStatus({ text: `Camera error: ${err.message}`, state: "error" });
      }
    };

    startCamera();
    return () => activeStream?.getTracks().forEach((t) => t.stop());
  }, []);

  useEffect(() => {
    let cancelled = false;

    const initDetector = async () => {
      try {
        const fileset = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.34/wasm"
        );
        const faceDetectorOptions = (delegate) => ({
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
            delegate,
          },
          runningMode: "VIDEO",
          minDetectionConfidence: 0.35,
        });
        let detector;
        try {
          detector = await FaceDetector.createFromOptions(fileset, faceDetectorOptions("GPU"));
        } catch {
          detector = await FaceDetector.createFromOptions(fileset, faceDetectorOptions("CPU"));
        }
        if (!cancelled) detectorRef.current = detector;

        const gestureOptions = (delegate) => ({
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
            delegate,
          },
          runningMode: "VIDEO",
          numHands: 1,
        });
        let gesture;
        try {
          gesture = await GestureRecognizer.createFromOptions(fileset, gestureOptions("GPU"));
        } catch {
          gesture = await GestureRecognizer.createFromOptions(fileset, gestureOptions("CPU"));
        }
        if (!cancelled) {
          gestureRef.current = gesture;
          setDetectorInfo("Face+Gesture: MediaPipe");
        }
      } catch (err) {
        if (!cancelled) {
          setDetectorInfo("Detector: unavailable");
          setStatus({ text: `Face detection unavailable: ${err.message}`, state: "error" });
        }
      }

      try {
        const tf = await import("@tensorflow/tfjs");
        await tf.ready();
        const cocoSsd = await import("@tensorflow-models/coco-ssd");
        const objectDetector = await cocoSsd.load({ base: "lite_mobilenet_v2" });
        if (!cancelled) {
          objectDetectorRef.current = objectDetector;
          setDetectorInfo("Face+Gesture+Objects: MediaPipe+TF");
        }
      } catch (err) {
        if (!cancelled) {
          setDetectorInfo((prev) => `${prev} (objects unavailable)`);
          console.warn("Object detector failed to load:", err.message);
        }
      }
    };

    const renderLoop = () => {
      const video = videoRef.current;
      const canvas = overlayRef.current;
      if (!canvas || !video) {
        rafRef.current = requestAnimationFrame(renderLoop);
        return;
      }

      const w = video.clientWidth || 0;
      const h = video.clientHeight || 0;
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }

      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, w, h);
      const mirrorX = (x, width = 0) => w - x - width;

      if (lastFrameTsRef.current > 0) {
        const delta = performance.now() - lastFrameTsRef.current;
        if (delta > 0) setDetectionFps((prev) => prev * 0.85 + (1000 / delta) * 0.15);
      }
      lastFrameTsRef.current = performance.now();

      const mode = PERFORMANCE_MODES[performanceMode] || PERFORMANCE_MODES.balanced;

      if (faceBoxesOn && video.videoWidth && detectorRef.current) {
        const now = performance.now();
        if (now - lastDetectTsRef.current > mode.faceMs) {
          lastDetectTsRef.current = now;
          try {
            const result = detectorRef.current.detectForVideo(video, now);
            const detections = result?.detections || [];
            const boxes = detections
              .map((d) => d?.boundingBox)
              .filter(Boolean)
              .map((b) => ({
                nx: b.originX / video.videoWidth,
                ny: b.originY / video.videoHeight,
                nw: b.width / video.videoWidth,
                nh: b.height / video.videoHeight,
                x: (b.originX / video.videoWidth) * w,
                y: (b.originY / video.videoHeight) * h,
                width: (b.width / video.videoWidth) * w,
                height: (b.height / video.videoHeight) * h,
              }));
            boxesRef.current = boxes;
            setFaceCount(boxes.length);
          } catch {
            // Ignore per-frame detection failures
          }
        }

        ctx.lineWidth = 2;
        ctx.strokeStyle = "#06b6d4";
        ctx.fillStyle = "rgba(6, 182, 212, 0.12)";
        for (const box of boxesRef.current) {
          const mx = mirrorX(box.x, box.width);
          ctx.fillRect(mx, box.y, box.width, box.height);
          ctx.strokeRect(mx, box.y, box.width, box.height);
        }
      } else if (!faceBoxesOn && faceCount !== 0) {
        setFaceCount(0);
      }

      if (gestureOn && video.videoWidth && gestureRef.current) {
        const now = performance.now();
        if (now - lastGestureTsRef.current > mode.gestureMs) {
          lastGestureTsRef.current = now;
          try {
            const result = gestureRef.current.recognizeForVideo(video, now);
            const topCategory = result?.gestures?.[0]?.[0] || null;
            const topGesture = topCategory?.categoryName || null;
            const topScore = Number(topCategory?.score ?? topCategory?.categoryScore ?? 0);
            const accepted =
              topGesture && topScore >= GESTURE_MIN_SCORE && ALLOWED_GESTURES.has(topGesture) ? topGesture : null;
            const landmarks = result?.landmarks?.[0] || [];
            const customGesture = isFingerHeartGesture(landmarks) ? "Finger_Heart" : null;
            const rawGesture = customGesture || accepted;
            if (rawGesture) {
              gestureMissFramesRef.current = 0;
              if (gestureCandidateRef.current === rawGesture) {
                gestureCandidateFramesRef.current += 1;
              } else {
                gestureCandidateRef.current = rawGesture;
                gestureCandidateFramesRef.current = 1;
              }
              if (gestureCandidateFramesRef.current >= GESTURE_CONFIRM_FRAMES) {
                confirmedGestureRef.current = rawGesture;
              }
            } else {
              gestureMissFramesRef.current += 1;
              if (gestureMissFramesRef.current >= GESTURE_CLEAR_FRAMES) {
                gestureCandidateRef.current = null;
                gestureCandidateFramesRef.current = 0;
                confirmedGestureRef.current = null;
              }
            }
            const detectedGesture = confirmedGestureRef.current;
            setCurrentGesture(detectedGesture ? GESTURE_LABELS[detectedGesture] : "None");
            if (detectedGesture) {
              if (reactionGestureRef.current !== detectedGesture) {
                reactionGestureRef.current = detectedGesture;
                reactionGestureStartTsRef.current = now;
              }
            } else {
              reactionGestureRef.current = null;
              reactionGestureStartTsRef.current = 0;
            }

            const heldLongEnough =
              detectedGesture &&
              reactionGestureRef.current === detectedGesture &&
              reactionGestureStartTsRef.current > 0 &&
              now - reactionGestureStartTsRef.current >= GESTURE_REACTION_HOLD_MS;

            const canTriggerFx =
              detectedGesture &&
              heldLongEnough &&
              (
                detectedGesture === "Victory" ||
                detectedGesture === "Thumb_Up" ||
                detectedGesture === "Thumb_Down" ||
                detectedGesture === "ILoveYou" ||
                detectedGesture === "Finger_Heart"
              ) &&
              (detectedGesture !== lastFxGestureRef.current || now - lastFxTsRef.current > 1600);
            if (canTriggerFx) {
              lastFxGestureRef.current = detectedGesture;
              lastFxTsRef.current = now;
              const glyph =
                detectedGesture === "Victory"
                  ? "🎈"
                  : detectedGesture === "Thumb_Up"
                    ? "🎉"
                    : detectedGesture === "Thumb_Down"
                      ? "😢"
                      : detectedGesture === "Finger_Heart"
                        ? "💖"
                        : "✨";
              setGestureFx({
                id: Date.now(),
                glyph,
                pieces: Array.from({ length: 12 }, (_, i) => ({
                  id: i,
                  left: 8 + Math.random() * 84,
                  delay: Math.random() * 0.25,
                  duration: 1.4 + Math.random() * 0.8,
                })),
              });
            }
            handLandmarksRef.current = landmarks;
          } catch {
            // Ignore per-frame gesture failures
          }
        }

        if (handLandmarksRef.current.length > 0) {
          ctx.fillStyle = "#4f46e5";
          let minX = 1;
          let minY = 1;
          let maxX = 0;
          let maxY = 0;
          for (const lm of handLandmarksRef.current) {
            const px = mirrorX(lm.x * w);
            const py = lm.y * h;
            minX = Math.min(minX, lm.x);
            minY = Math.min(minY, lm.y);
            maxX = Math.max(maxX, lm.x);
            maxY = Math.max(maxY, lm.y);
            ctx.beginPath();
            ctx.arc(px, py, 2.4, 0, Math.PI * 2);
            ctx.fill();
          }
          const bx = mirrorX(maxX * w);
          const by = minY * h;
          const bw = (maxX - minX) * w;
          const bh = (maxY - minY) * h;
          ctx.strokeStyle = "#4f46e5";
          ctx.lineWidth = 2;
          ctx.strokeRect(bx, by, bw, bh);
        }
      } else {
        handLandmarksRef.current = [];
        if (!gestureOn) setCurrentGesture("None");
      }

      if (objectOn && video.videoWidth && objectDetectorRef.current) {
        const now = performance.now();
        if (now - lastObjectTsRef.current > mode.objectMs) {
          lastObjectTsRef.current = now;
          objectDetectorRef.current
            .detect(video)
            .then((predictions) => {
              const top = (predictions || [])
                .filter((p) => p.score >= 0.30)
                .sort((a, b) => b.score - a.score)
                .slice(0, MAX_OBJECTS)
                .map((p) => ({
                  class: p.score < OBJECT_UNCERTAIN_THRESHOLD ? "Uncertain" : p.class,
                  score: p.score,
                  bbox: p.bbox,
                }));
              objectBoxesRef.current = top;
              setObjects(top);
            })
            .catch(() => {});
        }

        ctx.lineWidth = 2;
        for (const obj of objectBoxesRef.current) {
          const [x, y, bw, bh] = obj.bbox;
          const sx = (x / video.videoWidth) * w;
          const sy = (y / video.videoHeight) * h;
          const sw = (bw / video.videoWidth) * w;
          const sh = (bh / video.videoHeight) * h;
          const mx = mirrorX(sx, sw);
          ctx.strokeStyle = "#f59e0b";
          ctx.fillStyle = "rgba(245, 158, 11, 0.12)";
          ctx.fillRect(mx, sy, sw, sh);
          ctx.strokeRect(mx, sy, sw, sh);
          const label = `${obj.class} ${Math.round(obj.score * 100)}%`;
          ctx.font = "12px sans-serif";
          const textW = ctx.measureText(label).width;
          const textX = mx;
          const textY = Math.max(14, sy - 4);
          ctx.fillStyle = "#f59e0b";
          ctx.fillRect(textX, textY - 12, textW + 8, 14);
          ctx.fillStyle = "#111827";
          ctx.fillText(label, textX + 4, textY - 2);
        }
      } else {
        objectBoxesRef.current = [];
        if (!objectOn && objects.length) setObjects([]);
      }

      rafRef.current = requestAnimationFrame(renderLoop);
    };

    initDetector();
    rafRef.current = requestAnimationFrame(renderLoop);

    return () => {
      cancelled = true;
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      if (detectorRef.current?.close) detectorRef.current.close();
      if (gestureRef.current?.close) gestureRef.current.close();
    };
  }, [faceBoxesOn, gestureOn, objectOn, performanceMode]);

  useEffect(() => {
    fetch("/api/health")
      .then((r) => safeJson(r))
      .then((data) => {
        console.log("[health]", data.message);
        setStatus({
          text: data.ok ? "System ready." : "Ollama connection issue. Check server logs.",
          state: data.ok ? "normal" : "error",
        });
        if (data.models?.length) {
          setModels(data.models);
          const preferred =
            data.models.find((m) => m.includes("minicpm")) ||
            data.models.find((m) => m.includes("qwen")) ||
            data.models[0];
          setSelectedModel(preferred);
        }
      })
      .catch((err) => setStatus({ text: `Server error: ${err.message}`, state: "error" }));
  }, []);

  const analyze = useCallback(async () => {
    if (analyzingRef.current) return;
    const video = videoRef.current;
    if (!video?.videoWidth) return;

    analyzingRef.current = true;
    setAnalyzing(true);
    setCurrentEmotion(null);
    setCurrentDesc(null);
    setCurrentConfidence(0);
    setPersonResults([]);
    setStatus({ text: `Analyzing with ${selectedModel}…`, state: "analyzing" });

    const dataUrl = await captureStableFrame(video);
    if (!dataUrl) {
      setStatus({ text: "Camera frame not ready yet. Try again in a second.", state: "error" });
      analyzingRef.current = false;
      setAnalyzing(false);
      return;
    }
    setCurrentSnapshot(dataUrl);

    const startedAt = performance.now();

    try {
      const detectedCount = (boxesRef.current || []).length;
      setStatus({ text: `Analyzing ${detectedCount > 0 ? detectedCount : "all"} face(s) with ${selectedModel}…`, state: "analyzing" });

      const res = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: dataUrl, model: selectedModel }),
        signal: AbortSignal.timeout(120_000),
      });
      const result = await safeJson(res);
      setAnalysisLatencyMs(Math.round(performance.now() - startedAt));

      if (!result.success) {
        setStatus({ text: result.error || "Analysis failed", state: "error" });
        return;
      }

      const good = (result.faces || []).map((f, idx) => ({
        idx,
        ok: true,
        emotion: f.emotion || "unknown",
        description: f.description || "",
        confidence: Number(f.confidence ?? 50),
        error: "",
        snapshot: dataUrl,
      }));
      setPersonResults(good);

      if (good.length > 0) {
        const primary = good[0];
        const entry = {
          id: Date.now(),
          snapshot: dataUrl,
          emotion: primary.emotion,
          confidence: clamp(Math.round(primary.confidence), 0, 100),
          gesture: currentGesture,
          description: primary.description,
          time: timeLabel(),
          people: good.length,
        };
        setCurrentEmotion(primary.emotion);
        setCurrentDesc(primary.description);
        setCurrentConfidence(clamp(Math.round(primary.confidence), 0, 100));
        setHistory((prev) => [entry, ...prev].slice(0, HISTORY_MAX));
        const color = EMOTION_COLORS[primary.emotion] ?? "#FF3D7F";
        setMoodColor(color);
        burstConfetti(color);
        setStatus({
          text: `Detected ${good.length} face(s) — top: ${primary.emotion.toUpperCase()} (${entry.confidence}%)`,
          state: "normal",
        });
      } else {
        setStatus({ text: "No faces found in frame", state: "error" });
      }
    } catch (err) {
      setStatus({ text: err.message, state: "error" });
    } finally {
      analyzingRef.current = false;
      setAnalyzing(false);
    }
  }, [selectedModel, currentGesture]);

  useEffect(() => {
    if (autoOn) {
      analyze();
      autoTimerRef.current = setInterval(analyze, AUTO_INTERVAL_MS);
    } else {
      clearInterval(autoTimerRef.current);
    }
    return () => clearInterval(autoTimerRef.current);
  }, [autoOn, analyze]);

  const emoji = currentEmotion ? (EMOJI[currentEmotion.toLowerCase()] ?? "🤔") : null;
  const pillClass = status.state === "error" ? styles.statusPillError
    : status.state === "analyzing" ? styles.statusPillBusy : "";

  return (
    <div className={styles.app} style={{ "--mood": moodColor }}>

      {/* Background blobs */}
      <div className={styles.blob1} style={{ background: moodColor }} />
      <div className={styles.blob2} />

      {/* Header */}
      <header className={styles.header}>
        <div className={styles.brand}>
          <div className={styles.logoChip}>{emoji ?? "🪞"}</div>
          <div>
            <h1 className={styles.brandTitle}>Mood Mirror AI</h1>
            <div className={styles.brandTagline}>walk up. make a face. watch it guess.</div>
          </div>
        </div>
        <div className={styles.headerRight}>
          <span className={`${styles.statusPill} ${pillClass}`}>
            <span className={styles.statusDot} />
            {status.text}
          </span>
          <button
            className={`${styles.btn} ${styles.btnNav} ${styles.btnSettingsIcon}`}
            onClick={() => setSettingsOpen(true)}
            title="Settings"
          >
            ⚙
          </button>
        </div>
      </header>

      {/* Main grid */}
      <div className={styles.grid}>

        {/* Live feed */}
        <div className={`${styles.panel} ${styles.shadowPink}`}>
          <div className={styles.panelLabel}>📷 Live Feed</div>
          <div className={styles.feedFrame} ref={feedFrameRef} onClick={() => { if (!analyzing) analyze(); }}>
            <video ref={videoRef} autoPlay playsInline muted className={styles.video} />
            <canvas ref={overlayRef} className={styles.faceOverlay} />

            <div className={styles.liveBadge}>
              <span className={styles.pulseDot} />LIVE
            </div>

            <div className={styles.telemetry}>
              <span><span className={styles.telemetryVal}>{Math.round(detectionFps)}</span>fps</span>
              <span><span className={styles.telemetryVal}>{analysisLatencyMs}ms</span></span>
              <span><span className={styles.telemetryVal}>{objects.length}</span>obj</span>
            </div>

            {!currentEmotion && !analyzing && (
              <div className={styles.tapHint}>
                <span className={styles.tapHintEmoji}>👀</span>
                <span className={styles.tapHintCaption}>tap anywhere to read your mood</span>
              </div>
            )}

            <div className={styles.feedFooterLeft}>
              {detectorInfo} · Faces: {faceCount}
            </div>

            <div className={`${styles.gestureChip} ${currentGesture !== "None" ? styles.gestureChipHot : ""}`}>
              Gesture: {currentGesture}
            </div>

            {gestureFx && (
              <div key={gestureFx.id} className={styles.gestureFxLayer}>
                {gestureFx.pieces.map((p) => (
                  <span
                    key={`${gestureFx.id}-${p.id}`}
                    className={styles.gestureFxPiece}
                    style={{ left: `${p.left}%`, animationDelay: `${p.delay}s`, animationDuration: `${p.duration}s` }}
                  >
                    {gestureFx.glyph}
                  </span>
                ))}
              </div>
            )}
          </div>
          <div className={styles.feedCta}>
            <button
              className={`${styles.btn} ${styles.btnPrimary} ${styles.btnFeed}`}
              onClick={analyze}
              disabled={analyzing}
            >
              {analyzing ? <span className={styles.spinnerSm} /> : null}
              {analyzing ? "Reading…" : "👀 Read My Mood"}
            </button>
          </div>
        </div>

        {/* Result column — last capture + big mood result */}
        <div className={`${styles.panel} ${styles.shadowBlue} ${styles.resultPanel}`}>
          <div className={styles.panelLabel}>✨ Last Capture</div>
          <div className={styles.resultBody}>
            <div className={styles.lastCaptureFrame}>
              {currentSnapshot ? (
                <img src={currentSnapshot} alt="snapshot" className={styles.snapshot} />
              ) : (
                <div className={styles.emptyState}>
                  <span className={styles.emptyIcon}>📸</span>
                  <span className={styles.emptyCaption}>nothing yet — go make a face!</span>
                </div>
              )}
            </div>

            <div className={styles.moodResult} style={{ background: moodColor + "22" }}>
              {analyzing ? (
                <div className={styles.moodResultAnalyzing}>
                  <div className={styles.spinner} />
                  <span>Reading your mood…</span>
                </div>
              ) : currentEmotion ? (
                <>
                  <span className={styles.moodResultEmoji}>{emoji}</span>
                  <div className={styles.moodResultWord}>
                    {currentEmotion.charAt(0).toUpperCase() + currentEmotion.slice(1)}
                  </div>
                  {currentDesc && <p className={styles.moodResultDesc}>&ldquo;{currentDesc}&rdquo;</p>}
                  <div className={styles.moodConfBar}>
                    <div className={styles.moodConfTrack}>
                      <div className={styles.moodConfFill} style={{ width: `${currentConfidence}%`, background: moodColor }} />
                    </div>
                    <span className={styles.moodConfPct}>{currentConfidence}%</span>
                  </div>
                </>
              ) : (
                <div className={styles.moodResultPlaceholder}>
                  <span>🎭</span>
                  <span>your mood shows up here</span>
                </div>
              )}
            </div>

            {/* Multi-person results */}
            {personResults.length > 1 && (
              <div className={styles.multiPersonList}>
                <div className={styles.multiPersonLabel}>👥 {personResults.length} people</div>
                {personResults.map((p, idx) => (
                  <div key={`${p.idx}-${idx}`} className={styles.personRow}>
                    <span className={styles.personTag}>P{idx + 1}</span>
                    <span className={styles.personEmotion}>{p.emotion}</span>
                    <span className={styles.personConf}>{Math.round(p.confidence)}%</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

      </div>

      {/* History strip */}
      <div className={styles.historyBar}>
        <span className={styles.historyBarLabel}>
          🕘 Recent
          {history.length > 0 && <span className={styles.historyBarCount}>{history.length}</span>}
        </span>
        {history.length === 0 ? (
          <span className={styles.historyBarEmpty}>make a face to start your history</span>
        ) : (
          <div className={styles.historyStrip}>
            {history.map((entry) => (
              <button
                key={entry.id}
                type="button"
                className={styles.historyItem}
                onClick={() => setHistoryDetail(entry)}
              >
                {entry.snapshot ? (
                  <img src={entry.snapshot} alt={entry.emotion} className={styles.historyImg} />
                ) : (
                  <div
                    className={styles.historyImgFallback}
                    style={{ background: (EMOTION_COLORS[entry.emotion?.toLowerCase()] ?? "#FFCB2E") + "55" }}
                  >
                    {EMOJI[entry.emotion?.toLowerCase()] ?? "🤔"}
                  </div>
                )}
                <div className={styles.historyMeta}>
                  <span className={styles.historyEmotion}>
                    {EMOJI[entry.emotion?.toLowerCase()] ?? "🤔"} {entry.emotion}
                  </span>
                  <span className={styles.historyTime}>{entry.time}</span>
                </div>
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Toast */}
      <div className={`${styles.toast} ${toastVisible ? styles.toastShow : ""}`}>
        {toastMsg}
      </div>

      {/* History detail modal */}
      {historyDetail && (
        <div className={styles.modalOverlay} onClick={() => setHistoryDetail(null)}>
          <div
            className={styles.modalCard}
            onClick={(e) => e.stopPropagation()}
            style={{ "--mood": EMOTION_COLORS[historyDetail.emotion?.toLowerCase()] ?? "#FF3D7F" }}
          >
            <div className={styles.modalHeader}>
              <span>🕘 Capture detail</span>
              <button className={styles.settingsClose} onClick={() => setHistoryDetail(null)}>✕</button>
            </div>
            <div className={styles.modalBody}>
              <div className={styles.modalShot}>
                {historyDetail.snapshot ? (
                  <img src={historyDetail.snapshot} alt={historyDetail.emotion} />
                ) : (
                  <div className={styles.modalShotEmpty}>
                    {EMOJI[historyDetail.emotion?.toLowerCase()] ?? "🤔"}
                  </div>
                )}
              </div>
              <div className={styles.modalResult} style={{ background: (EMOTION_COLORS[historyDetail.emotion?.toLowerCase()] ?? "#FF3D7F") + "22" }}>
                <span className={styles.modalEmoji}>
                  {EMOJI[historyDetail.emotion?.toLowerCase()] ?? "🤔"}
                </span>
                <div className={styles.modalEmotion}>
                  {(historyDetail.emotion || "unknown").charAt(0).toUpperCase() + (historyDetail.emotion || "unknown").slice(1)}
                </div>
                {historyDetail.description && (
                  <p className={styles.modalDesc}>&ldquo;{historyDetail.description}&rdquo;</p>
                )}
                <div className={styles.moodConfBar}>
                  <div className={styles.moodConfTrack}>
                    <div
                      className={styles.moodConfFill}
                      style={{
                        width: `${historyDetail.confidence ?? 0}%`,
                        background: EMOTION_COLORS[historyDetail.emotion?.toLowerCase()] ?? "#FF3D7F",
                      }}
                    />
                  </div>
                  <span className={styles.moodConfPct}>{historyDetail.confidence ?? 0}%</span>
                </div>
                <div className={styles.modalMeta}>
                  <span>{historyDetail.time}</span>
                  {historyDetail.gesture && historyDetail.gesture !== "None" && (
                    <span>Gesture: {historyDetail.gesture}</span>
                  )}
                  {historyDetail.people > 1 && (
                    <span>{historyDetail.people} people</span>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Settings overlay */}
      {settingsOpen && (
        <div className={styles.settingsOverlay} onClick={() => setSettingsOpen(false)}>
          <div className={styles.settingsPanel} onClick={(e) => e.stopPropagation()}>
            <div className={styles.settingsHeader}>
              <span>⚙ Settings</span>
              <button className={styles.settingsClose} onClick={() => setSettingsOpen(false)}>✕</button>
            </div>
            <div className={styles.settingsBody}>

              <div className={styles.settingsSection}>
                <div className={styles.settingsSectionTitle}>Capture</div>
                <div className={styles.settingsToggles}>
                  <button
                    className={`${styles.settingsToggle} ${autoOn ? styles.settingsToggleOn : ""}`}
                    onClick={() => { setAutoOn((v) => !v); showToast(autoOn ? "Auto stopped" : "Auto every 5s"); }}
                  >
                    <span className={styles.settingsToggleDot} style={{ background: autoOn ? "#A6E22E" : "#ccc" }} />
                    {autoOn ? "Auto: ON (every 5s)" : "Auto: OFF"}
                  </button>
                </div>
              </div>

              <div className={styles.settingsSection}>
                <div className={styles.settingsSectionTitle}>Performance Mode</div>
                <select
                  className={styles.settingsSelect}
                  value={performanceMode}
                  onChange={(e) => { setPerformanceMode(e.target.value); showToast("Mode: " + e.target.options[e.target.selectedIndex].text); }}
                >
                  <option value="balanced">Balanced</option>
                  <option value="fast">Fast</option>
                  <option value="object">Object Demo</option>
                </select>
              </div>

              <div className={styles.settingsSection}>
                <div className={styles.settingsSectionTitle}>AI Model</div>
                <select
                  className={styles.settingsSelect}
                  value={selectedModel}
                  onChange={(e) => { setSelectedModel(e.target.value); showToast("Model: " + e.target.value); }}
                >
                  {models.map((m) => (
                    <option key={m} value={m}>{m}</option>
                  ))}
                </select>
              </div>

              <div className={styles.settingsSection}>
                <div className={styles.settingsSectionTitle}>Detection</div>
                <div className={styles.settingsToggles}>
                  <button
                    className={`${styles.settingsToggle} ${faceBoxesOn ? styles.settingsToggleOn : ""}`}
                    onClick={() => { setFaceBoxesOn((v) => !v); showToast((faceBoxesOn ? "Disabled " : "Enabled ") + "Faces"); }}
                  >
                    <span className={styles.settingsToggleDot} style={{ background: faceBoxesOn ? "#8C6BFF" : "#ccc" }} />
                    Faces
                  </button>
                  <button
                    className={`${styles.settingsToggle} ${gestureOn ? styles.settingsToggleOn : ""}`}
                    onClick={() => { setGestureOn((v) => !v); showToast((gestureOn ? "Disabled " : "Enabled ") + "Gestures"); }}
                  >
                    <span className={styles.settingsToggleDot} style={{ background: gestureOn ? "#FF8A3D" : "#ccc" }} />
                    Gestures
                  </button>
                  <button
                    className={`${styles.settingsToggle} ${objectOn ? styles.settingsToggleOn : ""}`}
                    onClick={() => { setObjectOn((v) => !v); showToast((objectOn ? "Disabled " : "Enabled ") + "Objects"); }}
                  >
                    <span className={styles.settingsToggleDot} style={{ background: objectOn ? "#2FA8FF" : "#ccc" }} />
                    Objects
                  </button>
                </div>
              </div>

            </div>
          </div>
        </div>
      )}

    </div>
  );
}
