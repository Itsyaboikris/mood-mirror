import { useCallback, useEffect, useRef, useState } from "react";
import { FaceDetector, FilesetResolver, GestureRecognizer } from "@mediapipe/tasks-vision";
import styles from "./App.module.css";

const EMOJI = {
  happy: "😊", sad: "😢", angry: "😠", surprised: "😲",
  confused: "😕", disgusted: "🤢", fearful: "😨", neutral: "😐",
};

const AUTO_INTERVAL_MS = 5000;
const HISTORY_MAX = 8;
const MAX_ANALYZE_FACES = 2;
const MAX_OBJECTS = 3;
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
};

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
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    if (!isMostlyBlackFrame(canvas)) {
      return canvas.toDataURL("image/jpeg", 0.85);
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
  const [history, setHistory] = useState([]);

  useEffect(() => {
    const mode = PERFORMANCE_MODES[performanceMode] || PERFORMANCE_MODES.balanced;
    setFaceBoxesOn(mode.face);
    setGestureOn(mode.gesture);
    setObjectOn(mode.object);
  }, [performanceMode]);

  useEffect(() => {
    navigator.mediaDevices
      .getUserMedia({ video: { width: 640, height: 480 }, audio: false })
      .then((stream) => {
        if (videoRef.current) videoRef.current.srcObject = stream;
      })
      .catch((err) => setStatus({ text: `Camera error: ${err.message}`, state: "error" }));
    return () => videoRef.current?.srcObject?.getTracks().forEach((t) => t.stop());
  }, []);

  useEffect(() => {
    let cancelled = false;

    const initDetector = async () => {
      try {
        const fileset = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );
        const detector = await FaceDetector.createFromOptions(fileset, {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          minDetectionConfidence: 0.35,
        });
        if (!cancelled) {
          detectorRef.current = detector;
        }

        const gesture = await GestureRecognizer.createFromOptions(fileset, {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          numHands: 1,
        });
        if (!cancelled) {
          gestureRef.current = gesture;
        }

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
          setDetectorInfo("Detector: unavailable");
          setStatus({ text: `Face detection unavailable: ${err.message}`, state: "error" });
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
          ctx.fillRect(box.x, box.y, box.width, box.height);
          ctx.strokeRect(box.x, box.y, box.width, box.height);
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
            const topGesture = result?.gestures?.[0]?.[0]?.categoryName || null;
            const accepted = topGesture && ALLOWED_GESTURES.has(topGesture) ? topGesture : null;
            setCurrentGesture(accepted ? GESTURE_LABELS[accepted] : "None");
            handLandmarksRef.current = result?.landmarks?.[0] || [];
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
            const px = lm.x * w;
            const py = lm.y * h;
            minX = Math.min(minX, lm.x);
            minY = Math.min(minY, lm.y);
            maxX = Math.max(maxX, lm.x);
            maxY = Math.max(maxY, lm.y);
            ctx.beginPath();
            ctx.arc(px, py, 2.4, 0, Math.PI * 2);
            ctx.fill();
          }
          const bx = minX * w;
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
                .filter((p) => p.score >= 0.35)
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
          ctx.strokeStyle = "#f59e0b";
          ctx.fillStyle = "rgba(245, 158, 11, 0.12)";
          ctx.fillRect(sx, sy, sw, sh);
          ctx.strokeRect(sx, sy, sw, sh);
          const label = `${obj.class} ${Math.round(obj.score * 100)}%`;
          ctx.font = "12px sans-serif";
          const textW = ctx.measureText(label).width;
          const textX = sx;
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
          setSelectedModel(data.models[0]);
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
      const faceBoxes = (boxesRef.current || []).slice(0, MAX_ANALYZE_FACES);
      const regions = faceBoxes.length > 0 ? faceBoxes : [{ nx: 0, ny: 0, nw: 1, nh: 1 }];
      setStatus({ text: `Analyzing ${regions.length} face(s) with ${selectedModel}…`, state: "analyzing" });

      const cropToDataUrl = (r) => {
        const cropCanvas = document.createElement("canvas");
        const sx = clamp(Math.floor(r.nx * video.videoWidth), 0, video.videoWidth - 1);
        const sy = clamp(Math.floor(r.ny * video.videoHeight), 0, video.videoHeight - 1);
        const sw = clamp(Math.floor(r.nw * video.videoWidth), 1, video.videoWidth - sx);
        const sh = clamp(Math.floor(r.nh * video.videoHeight), 1, video.videoHeight - sy);
        cropCanvas.width = sw;
        cropCanvas.height = sh;
        cropCanvas.getContext("2d").drawImage(video, sx, sy, sw, sh, 0, 0, sw, sh);
        return cropCanvas.toDataURL("image/jpeg", 0.85);
      };

      const requests = regions.map(async (region, idx) => {
        const faceDataUrl = cropToDataUrl(region);
        const res = await fetch("/api/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image: faceDataUrl, model: selectedModel }),
          signal: AbortSignal.timeout(45_000),
        });
        const result = await safeJson(res);
        return {
          idx,
          ok: !!result.success,
          emotion: result.emotion || "unknown",
          description: result.description || "",
          confidence: Number(result.confidence ?? 50),
          error: result.error || "",
          snapshot: faceDataUrl,
        };
      });

      const results = await Promise.all(requests);
      const good = results.filter((r) => r.ok);
      setPersonResults(good);
      setAnalysisLatencyMs(Math.round(performance.now() - startedAt));

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
        setStatus({
          text: `Detected ${good.length} face(s) — top: ${primary.emotion.toUpperCase()} (${entry.confidence}%)`,
          state: "normal",
        });
      } else {
        const firstError = results[0]?.error || "Analysis failed";
        setStatus({ text: firstError, state: "error" });
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
  const statusClass =
    status.state === "error" ? styles.statusError : status.state === "analyzing" ? styles.statusAnalyzing : "";

  return (
    <div className={styles.app}>
      <header className={styles.header}>
        <span className={styles.headerTitle}>Mood Mirror AI</span>
      </header>

      <div className={`${styles.status} ${statusClass}`}>{status.text}</div>

      <main className={styles.main}>
        <section className={styles.panel}>
          <p className={styles.panelTitle}>Live Feed</p>
          <div className={styles.videoWrapper}>
            <video ref={videoRef} autoPlay playsInline muted className={styles.video} />
            <canvas ref={overlayRef} className={styles.faceOverlay} />
            <div className={styles.livePill}>
              <span className={styles.liveDot} />LIVE
            </div>
            <div className={styles.faceDebug}>
              {detectorInfo} | Faces: {faceCount}
            </div>
            <div className={styles.gestureDebug}>
              Gesture: {currentGesture}
            </div>
            <div className={styles.metricsDebug}>
              FPS: {Math.round(detectionFps)} | Latency: {analysisLatencyMs}ms | Objects: {objects.length}
            </div>
          </div>
        </section>

        <div className={styles.resultPanel}>
          <div className={styles.snapshotPanel}>
            <p className={styles.panelTitle}>Last Capture</p>
            <div className={styles.snapshotWrapper}>
              {currentSnapshot ? (
                <img src={currentSnapshot} alt="snapshot" className={styles.snapshot} />
              ) : (
                <div className={styles.snapshotPlaceholder}>
                  <span className={styles.placeholderIcon}>📸</span>
                  No capture yet
                </div>
              )}
            </div>
          </div>

          <div className={styles.emotionCard}>
            {analyzing ? (
              <div className={styles.analyzingIndicator}>
                <div className={styles.spinner} />
                Analyzing your expression…
              </div>
            ) : currentEmotion ? (
              <>
                <span key={currentEmotion} className={styles.emotionEmoji}>
                  {emoji}
                </span>
                <div className={styles.emotionWord}>{currentEmotion}</div>
                {currentDesc && <p className={styles.emotionDesc}>"{currentDesc}"</p>}
                <div className={styles.confWrap}>
                  <div className={styles.confLabel}>Confidence {currentConfidence}%</div>
                  <div className={styles.confTrack}>
                    <div className={styles.confFill} style={{ width: `${currentConfidence}%` }} />
                  </div>
                </div>
              </>
            ) : (
              <div className={styles.emotionPlaceholder}>🎭</div>
            )}
          </div>

          {personResults.length > 1 && (
            <div className={styles.peoplePanel}>
              <p className={styles.panelTitle}>Multi-person Results</p>
              <div className={styles.peopleList}>
                {personResults.map((p, idx) => (
                  <div key={`${p.idx}-${idx}`} className={styles.personRow}>
                    <span className={styles.personTag}>P{idx + 1}</span>
                    <span className={styles.personEmotion}>{p.emotion}</span>
                    <span className={styles.personConf}>{Math.round(p.confidence)}%</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {objects.length > 0 && (
            <div className={styles.peoplePanel}>
              <p className={styles.panelTitle}>Objects</p>
              <div className={styles.peopleList}>
                {objects.map((o, idx) => (
                  <div key={`${o.class}-${idx}`} className={styles.personRow}>
                    <span className={styles.personTag}>O{idx + 1}</span>
                    <span className={styles.personEmotion}>{o.class}</span>
                    <span className={styles.personConf}>{Math.round(o.score * 100)}%</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </main>

      <div className={styles.controls}>
        <button className={styles.btnPrimary} onClick={analyze} disabled={analyzing}>
          📸 Capture &amp; Analyze
        </button>
        <button className={`${styles.btnToggle} ${autoOn ? styles.btnToggleOn : ""}`} onClick={() => setAutoOn((v) => !v)}>
          {autoOn ? "⏹ Auto: ON" : "▶ Auto: OFF"}
        </button>
        <button className={`${styles.btnToggle} ${faceBoxesOn ? styles.btnToggleOn : ""}`} onClick={() => setFaceBoxesOn((v) => !v)}>
          {faceBoxesOn ? "🟦 Face Boxes: ON" : "◻ Face Boxes: OFF"}
        </button>
        <button className={`${styles.btnToggle} ${gestureOn ? styles.btnToggleOn : ""}`} onClick={() => setGestureOn((v) => !v)}>
          {gestureOn ? "🖐 Gestures: ON" : "🖐 Gestures: OFF"}
        </button>
        <button className={`${styles.btnToggle} ${objectOn ? styles.btnToggleOn : ""}`} onClick={() => setObjectOn((v) => !v)}>
          {objectOn ? "📦 Objects: ON" : "📦 Objects: OFF"}
        </button>
        <span className={styles.modelLabel}>Mode:</span>
        <select
          className={styles.select}
          value={performanceMode}
          onChange={(e) => setPerformanceMode(e.target.value)}
        >
          <option value="balanced">Balanced</option>
          <option value="fast">Fast</option>
          <option value="object">Object Demo</option>
        </select>
        <span className={styles.modelLabel}>Model:</span>
        <select className={styles.select} value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}>
          {models.map((m) => (
            <option key={m} value={m}>
              {m}
            </option>
          ))}
        </select>
      </div>

      <div className={styles.historySection}>
        <span className={styles.historyLabel}>Recent</span>
        {history.length === 0 ? (
          <span className={styles.historyEmpty}>Captures will appear here after you analyze</span>
        ) : (
          <div className={styles.historyStrip}>
            {history.map((entry, i) => (
              <div key={entry.id} className={`${styles.historyItem} ${i === 0 ? styles.newest : ""}`}>
                <img src={entry.snapshot} alt={entry.emotion} className={styles.historyImg} />
                <div className={styles.historyMeta}>
                  <span className={styles.historyEmoji}>{EMOJI[entry.emotion?.toLowerCase()] ?? "🤔"}</span>
                  <span className={styles.historyEmotion}>{entry.emotion}</span>
                  <span className={styles.historyTime}>{entry.time}</span>
                </div>
                <div className={styles.historyConfidence}>Conf: {entry.confidence ?? 0}% · Faces: {entry.people ?? 1}</div>
                <div className={styles.historyGesture}>{entry.gesture || "None"}</div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
