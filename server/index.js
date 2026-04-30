import "dotenv/config";
import express from "express";
import cors from "cors";

const app = express();
const PORT = process.env.PORT || 5000;
const OLLAMA_BASE_URL = (process.env.OLLAMA_BASE_URL || "http://localhost:11434").replace(/\/$/, "");
let reqCounter = 0;

function logStep(reqId, step, details = "") {
  const ts = new Date().toISOString();
  const suffix = details ? ` | ${details}` : "";
  console.log(`[${ts}] [req:${reqId}] ${step}${suffix}`);
}

async function runStartupOllamaCheck() {
  const reqId = "startup";
  logStep(reqId, "startup.health.start", `ollama_url=${OLLAMA_BASE_URL}`);

  try {
    const tagsRes = await fetch(`${OLLAMA_BASE_URL}/api/tags`, {
      signal: AbortSignal.timeout(10_000),
    });
    logStep(reqId, "startup.health.tags_status", `status=${tagsRes.status}`);
    if (!tagsRes.ok) {
      const text = await tagsRes.text();
      logStep(reqId, "startup.health.tags_error", text.slice(0, 300));
      return;
    }

    const tags = await tagsRes.json();
    const models = (tags.models || []).map((m) => m.name);
    logStep(reqId, "startup.health.models", `count=${models.length} names=${models.join(", ")}`);

    const testModel = models.find((m) => m.includes("vision")) || models[0];
    if (!testModel) {
      logStep(reqId, "startup.health.skip_generate", "no models available");
      return;
    }

    logStep(reqId, "startup.generate.start", `model=${testModel}`);
    const genRes = await fetch(`${OLLAMA_BASE_URL}/api/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: testModel,
        prompt: "Reply with exactly: ok",
        stream: false,
      }),
      signal: AbortSignal.timeout(20_000),
    });
    logStep(reqId, "startup.generate.status", `status=${genRes.status}`);

    if (!genRes.ok) {
      const text = await genRes.text();
      logStep(reqId, "startup.generate.error", text.slice(0, 300));
      return;
    }

    const data = await genRes.json();
    const preview = String(data.response || "").trim().slice(0, 80);
    logStep(reqId, "startup.generate.ok", `response_preview=${preview}`);
  } catch (err) {
    logStep(reqId, "startup.health.exception", String(err?.message || err));
  }
}

const EMOTION_PROMPT =
  "Look at this image. Identify every visible face and analyse the expression of each person.\n" +
  'Respond with a JSON object with a single key "faces" whose value is an array.\n' +
  "Each element in the array represents one person and must contain exactly three keys:\n" +
  '  "emotion": a single word for the dominant emotion ' +
  "(happy, sad, angry, surprised, confused, disgusted, fearful, neutral),\n" +
  '  "description": one short sentence describing what you observe about that person,\n' +
  '  "confidence": integer from 0 to 100 representing your confidence in the emotion label.\n' +
  'If there are no visible faces return {"faces": []}.\n' +
  "Return only the JSON object, no extra text.";

app.use(cors());
app.use(express.json({ limit: "10mb" }));
app.use((req, res, next) => {
  const reqId = ++reqCounter;
  req.reqId = reqId;
  const start = Date.now();
  logStep(reqId, "incoming", `${req.method} ${req.originalUrl}`);
  res.on("finish", () => {
    const ms = Date.now() - start;
    logStep(reqId, "completed", `status=${res.statusCode} duration_ms=${ms}`);
  });
  next();
});
app.use((err, _req, res, next) => {
  if (err?.type === "entity.parse.failed") {
    return res.status(400).json({ success: false, error: "Invalid JSON body." });
  }
  return next(err);
});

/** GET /api/health — check Ollama connectivity and list available models */
app.get("/api/health", async (_req, res) => {
  const reqId = _req.reqId ?? "n/a";
  logStep(reqId, "health.start", `ollama_url=${OLLAMA_BASE_URL}`);
  try {
    const response = await fetch(`${OLLAMA_BASE_URL}/api/tags`, { signal: AbortSignal.timeout(5000) });
    logStep(reqId, "health.ollama_response", `status=${response.status}`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    const models = (data.models || []).map((m) => m.name);
    logStep(reqId, "health.ok", `models=${models.length}`);
    res.json({ ok: true, message: `Ollama is running at ${OLLAMA_BASE_URL}`, models });
  } catch (err) {
    logStep(reqId, "health.error", String(err?.message || err));
    res.json({
      ok: false,
      message: `Cannot reach Ollama at ${OLLAMA_BASE_URL}: ${err.message}`,
      models: ["llama3.2-vision", "llava:7b", "llava:13b", "minicpm-v"],
    });
  }
});

/** POST /api/analyze — send base64 image to Ollama, return emotion JSON */
app.post("/api/analyze", async (req, res) => {
  const reqId = req.reqId ?? "n/a";
  logStep(reqId, "analyze.start");
  const { image, model = "llama3.2-vision" } = req.body;
  if (!image) {
    logStep(reqId, "analyze.bad_request", "missing image");
    return res.status(400).json({ success: false, error: "Missing 'image' field." });
  }

  // Strip data-URL prefix if present: "data:image/jpeg;base64,..."
  const imageB64 = image.includes(",") ? image.split(",")[1] : image;
  logStep(reqId, "analyze.payload_ready", `model=${model} image_chars=${imageB64.length}`);

  try {
    logStep(reqId, "analyze.ollama_request.start", `${OLLAMA_BASE_URL}/api/generate`);
    const ollamaRes = await fetch(`${OLLAMA_BASE_URL}/api/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model,
        prompt: EMOTION_PROMPT,
        images: [imageB64],
        stream: false,
        format: "json",
      }),
      signal: AbortSignal.timeout(120_000),
    });
    logStep(reqId, "analyze.ollama_request.done", `status=${ollamaRes.status}`);

    if (!ollamaRes.ok) {
      const text = await ollamaRes.text();
      logStep(reqId, "analyze.ollama_error", `status=${ollamaRes.status} body=${text.slice(0, 300)}`);
      return res.json({ success: false, error: `Ollama error ${ollamaRes.status}: ${text}` });
    }

    const data = await ollamaRes.json();
    const rawText = data.response || "";
    logStep(reqId, "analyze.ollama_json_received", `response_chars=${rawText.length}`);

    let faces = [];

    try {
      const parsed = JSON.parse(rawText);
      const raw = Array.isArray(parsed.faces) ? parsed.faces : [];
      faces = raw.map((f) => {
        const confidence = Number(f.confidence);
        return {
          emotion: String(f.emotion || "neutral").toLowerCase().trim(),
          description: String(f.description || "").trim(),
          confidence: Number.isFinite(confidence) ? Math.max(0, Math.min(100, Math.round(confidence))) : 50,
        };
      });
      logStep(reqId, "analyze.parse_json.ok", `faces=${faces.length}`);
    } catch {
      // model didn't return valid JSON — treat the whole response as a single neutral face
      const firstWord = rawText.trim().split(/\s+/)[0] || "neutral";
      faces = [{ emotion: firstWord.toLowerCase().replace(/[.,!]+$/, ""), description: rawText.trim(), confidence: 50 }];
      logStep(reqId, "analyze.parse_json.fallback", `faces=1 emotion=${faces[0].emotion}`);
    }

    logStep(reqId, "analyze.success");
    res.json({ success: true, faces, raw: rawText });
  } catch (err) {
    logStep(reqId, "analyze.exception", String(err?.message || err));
    const isTimeout = String(err?.message || "").toLowerCase().includes("aborted");
    res.status(isTimeout ? 504 : 500).json({ success: false, error: err.message });
  }
});

app.listen(PORT, () => {
  console.log(`Emotion Recognizer server running at http://localhost:${PORT}`);
  console.log(`Ollama URL: ${OLLAMA_BASE_URL}`);
  runStartupOllamaCheck();
});

app.use((_req, res) => {
  res.status(404).json({ success: false, error: "Not found" });
});

app.use((err, _req, res, _next) => {
  const reqId = _req?.reqId ?? "n/a";
  logStep(reqId, "unhandled_error", String(err?.stack || err));
  res.status(500).json({ success: false, error: "Internal server error" });
});
