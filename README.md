# Mood Mirror AI

Interactive AI demo app for live webcam analysis.

It combines:
- Emotion analysis (via local/remote Ollama vision model)
- Face detection (MediaPipe)
- Hand gesture detection (MediaPipe)
- Object detection (TensorFlow COCO-SSD)
- Live metrics (FPS + analysis latency)

## Project Structure

```text
emotion recognition/
├── app/      # React + Vite frontend
└── server/   # Node.js + Express backend (Ollama proxy)
```

## Prerequisites

- Node.js 18+ (recommended 20+)
- npm
- Ollama running and reachable from `server/.env`
- A pulled vision model (example: `llama3.2-vision`, `llava:7b`, `qwen2.5vl`)

## 1) Configure Backend

Edit:

`server/.env`

Example:

```env
OLLAMA_BASE_URL=http://192.168.100.128:11434
PORT=5005
```

## 2) Install Dependencies

```bash
# backend
cd server
npm install

# frontend
cd ../app
npm install
```

## 3) Run the App

Use two terminals:

```bash
# Terminal A
cd server
npm run dev

# Terminal B
cd app
npm run dev
```

Open:

`http://localhost:5173`

## Core Features

- Capture & Analyze (single or multi-person face analysis)
- Confidence bar + per-person confidence
- Gesture recognition (thumbs up/down, victory, etc.)
- Object recognition with uncertainty handling
- Recent capture history
- Performance modes:
  - Balanced
  - Fast
  - Object Demo

## Tips for Stable Demos

- Prefer `llava:7b` or `llama3.2-vision` for responsiveness
- Use **Fast** mode for smoother live interaction
- Keep good front lighting
- Turn off Auto when testing model stability

## Troubleshooting

- **No emotion result**
  - Check backend terminal logs for `/api/analyze` timeouts or model loading errors
  - Switch to a lighter model
- **Face/object boxes off**
  - Ensure webcam permission is allowed in browser
  - Refresh after dependency installs
- **Ollama reachable but slow**
  - Model may still be loading (`llm server loading model`)
  - Wait and retry, or switch model

## Scripts

### server
- `npm run dev` — run with watch mode
- `npm start` — run normally

### app
- `npm run dev` — start Vite dev server
- `npm run build` — production build
- `npm run preview` — preview production build
