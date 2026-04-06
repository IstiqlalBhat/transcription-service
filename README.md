# Ensemble Transcription Service

A centralized transcription API that runs 4 open-source ASR models on RunPod Serverless and uses Claude as an LLM judge to produce the most accurate transcription.

## How It Works

```
Your Apps ──> Gateway (FastAPI) ──> RunPod Worker (4 ASR Models)
                │                         │
                │                         ├── Parakeet TDT (NVIDIA)
                │                         ├── Canary Qwen (NVIDIA)
                │                         ├── Whisper Large v3 (OpenAI)
                │                         └── Cohere Transcribe (Cohere)
                │
                ├── Phase 1: Instant result from Parakeet (~3s)
                │
                └── Phase 2: All 4 models + Claude judge (~30s)
                             Replaces Phase 1 with corrected text
```

**Two-phase transcription:**

1. **Fast** -- Parakeet TDT runs alone and returns a transcription in ~3 seconds
2. **Judged** -- All 4 models run, outputs are sent to Claude Sonnet 4.6 which uses a structured judging methodology (alignment, majority voting, proper noun detection) to produce the most accurate transcription

## Models

| Model | ID | Purpose |
|-------|----|---------|
| NVIDIA Parakeet TDT | `nvidia/parakeet-tdt-0.6b-v2` | Fast path, top-ranked on Open ASR Leaderboard |
| NVIDIA Canary Qwen | `nvidia/canary-qwen-2.5b` | Strong accuracy, handles accented speech |
| OpenAI Whisper Large v3 | `openai/whisper-large-v3` | Widely used baseline, multilingual |
| Cohere Transcribe | `CohereLabs/cohere-transcribe-03-2026` | High-ranking recent model |

## Architecture

### Three Deployments

- **RunPod Serverless Worker** (GPU) -- Docker container with all 4 models, runs on A40 48GB
- **Gateway** (CPU) -- FastAPI app on Render, handles auth, orchestrates worker + Claude judge
- **Frontend** -- Static HTML on Vercel, WebSocket test interface

### API

**REST:**

```bash
curl -X POST https://your-gateway.onrender.com/transcribe \
  -H "X-API-Key: your-key" \
  -F "audio=@recording.wav"
```

**WebSocket:**

```
ws://your-gateway.onrender.com/ws/transcribe?api_key=your-key
```

Send binary audio chunks, receive JSON responses:

```json
{"phase": "fast", "transcription": "Hello world", "confidence": "fast"}
{"phase": "judged", "transcription": "Hello, world.", "confidence": "high", "model_outputs": {...}}
```

**Analytics:**

```bash
curl https://your-gateway.onrender.com/analytics
```

Returns which model the judge selects most often.

## Project Structure

```
transcription-service/
├── gateway/                # FastAPI gateway (CPU)
│   ├── main.py             # REST + WebSocket endpoints
│   ├── auth.py             # API key validation
│   ├── judge.py            # Claude Sonnet 4.6 judge
│   ├── runpod_client.py    # RunPod serverless client
│   ├── analytics.py        # Model performance tracking
│   ├── config.py           # Pydantic settings
│   ├── Dockerfile
│   └── requirements.txt
├── worker/                 # RunPod serverless worker (GPU)
│   ├── handler.py          # RunPod handler (fast/full modes)
│   ├── models.py           # Load & run 4 ASR models
│   ├── audio_utils.py      # ffmpeg audio normalization
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/               # Vercel static frontend
│   └── index.html          # WebSocket test UI
├── tests/
│   ├── test_auth.py
│   ├── test_judge.py
│   ├── test_audio_utils.py
│   ├── test_runpod_client.py
│   ├── test_analytics.py
│   ├── test_websocket.py
│   └── test_gateway.py
└── .github/workflows/      # CI: build Docker images
    ├── build-worker.yml
    └── build-gateway.yml
```

## Setup

### Environment Variables

**Gateway:**

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Claude API key for judge |
| `RUNPOD_API_KEY` | RunPod API key |
| `RUNPOD_ENDPOINT_ID` | RunPod serverless endpoint ID |
| `API_KEYS` | JSON map of app name to API key, e.g. `{"ios_app":"key1"}` |

**Worker (RunPod):**

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | HuggingFace token (required for gated Cohere model) |

### Deploy Worker (RunPod)

1. Create a serverless endpoint on [runpod.io](https://www.runpod.io/console/serverless)
2. Deploy from GitHub: `IstiqlalBhat/transcription-service`
3. Dockerfile path: `worker/Dockerfile`, build context: `worker`
4. GPU: A40 48GB, min workers: 0, max workers: 3
5. Add `HF_TOKEN` environment variable

### Deploy Gateway (Render)

1. Create a web service on [render.com](https://render.com)
2. Connect GitHub repo, root directory: `gateway`, runtime: Docker
3. Add environment variables (see table above)

### Deploy Frontend (Vercel)

```bash
cd frontend
vercel --prod
```

## Local Development

```bash
# Install gateway dependencies
pip install -r gateway/requirements.txt

# Set env vars
export ANTHROPIC_API_KEY="sk-ant-..."
export RUNPOD_API_KEY="rpa_..."
export RUNPOD_ENDPOINT_ID="your-endpoint-id"
export API_KEYS='{"test_app":"test-key-123"}'

# Run gateway
cd transcription-service
PYTHONPATH=. uvicorn gateway.main:app --port 8000

# Run tests
python -m pytest tests/ -v
```

## Performance

| Metric | Value |
|--------|-------|
| Fast transcription (Parakeet) | ~3-4 seconds |
| Full ensemble + judge | ~30-60 seconds |
| Cold start (worker scaling from zero) | ~90 seconds |
| Models loaded | 4/4 on A40 48GB (~16-19GB VRAM) |

## Tech Stack

- **Python 3.11**, FastAPI, PyTorch 2.6, Transformers, NeMo Toolkit
- **Claude Sonnet 4.6** for LLM-as-a-judge (structured methodology: alignment, majority voting, proper noun detection)
- **RunPod Serverless** for GPU inference
- **Render** for gateway hosting
- **Vercel** for frontend hosting
- **GitHub Actions** for Docker image builds
