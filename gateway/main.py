import logging

from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.responses import JSONResponse

from gateway.auth import validate_api_key
from gateway.judge import judge_transcriptions
from gateway.runpod_client import send_audio_to_worker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Ensemble Transcription Service")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile | None = File(None),
    x_api_key: str | None = Header(None),
):
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")

    app_name = validate_api_key(x_api_key)
    if not app_name:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if audio is None:
        raise HTTPException(status_code=422, detail="audio field is required")

    audio_bytes = await audio.read()

    try:
        worker_result = send_audio_to_worker(audio_bytes)
    except RuntimeError as e:
        logger.error(f"RunPod worker failed: {e}")
        raise HTTPException(status_code=502, detail=str(e))

    model_outputs = worker_result["model_outputs"]

    judge_result = judge_transcriptions(model_outputs)

    return {
        "transcription": judge_result["transcription"],
        "confidence": judge_result["confidence"],
        "model_outputs": model_outputs,
    }
