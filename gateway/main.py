import logging

from fastapi import FastAPI, UploadFile, File, Header, HTTPException, WebSocket, WebSocketDisconnect, Query

from gateway.auth import validate_api_key
from gateway.judge import judge_transcriptions
from gateway.runpod_client import send_audio_to_worker
from gateway.analytics import tracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Ensemble Transcription Service")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/analytics")
def analytics():
    return tracker.get_stats()


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

    tracker.record(judge_result.get("primary_model"))

    return {
        "transcription": judge_result["transcription"],
        "confidence": judge_result["confidence"],
        "model_outputs": model_outputs,
    }


@app.websocket("/ws/transcribe")
async def websocket_transcribe(
    websocket: WebSocket,
    api_key: str = Query(default=None),
):
    if not api_key:
        await websocket.close(code=4001, reason="Missing api_key query parameter")
        return

    app_name = validate_api_key(api_key)
    if not app_name:
        await websocket.close(code=4001, reason="Invalid API key")
        return

    await websocket.accept()
    logger.info(f"WebSocket connected: {app_name}")

    try:
        while True:
            audio_bytes = await websocket.receive_bytes()

            try:
                worker_result = send_audio_to_worker(audio_bytes)
                model_outputs = worker_result["model_outputs"]
                judge_result = judge_transcriptions(model_outputs)

                tracker.record(judge_result.get("primary_model"))

                await websocket.send_json({
                    "transcription": judge_result["transcription"],
                    "confidence": judge_result["confidence"],
                    "model_outputs": model_outputs,
                })

            except RuntimeError as e:
                logger.error(f"Worker failed during WebSocket: {e}")
                await websocket.send_json({"error": str(e)})

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {app_name}")
