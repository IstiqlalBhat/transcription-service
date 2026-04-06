import asyncio
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
    loop = asyncio.get_event_loop()

    try:
        worker_result = await loop.run_in_executor(None, send_audio_to_worker, audio_bytes, "fast")
    except RuntimeError as e:
        logger.error(f"RunPod worker failed: {e}")
        raise HTTPException(status_code=502, detail=str(e))

    return {
        "transcription": worker_result.get("transcription", ""),
        "confidence": "fast",
        "model_outputs": worker_result.get("model_outputs", {}),
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
            loop = asyncio.get_event_loop()

            # Phase 1: Fast transcription (Parakeet only, ~2-3s)
            try:
                fast_result = await loop.run_in_executor(
                    None, send_audio_to_worker, audio_bytes, "fast"
                )
                await websocket.send_json({
                    "phase": "fast",
                    "transcription": fast_result.get("transcription", ""),
                    "confidence": "fast",
                    "model_outputs": fast_result.get("model_outputs", {}),
                })
            except RuntimeError as e:
                logger.error(f"Fast transcription failed: {e}")
                await websocket.send_json({"error": str(e)})
                continue

            # Phase 2: Full ensemble + judge (background, ~15-30s)
            try:
                full_result = await loop.run_in_executor(
                    None, send_audio_to_worker, audio_bytes, "full"
                )
                model_outputs = full_result.get("model_outputs", {})
                judge_result = await loop.run_in_executor(
                    None, judge_transcriptions, model_outputs
                )

                tracker.record(judge_result.get("primary_model"))

                await websocket.send_json({
                    "phase": "judged",
                    "transcription": judge_result["transcription"],
                    "confidence": judge_result["confidence"],
                    "model_outputs": model_outputs,
                })
            except RuntimeError as e:
                logger.error(f"Full transcription failed: {e}")
                # Fast result already sent — don't error, just log

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {app_name}")
