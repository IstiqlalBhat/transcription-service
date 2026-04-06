import base64
import logging

try:
    import runpod
except ImportError:
    runpod = None

from gateway.config import settings

logger = logging.getLogger(__name__)


def send_audio_to_worker(audio_bytes: bytes) -> dict:
    """Send audio to RunPod serverless worker, return model outputs."""
    if runpod is None:
        raise RuntimeError("runpod module not installed")

    runpod.api_key = settings.runpod_api_key
    endpoint = runpod.Endpoint(settings.runpod_endpoint_id)

    audio_b64 = base64.b64encode(audio_bytes).decode()

    try:
        run = endpoint.run_sync(
            request_input={"audio": audio_b64},
            timeout=120,
        )

        if isinstance(run, dict) and "error" in run:
            raise RuntimeError(f"RunPod worker error: {run['error']}")

        return run

    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"RunPod request failed: {e}")
