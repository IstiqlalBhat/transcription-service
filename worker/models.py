import io
import logging
import tempfile
import os

logger = logging.getLogger(__name__)

# Global model references — loaded once on cold start
_whisper_model = None
_whisper_processor = None
_parakeet_model = None
_canary_model = None
_cohere_model = None
_cohere_processor = None


def load_models():
    """Load all 4 ASR models into VRAM. Called once on worker cold start."""
    global _whisper_model, _whisper_processor
    global _parakeet_model, _canary_model
    global _cohere_model, _cohere_processor

    import torch

    try:
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        logger.info("Loading Whisper Large v3...")
        _whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
        _whisper_model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-large-v3"
        ).to("cuda")
        logger.info("Whisper Large v3 loaded.")
    except Exception as e:
        logger.error(f"Failed to load Whisper Large v3: {e}")

    try:
        import nemo.collections.asr as nemo_asr
        logger.info("Loading Parakeet TDT 0.6b v2...")
        _parakeet_model = nemo_asr.models.ASRModel.from_pretrained(
            "nvidia/parakeet-tdt-0.6b-v2"
        )
        _parakeet_model = _parakeet_model.to("cuda")
        logger.info("Parakeet TDT loaded.")
    except Exception as e:
        logger.error(f"Failed to load Parakeet TDT: {e}")

    try:
        from nemo.collections.speechlm2.models import SALM
        logger.info("Loading Canary Qwen 2.5b...")
        _canary_model = SALM.from_pretrained("nvidia/canary-qwen-2.5b")
        logger.info("Canary Qwen loaded.")
    except Exception as e:
        logger.error(f"Failed to load Canary Qwen: {e}")

    try:
        from transformers import AutoProcessor, CohereAsrForConditionalGeneration
        logger.info("Loading Cohere Transcribe...")
        _cohere_processor = AutoProcessor.from_pretrained("CohereLabs/cohere-transcribe-03-2026")
        _cohere_model = CohereAsrForConditionalGeneration.from_pretrained(
            "CohereLabs/cohere-transcribe-03-2026",
            device_map="auto",
        )
        logger.info("Cohere Transcribe loaded.")
    except Exception as e:
        logger.error(f"Failed to load Cohere Transcribe: {e}")

    loaded = sum(1 for m in [_whisper_model, _parakeet_model, _canary_model, _cohere_model] if m is not None)
    logger.info(f"Model loading complete: {loaded}/4 models loaded.")


def _wav_bytes_to_array(wav_bytes: bytes):
    """Convert WAV bytes (16kHz mono) to float32 numpy array."""
    import soundfile as sf
    audio_array, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    return audio_array


def _transcribe_whisper(wav_bytes: bytes) -> str:
    import torch

    audio = _wav_bytes_to_array(wav_bytes)
    inputs = _whisper_processor(
        audio, sampling_rate=16000, return_tensors="pt"
    )
    inputs = inputs.to(device=_whisper_model.device, dtype=_whisper_model.dtype)
    with torch.no_grad():
        predicted_ids = _whisper_model.generate(**inputs)
    return _whisper_processor.batch_decode(
        predicted_ids, skip_special_tokens=True
    )[0].strip()


def _transcribe_parakeet(wav_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        f.flush()
        try:
            results = _parakeet_model.transcribe([f.name])
            return results[0].text.strip() if hasattr(results[0], 'text') else str(results[0]).strip()
        finally:
            os.unlink(f.name)


def _transcribe_canary(wav_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        f.flush()
        try:
            answer_ids = _canary_model.generate(
                prompts=[
                    [{"role": "user", "content": f"Transcribe the following: {_canary_model.audio_locator_tag}", "audio": [f.name]}]
                ],
                max_new_tokens=256,
            )
            return _canary_model.tokenizer.ids_to_text(answer_ids[0].cpu()).strip()
        finally:
            os.unlink(f.name)


def _transcribe_cohere(wav_bytes: bytes) -> str:
    from transformers.audio_utils import load_audio

    audio = _wav_bytes_to_array(wav_bytes)
    inputs = _cohere_processor(
        audio, sampling_rate=16000, return_tensors="pt", language="en"
    )
    inputs = inputs.to(_cohere_model.device, dtype=_cohere_model.dtype)

    outputs = _cohere_model.generate(**inputs, max_new_tokens=256)
    return _cohere_processor.decode(outputs, skip_special_tokens=True).strip()


def transcribe_all(wav_bytes: bytes) -> dict[str, str | None]:
    """Run all 4 models in parallel and return their outputs. If a model fails, its value is None."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    models = {
        "whisper_large_v3": (_transcribe_whisper, _whisper_model),
        "parakeet_tdt": (_transcribe_parakeet, _parakeet_model),
        "canary_qwen": (_transcribe_canary, _canary_model),
        "cohere_transcribe": (_transcribe_cohere, _cohere_model),
    }

    results = {}
    futures = {}

    with ThreadPoolExecutor(max_workers=4) as executor:
        for name, (fn, model_ref) in models.items():
            if model_ref is None:
                logger.warning(f"Model {name} not loaded, skipping")
                results[name] = None
                continue
            futures[executor.submit(fn, wav_bytes)] = name

        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as e:
                logger.error(f"Model {name} failed: {e}")
                results[name] = None

    return results
