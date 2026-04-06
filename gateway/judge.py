import json
import logging

import anthropic

from gateway.config import settings

logger = logging.getLogger(__name__)

MODEL_DISPLAY_NAMES = {
    "whisper_large_v3": "Whisper Large v3",
    "parakeet_tdt": "Parakeet TDT",
    "canary_qwen": "Canary Qwen",
    "cohere_transcribe": "Cohere Transcribe",
}

# Fallback priority — best model first
FALLBACK_ORDER = ["parakeet_tdt", "canary_qwen", "cohere_transcribe", "whisper_large_v3"]


def _build_judge_prompt(model_outputs: dict[str, str | None]) -> str:
    lines = []
    for key, display_name in MODEL_DISPLAY_NAMES.items():
        if model_outputs.get(key) is not None:
            lines.append(f"- {display_name} ({key}): \"{model_outputs[key]}\"")

    outputs_block = "\n".join(lines)

    return f"""You are a transcription judge. You received transcriptions of the same audio from different ASR models. Your job:

1. Compare all outputs
2. Identify the most accurate transcription, or merge the best parts from each
3. Fix obvious errors (grammar, missing words) that you can infer from cross-referencing the outputs
4. Identify which model was most accurate (primary_model) using the model key in parentheses
5. Rate confidence: "high" if models mostly agree, "medium" if some diverge, "low" if major disagreement

Respond with ONLY valid JSON: {{"transcription": "final text", "confidence": "high|medium|low", "primary_model": "model_key"}}

Model outputs:
{outputs_block}"""


def judge_transcriptions(model_outputs: dict[str, str | None]) -> dict:
    """Send model outputs to Claude Haiku for judging. Returns dict with transcription, confidence, and primary_model."""
    available = {k: v for k, v in model_outputs.items() if v is not None}

    if len(available) == 0:
        return {"transcription": "", "confidence": "low", "primary_model": None}

    if len(available) == 1:
        only_key = next(iter(available.keys()))
        only_text = available[only_key]
        return {"transcription": only_text, "confidence": "low", "primary_model": only_key}

    try:
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        prompt = _build_judge_prompt(model_outputs)

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = response.content[0].text.strip()
        result = json.loads(raw)
        return {
            "transcription": result["transcription"],
            "confidence": result.get("confidence", "medium"),
            "primary_model": result.get("primary_model"),
        }

    except Exception as e:
        logger.error(f"Claude judge failed: {e}")
        # Fallback to best available model
        for model_key in FALLBACK_ORDER:
            if model_outputs.get(model_key) is not None:
                return {
                    "transcription": model_outputs[model_key],
                    "confidence": "low",
                    "primary_model": model_key,
                }
        return {"transcription": "", "confidence": "low", "primary_model": None}
