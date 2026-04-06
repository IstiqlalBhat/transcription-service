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

FALLBACK_ORDER = ["parakeet_tdt", "canary_qwen", "cohere_transcribe", "whisper_large_v3"]

JUDGE_SYSTEM_PROMPT = """You are an expert transcription judge. You receive multiple transcriptions of the same audio from different ASR models. Your job is to produce the single most accurate transcription.

## Judging Methodology

### Step 1: Alignment
Align the transcriptions word by word. Identify where they agree and where they diverge.

### Step 2: Majority Voting
For each segment where models disagree, use majority voting. If 3 out of 4 models agree on a word or phrase, that's almost certainly correct.

### Step 3: Proper Nouns and Technical Terms
ASR models often fail on proper nouns, brand names, and technical terms (e.g., "OpenAI" transcribed as "open eye" or "opening I"). When one model produces a recognizable proper noun or technical term and others produce phonetically similar but incorrect text, trust the proper noun version. Examples:
- "opening I Whisper" vs "OpenAI Whisper" → "OpenAI Whisper"
- "pie torch" vs "PyTorch" → "PyTorch"
- "hugging face" vs "Hugging Face" → "Hugging Face"

### Step 4: Grammar and Punctuation
- Add proper capitalization and punctuation
- Fix obvious grammatical errors only if the correct form is supported by at least one model's output
- Do NOT add words that no model produced
- Do NOT rephrase or paraphrase — preserve the speaker's exact words

### Step 5: Confidence Assessment
- **high**: All models produce substantially the same output (minor punctuation/capitalization differences only)
- **medium**: Models agree on the core content but differ on 1-2 words or phrases
- **low**: Major disagreements — models produced significantly different text

## Rules
- NEVER hallucinate words that appear in zero model outputs
- NEVER correct grammar beyond what the models themselves suggest
- When in doubt, prefer the output from the model that has the most complete, grammatically coherent sentence
- Proper nouns always win over phonetic approximations"""


def _build_judge_prompt(model_outputs: dict[str, str | None]) -> str:
    lines = []
    for key, display_name in MODEL_DISPLAY_NAMES.items():
        if model_outputs.get(key) is not None:
            lines.append(f"**{display_name}** (`{key}`): \"{model_outputs[key]}\"")

    outputs_block = "\n".join(lines)

    return f"""Here are the transcriptions from {len(lines)} ASR models for the same audio:

{outputs_block}

Analyze these step by step using the methodology, then respond with ONLY this JSON:

```json
{{
  "reasoning": "Brief explanation of your judging decisions (1-2 sentences)",
  "transcription": "The final corrected transcription",
  "confidence": "high|medium|low",
  "primary_model": "model_key of the model whose output was closest to your final answer"
}}
```"""


def judge_transcriptions(model_outputs: dict[str, str | None]) -> dict:
    """Send model outputs to Claude Sonnet for judging."""
    available = {k: v for k, v in model_outputs.items() if v is not None}

    if len(available) == 0:
        return {"transcription": "", "confidence": "low", "primary_model": None}

    if len(available) == 1:
        only_key = next(iter(available.keys()))
        only_text = available[only_key]
        return {"transcription": only_text, "confidence": "low", "primary_model": only_key}

    # If all models agree (exact or near-exact), skip the API call
    texts = list(available.values())
    normalized = [t.strip().lower().rstrip('.!?') for t in texts]
    if len(set(normalized)) == 1:
        best_key = next(iter(available.keys()))
        # Return the version with best punctuation (longest = most punctuation)
        best_key = max(available, key=lambda k: len(available[k]))
        return {
            "transcription": available[best_key],
            "confidence": "high",
            "primary_model": best_key,
        }

    try:
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        prompt = _build_judge_prompt(model_outputs)

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=JUDGE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = response.content[0].text.strip()

        # Extract JSON from response — Claude may include reasoning text before the JSON block
        json_str = None
        if "```json" in raw:
            json_str = raw.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in raw:
            json_str = raw.split("```", 1)[1].split("```", 1)[0].strip()
        elif "{" in raw:
            # Find the JSON object directly
            start = raw.index("{")
            end = raw.rindex("}") + 1
            json_str = raw[start:end]
        else:
            json_str = raw

        result = json.loads(json_str)

        logger.info(f"Judge reasoning: {result.get('reasoning', 'none')}")

        return {
            "transcription": result["transcription"],
            "confidence": result.get("confidence", "medium"),
            "primary_model": result.get("primary_model"),
        }

    except Exception as e:
        logger.error(f"Claude judge failed: {e}")
        for model_key in FALLBACK_ORDER:
            if model_outputs.get(model_key) is not None:
                return {
                    "transcription": model_outputs[model_key],
                    "confidence": "low",
                    "primary_model": model_key,
                }
        return {"transcription": "", "confidence": "low", "primary_model": None}
