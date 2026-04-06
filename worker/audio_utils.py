import subprocess
import tempfile
import os


def normalize_audio(audio_bytes: bytes) -> bytes:
    """Convert any audio format to 16kHz mono WAV using ffmpeg."""
    if not audio_bytes:
        raise ValueError("Audio data is empty")

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input")
        output_path = os.path.join(tmpdir, "output.wav")

        with open(input_path, "wb") as f:
            f.write(audio_bytes)

        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-ar", "16000",
            "-ac", "1",
            "-sample_fmt", "s16",
            "-f", "wav",
            output_path,
        ]

        result = subprocess.run(
            cmd, capture_output=True, timeout=30,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed: {result.stderr.decode(errors='replace')}"
            )

        with open(output_path, "rb") as f:
            return f.read()
