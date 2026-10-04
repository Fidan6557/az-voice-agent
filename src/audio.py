"""Audio modulu: mikrofondan səs yazır.

Çıxış formatı WAV baytlarıdır. Bu format hər STT xidməti tərəfindən qəbul olunur.
"""
import io
import wave

import sounddevice as sd

import config


def record(seconds: float = config.RECORD_SECONDS) -> bytes:
    """Mikrofondan `seconds` saniyə yaz və WAV baytları qaytar."""
    frames = int(seconds * config.SAMPLE_RATE)

    # sd.rec yazmağa başlayır və dərhal qayıdır (arxa planda yazır)
    audio = sd.rec(
        frames,
        samplerate=config.SAMPLE_RATE,
        channels=config.CHANNELS,
        dtype="int16",
    )
    sd.wait()  # yazılma bitənə qədər gözləyirik

    # Xam rəqəmləri WAV formatına bükürük (yaddaşda, diskə yazmadan)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(config.CHANNELS)
        wav.setsampwidth(2)  # int16 = 2 bayt
        wav.setframerate(config.SAMPLE_RATE)
        wav.writeframes(audio.tobytes())

    return buffer.getvalue()