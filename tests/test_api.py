import io
import math
import struct
import wave

from fastapi.testclient import TestClient

from main import MAX_FILE_SIZE, app

EXPECTED_LABELS = [
    "Clutch-Shoe",
    "Conecting-Rod",
    "Drive-Belt",
    "Piston",
    "Tensioner",
    "Slider",
    "Roller",
    "Face-Drive",
]


def make_wav(duration: float, sample_rate: int = 16_000) -> bytes:
    frame_count = int(duration * sample_rate)
    pcm = bytearray()

    for index in range(frame_count):
        time = index / sample_rate
        sample = 0.3 * math.sin(2 * math.pi * 220 * time)
        sample += 0.15 * math.sin(2 * math.pi * 440 * time)
        pcm.extend(struct.pack("<h", int(sample * 32_767)))

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return buffer.getvalue()


def test_health_reports_ready_models():
    with TestClient(app) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "ready": True,
        "labels": EXPECTED_LABELS,
        "numLabels": 8,
        "engine": "YAMNet LiteRT + Sequential LiteRT",
    }


def test_diagnosis_returns_normalized_scores():
    with TestClient(app) as client:
        response = client.post(
            "/api/diagnosis",
            files={"audio": ("engine.wav", make_wav(6), "audio/wav")},
        )

    assert response.status_code == 200
    result = response.json()
    assert result["label"] in result["labels"]
    assert result["labels"] == EXPECTED_LABELS
    assert result["sampleRate"] == 16_000
    assert result["framesProcessed"] > 0
    assert result["engine"] == "YAMNet LiteRT + Sequential LiteRT"
    assert result["mode"] == "ai"
    assert result["confidence"] == max(result["scores"])


def test_invalid_audio_is_rejected():
    with TestClient(app) as client:
        response = client.post(
            "/api/diagnosis",
            files={"audio": ("broken.wav", b"not-a-wave", "audio/wav")},
        )

    assert response.status_code == 422
    assert "WAV PCM16" in response.json()["detail"]


def test_short_audio_is_rejected():
    with TestClient(app) as client:
        response = client.post(
            "/api/diagnosis",
            files={"audio": ("short.wav", make_wav(0.5), "audio/wav")},
        )

    assert response.status_code == 422
    assert "terlalu pendek" in response.json()["detail"]


def test_oversized_audio_is_rejected():
    with TestClient(app) as client:
        response = client.post(
            "/api/diagnosis",
            files={
                "audio": (
                    "large.wav",
                    b"0" * (MAX_FILE_SIZE + 1),
                    "audio/wav",
                )
            },
        )

    assert response.status_code == 413
