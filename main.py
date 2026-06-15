import io
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import librosa
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("motosense")

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models" / "sequential"
TFLITE_PATH = MODELS_DIR / "yamnet_sequential.tflite"
SCALER_PATH = MODELS_DIR / "yamnet_scaler.joblib"

SAMPLE_RATE = 16_000
MIN_DURATION_SECONDS = 1
MAX_DURATION_SECONDS = 8
MAX_FILE_SIZE = 25 * 1024 * 1024
ACCEPTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"}
CLASSES = [
    "Clutch-Shoe",
    "Conecting-Rod",
    "Drive-Belt",
    "Piston",
    "Tensioner",
    "Slider",
    "Roller",
    "Face-Drive",
]

yamnet_model = None
interpreter: tf.lite.Interpreter | None = None
input_details = None
output_details = None
scaler = None


class ClassScore(BaseModel):
    label: str
    probability: float


class PredictionResponse(BaseModel):
    filename: str
    predicted_class: str
    confidence: float
    all_scores: list[ClassScore]
    inference_ms: float


def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} tidak ditemukan: {path}")


@asynccontextmanager
async def lifespan(_: FastAPI):
    global yamnet_model, interpreter, input_details, output_details, scaler

    require_file(TFLITE_PATH, "TFLite model")
    require_file(SCALER_PATH, "Scaler")

    logger.info("Memuat YAMNet dari TF-Hub")
    yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

    interpreter = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    scaler = joblib.load(SCALER_PATH)

    output_classes = int(output_details[0]["shape"][-1])
    if output_classes != len(CLASSES):
        raise RuntimeError(
            f"Output model {output_classes}, tetapi label berjumlah {len(CLASSES)}."
        )

    logger.info("Model AI siap dengan %d label", len(CLASSES))
    yield


app = FastAPI(
    title="MotoSense API",
    description="Klasifikasi kerusakan mesin motor dari rekaman audio.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocess_audio(audio_bytes: bytes) -> tuple[np.ndarray, float, int]:
    try:
        waveform, _ = librosa.load(
            io.BytesIO(audio_bytes),
            sr=SAMPLE_RATE,
            mono=True,
            duration=MAX_DURATION_SECONDS,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail="Audio tidak dapat dibaca. Gunakan WAV, MP3, M4A, OGG, atau FLAC.",
        ) from exc

    duration = waveform.size / SAMPLE_RATE
    if duration < MIN_DURATION_SECONDS:
        raise HTTPException(
            status_code=422,
            detail="Audio terlalu pendek. Rekam minimal 1 detik.",
        )

    waveform, _ = librosa.effects.trim(waveform, top_db=30)
    if waveform.size == 0:
        raise HTTPException(status_code=422, detail="Audio tidak berisi suara.")

    waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
    _, embeddings, _ = yamnet_model(waveform.astype(np.float32))
    embedding = np.mean(embeddings.numpy(), axis=0).astype(np.float32)
    frames_processed = int(embeddings.shape[0])
    return embedding, duration, frames_processed


def run_inference(
    embedding: np.ndarray,
) -> tuple[str, float, np.ndarray]:
    scaled = scaler.transform(embedding.reshape(1, -1)).astype(np.float32)
    expected_shape = tuple(int(value) for value in input_details[0]["shape"])
    interpreter.set_tensor(
        input_details[0]["index"],
        scaled.reshape(expected_shape),
    )
    interpreter.invoke()
    scores = interpreter.get_tensor(output_details[0]["index"])[0]
    scores = np.clip(scores.astype(np.float64), 0.0, 1.0)
    best_index = int(np.argmax(scores))
    return CLASSES[best_index], float(scores[best_index]), scores


def ensure_ready() -> None:
    if any(value is None for value in (yamnet_model, interpreter, scaler)):
        raise HTTPException(status_code=503, detail="Model AI belum siap.")


def validate_upload(upload: UploadFile, audio_bytes: bytes) -> None:
    extension = Path(upload.filename or "").suffix.lower()
    if extension and extension not in ACCEPTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Format '{extension}' tidak didukung.",
        )
    if len(audio_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Ukuran audio maksimal 25 MB.")
    if not audio_bytes:
        raise HTTPException(status_code=422, detail="File audio kosong.")


@app.get("/")
async def root():
    ready = all(
        value is not None for value in (yamnet_model, interpreter, scaler)
    )
    return {
        "status": "ready" if ready else "loading",
        "classes": CLASSES,
        "model": "YAMNet + Sequential TFLite",
        "docs": "/docs",
    }


@app.get("/classes")
async def get_classes():
    return {"num_classes": len(CLASSES), "classes": CLASSES}


@app.get("/api/health")
async def health():
    ready = all(
        value is not None for value in (yamnet_model, interpreter, scaler)
    )
    return {
        "status": "ok" if ready else "loading",
        "ready": ready,
        "labels": CLASSES,
        "numLabels": len(CLASSES),
        "engine": "YAMNet + Sequential TFLite",
    }


async def analyze_upload(upload: UploadFile):
    audio_bytes = await upload.read(MAX_FILE_SIZE + 1)
    validate_upload(upload, audio_bytes)
    ensure_ready()

    started = time.perf_counter()
    embedding, duration, frames_processed = preprocess_audio(audio_bytes)
    predicted_class, confidence, scores = run_inference(embedding)
    inference_ms = (time.perf_counter() - started) * 1000
    return (
        predicted_class,
        confidence,
        scores,
        duration,
        frames_processed,
        inference_ms,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    label, confidence, scores, _, _, inference_ms = await analyze_upload(file)
    return PredictionResponse(
        filename=file.filename or "unknown",
        predicted_class=label,
        confidence=round(confidence, 4),
        all_scores=[
            ClassScore(label=name, probability=float(score))
            for name, score in zip(CLASSES, scores, strict=True)
        ],
        inference_ms=round(inference_ms, 1),
    )


@app.post("/api/diagnosis")
async def diagnosis(audio: UploadFile = File(...)):
    (
        label,
        confidence,
        scores,
        duration,
        frames_processed,
        inference_ms,
    ) = await analyze_upload(audio)
    return {
        "label": label,
        "confidence": confidence,
        "scores": [float(score) for score in scores],
        "labels": CLASSES,
        "duration": round(duration, 3),
        "sampleRate": SAMPLE_RATE,
        "framesProcessed": frames_processed,
        "inferenceMs": round(inference_ms, 1),
        "engine": "YAMNet + Sequential TFLite",
        "mode": "ai",
    }
