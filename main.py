import io
import logging
import time
import wave
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from ai_edge_litert.interpreter import Interpreter
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("motosense")

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
YAMNET_PATH = MODELS_DIR / "yamnet" / "yamnet.tflite"
CLASSIFIER_PATH = MODELS_DIR / "sequential" / "yamnet_sequential.tflite"
SCALER_PATH = MODELS_DIR / "sequential" / "yamnet_scaler.joblib"

SAMPLE_RATE = 16_000
YAMNET_FRAME_SAMPLES = 15_600
MIN_DURATION_SECONDS = 1
MAX_DURATION_SECONDS = 8
MAX_FILE_SIZE = 25 * 1024 * 1024
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

yamnet: Interpreter | None = None
classifier: Interpreter | None = None
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


def load_interpreter(path: Path) -> Interpreter:
    if not path.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {path}")
    runtime = Interpreter(model_path=str(path))
    runtime.allocate_tensors()
    return runtime


@asynccontextmanager
async def lifespan(_: FastAPI):
    global yamnet, classifier, scaler

    yamnet = load_interpreter(YAMNET_PATH)
    classifier = load_interpreter(CLASSIFIER_PATH)
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler tidak ditemukan: {SCALER_PATH}")
    scaler = joblib.load(SCALER_PATH)

    output_classes = int(classifier.get_output_details()[0]["shape"][-1])
    if output_classes != len(CLASSES):
        raise RuntimeError(
            f"Output model {output_classes}, tetapi label berjumlah {len(CLASSES)}."
        )

    logger.info("LiteRT siap dengan %d label", len(CLASSES))
    yield


app = FastAPI(
    title="MotoSense API",
    description="Klasifikasi suara mesin dengan YAMNet dan Sequential LiteRT.",
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


def decode_pcm16_wav(audio_bytes: bytes) -> np.ndarray:
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
            if wav_file.getsampwidth() != 2 or wav_file.getcomptype() != "NONE":
                raise ValueError
            channels = wav_file.getnchannels()
            source_rate = wav_file.getframerate()
            frames = wav_file.readframes(wav_file.getnframes())
    except (EOFError, ValueError, wave.Error) as exc:
        raise HTTPException(
            status_code=422,
            detail="Audio tidak dapat dibaca. Kirim WAV PCM16.",
        ) from exc

    if channels < 1 or source_rate < 1:
        raise HTTPException(status_code=422, detail="Metadata WAV tidak valid.")

    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)

    if source_rate != SAMPLE_RATE and samples.size:
        target_length = round(samples.size * SAMPLE_RATE / source_rate)
        old_positions = np.linspace(0, 1, samples.size, endpoint=False)
        new_positions = np.linspace(0, 1, target_length, endpoint=False)
        samples = np.interp(new_positions, old_positions, samples).astype(np.float32)

    return samples[: SAMPLE_RATE * MAX_DURATION_SECONDS]


def normalize_audio(samples: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak < 1e-6:
        raise HTTPException(status_code=422, detail="Audio tidak berisi suara.")
    return samples / peak


def extract_embedding(samples: np.ndarray) -> tuple[np.ndarray, int]:
    assert yamnet is not None
    input_detail = yamnet.get_input_details()[0]
    embedding_detail = next(
        detail
        for detail in yamnet.get_output_details()
        if list(detail["shape"]) == [1, 1024]
    )
    frame_count = max(1, int(np.ceil(samples.size / SAMPLE_RATE)))
    embeddings = []

    for frame_index in range(frame_count):
        start = frame_index * SAMPLE_RATE
        frame = samples[start : start + YAMNET_FRAME_SAMPLES]
        frame = np.pad(frame, (0, max(0, YAMNET_FRAME_SAMPLES - frame.size)))
        yamnet.set_tensor(
            input_detail["index"],
            frame.reshape(1, -1).astype(np.float32),
        )
        yamnet.invoke()
        embeddings.append(yamnet.get_tensor(embedding_detail["index"])[0])

    return np.mean(embeddings, axis=0).astype(np.float32), frame_count


def run_inference(embedding: np.ndarray) -> tuple[str, float, np.ndarray]:
    assert classifier is not None and scaler is not None
    input_detail = classifier.get_input_details()[0]
    output_detail = classifier.get_output_details()[0]
    scaled = scaler.transform(embedding.reshape(1, -1)).astype(np.float32)
    expected_shape = tuple(int(value) for value in input_detail["shape"])
    classifier.set_tensor(input_detail["index"], scaled.reshape(expected_shape))
    classifier.invoke()
    scores = classifier.get_tensor(output_detail["index"])[0]
    scores = np.clip(scores.astype(np.float64), 0.0, 1.0)
    best_index = int(np.argmax(scores))
    return CLASSES[best_index], float(scores[best_index]), scores


def ready() -> bool:
    return yamnet is not None and classifier is not None and scaler is not None


async def analyze_upload(upload: UploadFile):
    audio_bytes = await upload.read(MAX_FILE_SIZE + 1)
    if len(audio_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Ukuran audio maksimal 25 MB.")
    if not audio_bytes:
        raise HTTPException(status_code=422, detail="File audio kosong.")
    if not ready():
        raise HTTPException(status_code=503, detail="Model AI belum siap.")

    started = time.perf_counter()
    samples = decode_pcm16_wav(audio_bytes)
    duration = samples.size / SAMPLE_RATE
    if duration < MIN_DURATION_SECONDS:
        raise HTTPException(
            status_code=422,
            detail="Audio terlalu pendek. Rekam minimal 1 detik.",
        )

    embedding, frames_processed = extract_embedding(normalize_audio(samples))
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


@app.get("/")
async def root():
    return {
        "status": "ready" if ready() else "loading",
        "classes": CLASSES,
        "model": "YAMNet LiteRT + Sequential LiteRT",
        "docs": "/docs",
    }


@app.get("/classes")
async def get_classes():
    return {"num_classes": len(CLASSES), "classes": CLASSES}


@app.get("/api/health")
async def health():
    is_ready = ready()
    return {
        "status": "ok" if is_ready else "loading",
        "ready": is_ready,
        "labels": CLASSES,
        "numLabels": len(CLASSES),
        "engine": "YAMNet LiteRT + Sequential LiteRT",
    }


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
        "engine": "YAMNet LiteRT + Sequential LiteRT",
        "mode": "ai",
    }
