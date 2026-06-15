# MotoSense Backend

## Menjalankan API

```powershell
uv sync
uv run uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

API tersedia di `http://127.0.0.1:8000`, dengan dokumentasi interaktif
di `/docs`.

Model classifier dan scaler tersimpan lokal di:

```text
models/
├── yamnet/
│   └── yamnet.tflite
└── sequential/
    ├── yamnet_sequential.tflite
    └── yamnet_scaler.joblib
```

Backend menggunakan LiteRT untuk YAMNet dan classifier, tanpa TensorFlow.
Endpoint diagnosis menerima WAV PCM16. Website mengonversi audio menjadi format
tersebut sebelum upload.

## Pengujian

```powershell
uv run pytest
```
