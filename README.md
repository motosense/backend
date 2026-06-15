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
└── sequential/
    ├── yamnet_sequential.tflite
    └── yamnet_scaler.joblib
```

YAMNet dimuat melalui TensorFlow Hub saat backend pertama kali dijalankan.

## Pengujian

```powershell
uv run pytest
```
