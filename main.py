import logging
import logging.config
import shutil
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import whisper
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

APP_TITLE = "KawanDengar API Service"
MODEL_NAME = "medium"
SUPPORTED_EXTENSIONS = {".wav", ".m4a", ".mp3", ".mp4"}
INITIAL_PROMPT = (
    "PERAN ANDA: Anda adalah sebuah sistem AI Pemroses Bahasa dan Terapis Wicara yang ahli dalam memahami pola komunikasi anak tunarungu atau anak dengan gangguan artikulasi (speech delay). TUGAS ANDA: Anda akan menerima input teks kasar hasil transkripsi suara anak. Teks tersebut mungkin terdengar seperti gumaman, kehilangan huruf konsonan (cadel), atau suku kata yang terpotong. Tugas Anda adalah menebak dan memperbaiki teks tersebut menjadi kata atau kalimat bahasa Indonesia yang baku namun bernada percakapan sehari-hari. KONTEKS & RUANG LINGKUP: Lingkup pembicaraan adalah percakapan fungsional anak sehari-hari (contoh: meminta makan, menunjuk benda, instruksi dasar, aktivitas harian). Pahami pola artikulasi umum: huruf 'R', 'S', atau konsonan di awal/akhir kata sering hilang (contoh: 'aju iru'' -> baju biru, 'au akan' -> mau makan, 'ucu'' -> minum susu, 'enja' -> meja). ATURAN OUTPUT (SANGAT KETAT): JANGAN memberikan penjelasan, sapaan, atau basa-basi apa pun (Jangan katakan 'Maksud anak tersebut adalah...'). JANGAN menambahkan tanda baca yang berlebihan. KELUARKAN HANYA hasil tebakan akhir yang sudah benar. Output Anda ini akan langsung dikirim ke mesin Text-to-Speech (TTS) untuk dibacakan kepada anak sebagai koreksi suara. CONTOH INPUT DAN OUTPUT: Input: 'au ain oa' Output: mau main bola Input: 'pa ai o-pi' Output: pakai topi"
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = None

def configure_logging() -> None:
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                    "level": "INFO",
                }
            },
            "loggers": {
                "kawandengar.api": {
                    "handlers": ["console"],
                    "level": "INFO",
                    "propagate": False,
                }
            },
        }
    )


configure_logging()
logger = logging.getLogger("kawandengar.api")

@asynccontextmanager
async def lifespan(_: FastAPI):
    global model

    start_load = time.time()
    logger.info("Server startup initiated")
    logger.info("Hardware detected: %s", device.upper())

    model = whisper.load_model(MODEL_NAME, device=device)
    logger.info("Model '%s' loaded in %.2f seconds", MODEL_NAME, time.time() - start_load)
    yield
    logger.info("Server shutdown complete")


app = FastAPI(title=APP_TITLE, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def save_upload_to_tmp(audio_file: UploadFile) -> str:
    suffix = Path(audio_file.filename).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        shutil.copyfileobj(audio_file.file, tmp_file)
        return tmp_file.name

@app.post("/transcribe")
async def transcribe_audio(audio_file: UploadFile = File(...)):
    request_id = uuid.uuid4().hex[:8]

    if not audio_file.filename:
        logger.warning("request_id=%s upload has no filename", request_id)
        raise HTTPException(status_code=400, detail="Nama file tidak valid.")

    extension = Path(audio_file.filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        logger.warning(
            "request_id=%s unsupported format filename=%s extension=%s",
            request_id,
            audio_file.filename,
            extension,
        )
        raise HTTPException(status_code=400, detail="Format tidak didukung. Gunakan .mp3, .wav, .m4a, atau .mp4")

    if model is None:
        logger.error("request_id=%s model is not ready", request_id)
        raise HTTPException(status_code=503, detail="Model belum siap. Coba lagi sebentar.")

    temp_file_path = ""
    try:
        logger.info("request_id=%s transcribe started filename=%s", request_id, audio_file.filename)
        temp_file_path = save_upload_to_tmp(audio_file)

        start_process = time.time()
        result = model.transcribe(
            temp_file_path,
            language="id",
            initial_prompt=INITIAL_PROMPT,
        )

        processing_time = time.time() - start_process
        transcribed_text = result.get("text", "").strip()

        logger.info(
            "request_id=%s transcribe success duration=%.2fs device=%s text_length=%d",
            request_id,
            processing_time,
            device,
            len(transcribed_text),
        )

        return JSONResponse(
            content={
                "status": "success",
                "text": transcribed_text,
                "processing_time_seconds": round(processing_time, 2),
                "device_used": device,
            }
        )

    except HTTPException:
        raise
    except Exception:
        logger.exception("request_id=%s transcribe failed filename=%s", request_id, audio_file.filename)
        raise HTTPException(status_code=500, detail="Gagal memproses audio.")
    finally:
        if temp_file_path and Path(temp_file_path).exists():
            try:
                Path(temp_file_path).unlink()
            except OSError:
                logger.warning(
                    "request_id=%s failed to delete temp file path=%s",
                    request_id,
                    temp_file_path,
                )