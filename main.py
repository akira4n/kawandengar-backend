import os
import io
import logging
import logging.config
import time
import uuid
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel

APP_TITLE = "API Kawan Dengar"
MODEL_NAME = "medium"
SUPPORTED_EXTENSIONS = {".wav", ".m4a", ".mp3", ".mp4"}
INITIAL_PROMPT = (
    "PERAN ANDA: Anda adalah sebuah sistem AI Pemroses Bahasa dan Terapis Wicara yang ahli dalam memahami pola komunikasi anak tunarungu atau anak dengan gangguan artikulasi (speech delay). TUGAS ANDA: Anda akan menerima input teks kasar hasil transkripsi suara anak. Teks tersebut mungkin terdengar seperti gumaman, kehilangan huruf konsonan (cadel), atau suku kata yang terpotong. Tugas Anda adalah menebak dan memperbaiki teks tersebut menjadi kata atau kalimat bahasa Indonesia yang baku namun bernada percakapan sehari-hari. KONTEKS & RUANG LINGKUP: Lingkup pembicaraan adalah percakapan fungsional anak sehari-hari (contoh: meminta makan, menunjuk benda, instruksi dasar, aktivitas harian). Pahami pola artikulasi umum: huruf 'R', 'S', atau konsonan di awal/akhir kata sering hilang (contoh: 'aju iru'' -> baju biru, 'au akan' -> mau makan, 'ucu'' -> minum susu, 'enja' -> meja). ATURAN OUTPUT (SANGAT KETAT): JANGAN memberikan penjelasan, sapaan, atau basa-basi apa pun (Jangan katakan 'Maksud anak tersebut adalah...'). JANGAN menambahkan tanda baca yang berlebihan. KELUARKAN HANYA hasil tebakan akhir yang sudah benar. Output Anda ini akan langsung dikirim ke mesin Text-to-Speech (TTS) untuk dibacakan kepada anak sebagai koreksi suara. CONTOH INPUT DAN OUTPUT: Input: 'au ain oa' Output: mau main bola Input: 'pa ai o-pi' Output: pakai topi"
)

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
load_dotenv()

device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"
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

    model = WhisperModel(MODEL_NAME, device=device, compute_type=compute_type)
    logger.info(
        "Model '%s' loaded in %.2f seconds device=%s compute_type=%s",
        MODEL_NAME,
        time.time() - start_load,
        device,
        compute_type,
    )
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

    try:
        logger.info("request_id=%s transcribe started filename=%s", request_id, audio_file.filename)

        audio_bytes = await audio_file.read()
        audio_stream = io.BytesIO(audio_bytes)

        start_process = time.time()
        
        def run_model():
            segments_gen, info_result = model.transcribe(
                audio_stream, 
                language="id",
                initial_prompt=INITIAL_PROMPT,
                temperature=0.0,
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
                compression_ratio_threshold=2.4,
            )
            
            teks_gabungan = " ".join(segment.text for segment in segments_gen).strip()
            return teks_gabungan, info_result
        
        transcribed_text, _info = await asyncio.to_thread(run_model)

        processing_time = time.time() - start_process

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