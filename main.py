import os
import io
import logging
import logging.config
import time
import uuid
import asyncio
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel

APP_TITLE = "API Kawan Dengar"
MODEL_NAME = "large-v3"
SUPPORTED_EXTENSIONS = {".wav", ".m4a", ".mp3"}
GEMINI_SYSTEM_PROMPT = """
PERAN ANDA: Anda adalah penerjemah ahli untuk ucapan anak tunarungu/speech delay ke bahasa Indonesia sehari-hari.

TUGAS UTAMA: 
Anda akan menerima teks transkripsi dari suara anak. Tugas Anda adalah menerjemahkan kata yang cadel, terpotong, atau tidak jelas menjadi frasa yang logis.

ATURAN KETAT (WAJIB DIPATUHI):
1. JIKA INPUT SUDAH JELAS: Jika teks input sudah berupa kata/kalimat yang normal, masuk akal, dan jelas (misal: "Halo, siapa?", "Aku mau makan"), JANGAN diubah sama sekali. Keluarkan persis seperti input.
2. JIKA INPUT BERANTAKAN/CADEL: Tebak maksudnya berdasarkan kemiripan bunyi fonetik (misal: "au aan" -> "mau makan", "ucu" -> "minum susu").
3. JANGAN BERBICARA: Ini bukan chatbot percakapan. Jika inputnya "Halo", output HANYA "Halo". JANGAN membalas dengan "Halo juga" atau sapaan balik.
4. BATAS PANJANG: Maksimal 5 kata. Jika lebih, ambil inti subjek dan predikatnya saja.
5. FORMAT FINAL: HANYA keluarkan hasil teks akhir. Dilarang memberikan tanda kutip, pengantar, penjelasan, atau basa-basi.
"""
GEMINI_MODEL_NAME = "gemini-2.5-flash-lite"
LOG_TEXT_PREVIEW_CHARS = 80

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
load_dotenv()

device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"
whisper_model = None
gemini_model = None
whisper_lock = threading.Lock()
gemini_lock = threading.Lock()


def _preview_text(text: str, max_chars: int = LOG_TEXT_PREVIEW_CHARS) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= max_chars:
        return compact
    return f"{compact[:max_chars]}..."

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
    global whisper_model, gemini_model

    start_load = time.time()
    logger.info("Server startup initiated")
    logger.info("Hardware detected: %s", device.upper())

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY tidak ditemukan. Silakan set di environment.")
    logger.info("Gemini API key detected: %s", "yes")

    whisper_model = WhisperModel(MODEL_NAME, device=device, compute_type=compute_type)
    genai.configure(api_key=gemini_api_key)
    gemini_model = genai.GenerativeModel(
        model_name=GEMINI_MODEL_NAME,
        system_instruction=GEMINI_SYSTEM_PROMPT,
    )

    logger.info(
        "Models loaded in %.2f seconds whisper=%s gemini=%s device=%s compute_type=%s",
        time.time() - start_load,
        MODEL_NAME,
        GEMINI_MODEL_NAME,
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
        raise HTTPException(status_code=400, detail="Format tidak didukung. Gunakan .wav, .m4a, atau .mp3")

    if whisper_model is None or gemini_model is None:
        logger.error("request_id=%s model is not ready", request_id)
        raise HTTPException(status_code=503, detail="Model belum siap. Coba lagi sebentar.")

    try:
        request_started_at = time.time()
        logger.info(
            "request_id=%s request accepted filename=%s content_type=%s",
            request_id,
            audio_file.filename,
            audio_file.content_type,
        )

        audio_bytes = await audio_file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="File audio kosong.")

        logger.info(
            "request_id=%s audio loaded size_bytes=%d size_kb=%.2f",
            request_id,
            len(audio_bytes),
            len(audio_bytes) / 1024,
        )

        audio_stream = io.BytesIO(audio_bytes)

        start_process = time.time()

        def run_whisper(audio_buffer: io.BytesIO) -> str:
            with whisper_lock:
                segments_gen, _info_result = whisper_model.transcribe(
                    audio_buffer,
                    language="id",
                    beam_size=7,
                    temperature=0.0,
                    patience=1.5,
                    condition_on_previous_text=False,

                    no_speech_threshold=0.35,
                    compression_ratio_threshold=2.0,
                )

            return " ".join(segment.text for segment in segments_gen).strip()

        logger.info("request_id=%s whisper started model=%s", request_id, MODEL_NAME)
        whisper_started_at = time.time()
        raw_transcript = await asyncio.to_thread(run_whisper, audio_stream)
        whisper_duration = time.time() - whisper_started_at

        if not raw_transcript:
            raise HTTPException(status_code=422, detail="Transkripsi kosong. Coba rekam ulang audio.")

        logger.info(
            "request_id=%s whisper completed duration=%.2fs raw_length=%d raw_preview=%s",
            request_id,
            whisper_duration,
            len(raw_transcript),
            _preview_text(raw_transcript),
        )

        def run_gemini(raw_text: str) -> str:
            with gemini_lock:
                response = gemini_model.generate_content(raw_text)
            return (response.text or "").strip()

        try:
            logger.info("request_id=%s gemini started model=%s", request_id, GEMINI_MODEL_NAME)
            gemini_started_at = time.time()
            final_text = await asyncio.to_thread(run_gemini, raw_transcript)
            gemini_duration = time.time() - gemini_started_at
        except Exception as gemini_error:
            logger.exception(
                "request_id=%s gemini failed error_type=%s error=%s raw_length=%d",
                request_id,
                type(gemini_error).__name__,
                str(gemini_error),
                len(raw_transcript),
            )
            raise HTTPException(status_code=502, detail="Gagal memproses hasil transkripsi dengan Gemini.") from gemini_error

        if not final_text:
            raise HTTPException(status_code=502, detail="Gemini tidak menghasilkan output.")

        logger.info(
            "request_id=%s gemini completed duration=%.2fs final_length=%d final_preview=%s",
            request_id,
            gemini_duration,
            len(final_text),
            _preview_text(final_text),
        )

        processing_time = time.time() - start_process
        request_total_time = time.time() - request_started_at

        logger.info(
            "request_id=%s transcribe success pipeline_duration=%.2fs total_duration=%.2fs device=%s raw_length=%d final_length=%d",
            request_id,
            processing_time,
            request_total_time,
            device,
            len(raw_transcript),
            len(final_text),
        )

        return JSONResponse(
            content={
                "status": "success",
                "raw_text": raw_transcript,
                "text": final_text,
                "processing_time_seconds": round(processing_time, 2),
                "device_used": device,
            }
        )

    except HTTPException:
        raise
    except Exception:
        logger.exception("request_id=%s transcribe failed filename=%s", request_id, audio_file.filename)
        raise HTTPException(status_code=500, detail="Gagal memproses audio.")