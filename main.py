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
SUPPORTED_AUDIO_MIME_TYPES = {
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".mp3": "audio/mpeg",
}
GEMINI_SYSTEM_PROMPT = """
PERAN ANDA: Anda adalah penerjemah ahli untuk ucapan anak tunarungu / speech delay berbahasa Indonesia.

TUGAS UTAMA: 
Ubah transkripsi mentah dari Whisper (yang sering cadel, terpotong, atau tidak jelas) menjadi kalimat bahasa Indonesia sehari-hari yang paling masuk akal.

ATURAN KETAT (WAJIB DIPATUHI):
1. JIKA INPUT SUDAH JELAS: Jika teks input sudah berupa kata/kalimat yang normal, masuk akal, dan jelas (misal: "Halo, siapa?", "Aku mau makan"), JANGAN diubah sama sekali. Keluarkan persis seperti input.
2. JIKA INPUT BERANTAKAN/CADEL: Tebak maksudnya berdasarkan kemiripan bunyi fonetik, konteks anak, dan pola bicara anak tunarungu (misal: "au aan" -> "mau makan", "ucu" -> "minum susu").
3. JANGAN BERBICARA: Ini bukan chatbot percakapan. Jika inputnya "Halo", output HANYA "Halo". JANGAN membalas dengan "Halo juga" atau sapaan balik.
4. BATAS PANJANG: Maksimal 6 kata. Jika lebih, ambil inti subjek dan predikatnya saja.
5. JIKA TIDAK YAKIN SAMA SEKALI: keluarkan teks asli tanpa perubahan.
5. FORMAT FINAL: HANYA keluarkan hasil teks akhir. Dilarang memberikan tanda kutip, pengantar, penjelasan, atau basa-basi.
"""
GEMINI_MODEL_NAME = "gemini-3.1-flash-lite-preview"
LOG_TEXT_PREVIEW_CHARS = 80

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
load_dotenv()

device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"
whisper_model = None
gemini_model = None
whisper_lock = threading.Lock()
gemini_lock = threading.Lock()

gemini_api_keys = []
current_gemini_key_idx = 0


def _preview_text(text: str, max_chars: int = LOG_TEXT_PREVIEW_CHARS) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= max_chars:
        return compact
    return f"{compact[:max_chars]}..."


def _audio_mime_type_for_extension(extension: str) -> str:
    return SUPPORTED_AUDIO_MIME_TYPES.get(extension, "application/octet-stream")

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
    global whisper_model, gemini_model, gemini_api_keys, current_gemini_key_idx

    start_load = time.time()
    logger.info("Server startup initiated")
    logger.info("Hardware detected: %s", device.upper())

    keys_str = os.getenv("GEMINI_API_KEYS", os.getenv("GEMINI_API_KEY", ""))
    gemini_api_keys = [k.strip() for k in keys_str.split(",") if k.strip()]
    
    if not gemini_api_keys:
        raise RuntimeError("GEMINI_API_KEY atau GEMINI_API_KEYS tidak ditemukan. Silakan set di environment.")
    logger.info("Gemini API keys loaded: %d key(s)", len(gemini_api_keys))

    current_gemini_key_idx = 0
    logger.info("Gemini initial API key index=%d", current_gemini_key_idx)

    whisper_model = WhisperModel(MODEL_NAME, device=device, compute_type=compute_type)
    
    genai.configure(api_key=gemini_api_keys[0])
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

        audio_mime_type = _audio_mime_type_for_extension(extension)

        logger.info(
            "request_id=%s audio loaded size_bytes=%d size_kb=%.2f mime_type=%s",
            request_id,
            len(audio_bytes),
            len(audio_bytes) / 1024,
            audio_mime_type,
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

        def run_gemini(raw_text: str, audio_bytes: bytes = None, mime_type: str = "application/octet-stream") -> str:
            global current_gemini_key_idx
            with gemini_lock:
                max_attempts = len(gemini_api_keys)
                attempts = 0
                while attempts < max_attempts:
                    try:
                        genai.configure(api_key=gemini_api_keys[current_gemini_key_idx])
                        logger.info(
                            "request_id=%s using_gemini_key_index=%d",
                            request_id,
                            current_gemini_key_idx,
                        )

                        local_model = genai.GenerativeModel(
                            model_name=GEMINI_MODEL_NAME,
                            system_instruction=GEMINI_SYSTEM_PROMPT,
                        )

                        if audio_bytes:
                            logger.info(
                                "request_id=%s uploading audio to gemini size_bytes=%d mime_type=%s",
                                request_id,
                                len(audio_bytes),
                                mime_type,
                            )
                            audio_file = genai.upload_file(
                                io.BytesIO(audio_bytes),
                                mime_type=mime_type,
                                display_name=f"audio_{request_id}",
                            )
                            logger.info(
                                "request_id=%s audio uploaded to gemini file_uri=%s",
                                request_id,
                                audio_file.uri,
                            )
                            
                            response = local_model.generate_content([
                                "Dengarkan audio ini dan verifikasi transkripsi teks berikut (jika ada). "
                                "Jika audio dan teks cocok, kembalikan teks yang sudah dinormalisasi. "
                                "Jika berbeda, berikan teks yang benar dari audio:\n",
                                audio_file,
                                f"\nTranskripsi dari Whisper: {raw_text}"
                            ])
                        else:
                            response = local_model.generate_content(raw_text)
                        return (response.text or "").strip()
                    except Exception as e:
                        error_msg = str(e).lower()
                        if "429" in error_msg or "quota" in error_msg or "exhausted" in error_msg:
                            logger.warning(
                                "request_id=%s API key %d exhausted/rate-limited. Switching to next key.", 
                                request_id, 
                                current_gemini_key_idx
                            )
                            current_gemini_key_idx = (current_gemini_key_idx + 1) % len(gemini_api_keys)
                            attempts += 1
                            time.sleep(0.5)
                        else:
                            raise e
                raise RuntimeError("Semua API key Gemini kehabisan kuota atau gagal.")

        try:
            logger.info(
                "request_id=%s gemini started model=%s gemini_key_index=%d",
                request_id,
                GEMINI_MODEL_NAME,
                current_gemini_key_idx,
            )
            gemini_started_at = time.time()
            final_text = await asyncio.to_thread(run_gemini, raw_transcript, audio_bytes, audio_mime_type)
            gemini_duration = time.time() - gemini_started_at
        except Exception as gemini_error:
            logger.exception(
                "request_id=%s gemini failed error_type=%s error=%s raw_length=%d gemini_key_index=%d",
                request_id,
                type(gemini_error).__name__,
                str(gemini_error),
                len(raw_transcript),
                current_gemini_key_idx,
            )
            raise HTTPException(status_code=502, detail="Gagal memproses hasil transkripsi dengan Gemini.") from gemini_error

        if not final_text:
            raise HTTPException(status_code=502, detail="Gemini tidak menghasilkan output.")

        logger.info(
            "request_id=%s gemini completed duration=%.2fs final_length=%d final_preview=%s gemini_key_index=%d",
            request_id,
            gemini_duration,
            len(final_text),
            _preview_text(final_text),
            current_gemini_key_idx,
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