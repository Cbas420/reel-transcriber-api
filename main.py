import os
import time
import uuid
import asyncio
import subprocess
import tempfile
import base64
import requests
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv
from typing import Dict, Optional, Union, List

load_dotenv()

app = FastAPI(title="Reel Transcriber API", version="2.0.0")

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY", "")
ASSEMBLYAI_KEY = os.getenv("EMBLYAI_KEY", "")  # Corregido: ASSEMBLYAI_KEY

tasks: Dict[str, dict] = {}

# ------------------------------------------------------------
# Modelos
# ------------------------------------------------------------
class ReelRequest(BaseModel):
    reel_url: HttpUrl

class TaskResponse(BaseModel):
    task_id: str
    status: str

class TranscriptionResult(BaseModel):
    task_id: str
    status: str
    transcription: Optional[str] = None
    frames: Optional[List[str]] = None   # Lista de imágenes en base64
    error: Optional[str] = None

class FramesOnlyResponse(BaseModel):
    task_id: str
    status: str
    frames: Optional[List[str]] = None
    error: Optional[str] = None

# ------------------------------------------------------------
# Funciones auxiliares
# ------------------------------------------------------------
def download_reel_url(reel_url: str, rapid_key: str) -> str:
    if not rapid_key:
        raise Exception("Falta la clave de RapidAPI.")
    endpoint = "https://instagram-reels-downloader-api.p.rapidapi.com/download"
    params = {"url": reel_url}
    headers = {
        "x-rapidapi-key": rapid_key,
        "x-rapidapi-host": "instagram-reels-downloader-api.p.rapidapi.com"
    }
    try:
        response = requests.get(endpoint, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error al contactar RapidAPI: {str(e)}")
    medias = data.get("data", {}).get("medias")
    if not medias or not isinstance(medias, list):
        raise Exception("La respuesta de RapidAPI no contiene la lista de medios esperada.")
    video_item = next((item for item in medias if item.get("type") == "video"), None)
    if not video_item or not video_item.get("url"):
        raise Exception("No se encontró URL de video en la respuesta de RapidAPI.")
    return video_item["url"]

def transcribe_audio(video_url: str, assembly_key: str) -> str:
    if not assembly_key:
        raise Exception("Falta la clave de AssemblyAI.")
    headers = {
        "Authorization": assembly_key,
        "Content-Type": "application/json"
    }
    payload = {
        "audio_url": video_url,
        "language_detection": True,
        "speech_models": ["universal-3-pro", "universal-2"]
    }
    try:
        submit_resp = requests.post(
            "https://api.assemblyai.com/v2/transcript",
            json=payload,
            headers=headers,
            timeout=30
        )
        submit_resp.raise_for_status()
        transcript_id = submit_resp.json().get("id")
        if not transcript_id:
            raise Exception("AssemblyAI no devolvió un ID de transcripción.")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error al enviar a AssemblyAI: {str(e)}")
    poll_url = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"
    for _ in range(40):
        time.sleep(3)
        try:
            poll_resp = requests.get(poll_url, headers=headers, timeout=10)
            poll_resp.raise_for_status()
            data = poll_resp.json()
            status = data.get("status")
            if status == "completed":
                return data.get("text", "")
            elif status == "error":
                raise Exception(f"AssemblyAI error: {data.get('error')}")
        except requests.exceptions.RequestException:
            continue
    raise Exception("La transcripción tardó demasiado. Inténtalo de nuevo más tarde.")

def extract_frames(video_url: str, num_frames: int = 5) -> List[str]:
    """
    Extrae frames usando imageio (que incluye FFmpeg embebido).
    """
    try:
        # Leer el video desde la URL
        reader = iio.imiter(video_url, plugin="pyav")
        
        # Obtener la duración (no es directo, así que calculamos número total de frames)
        # Primero, contamos los frames (esto puede ser lento, pero es una vez)
        frames_list = list(reader)
        total_frames = len(frames_list)
        if total_frames == 0:
            raise Exception("No se pudo leer el video")
        
        # Calcular índices de los frames a extraer
        indices = [int(i * total_frames / (num_frames + 1)) for i in range(1, num_frames + 1)]
        
        frames_b64 = []
        for idx in indices:
            frame = frames_list[idx]
            # Convertir frame (numpy array) a imagen PIL y luego a base64
            pil_img = Image.fromarray(frame)
            buffer = BytesIO()
            pil_img.save(buffer, format="JPEG", quality=70)
            img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            frames_b64.append(img_b64)
        
        return frames_b64
    except Exception as e:
        raise Exception(f"Error extrayendo frames con imageio: {str(e)}")

async def process_transcription(task_id: str, reel_url: str, include_frames: bool = True):
    try:
        video_url = download_reel_url(reel_url, RAPIDAPI_KEY)
        transcript = transcribe_audio(video_url, ASSEMBLYAI_KEY)
        frames = None
        if include_frames:
            frames = extract_frames(video_url)
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["result"] = transcript
        tasks[task_id]["frames"] = frames
    except Exception as e:
        tasks[task_id]["status"] = "error"
        tasks[task_id]["error_msg"] = str(e)

# ------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------
@app.post("/transcribe", response_model=Union[TaskResponse, TranscriptionResult])
async def start_transcription(
    request: ReelRequest,
    sync: bool = Query(False, description="Si es True, espera resultado completo"),
    frames: bool = Query(True, description="Si es True, extrae frames del video")
):
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "status": "processing",
        "result": None,
        "frames": None,
        "error_msg": None
    }
    asyncio.create_task(process_transcription(task_id, str(request.reel_url), include_frames=frames))
    
    if sync:
        timeout = 120
        start_time = time.time()
        while time.time() - start_time < timeout:
            task = tasks[task_id]
            if task["status"] == "completed":
                return TranscriptionResult(
                    task_id=task_id,
                    status="completed",
                    transcription=task["result"],
                    frames=task.get("frames")
                )
            elif task["status"] == "error":
                return TranscriptionResult(
                    task_id=task_id,
                    status="error",
                    error=task["error_msg"]
                )
            await asyncio.sleep(1)
        tasks[task_id]["status"] = "error"
        tasks[task_id]["error_msg"] = "Tiempo de espera agotado (120s)"
        return TranscriptionResult(
            task_id=task_id,
            status="error",
            error="Tiempo de espera agotado"
        )
    else:
        return TaskResponse(task_id=task_id, status="processing")

@app.get("/transcribe/{task_id}", response_model=TranscriptionResult)
async def get_transcription(task_id: str):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task ID no encontrado")
    if task["status"] == "completed":
        return TranscriptionResult(
            task_id=task_id,
            status="completed",
            transcription=task["result"],
            frames=task.get("frames")
        )
    elif task["status"] == "error":
        return TranscriptionResult(
            task_id=task_id,
            status="error",
            error=task["error_msg"]
        )
    else:
        return TranscriptionResult(task_id=task_id, status="processing")

@app.post("/frames-only", response_model=FramesOnlyResponse)
async def extract_only_frames(request: ReelRequest):
    """
    Endpoint exclusivo para extraer frames de un reel (sin transcripción).
    No consume AssemblyAI, solo RapidAPI (1 request por reel).
    """
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "status": "processing",
        "frames": None,
        "error_msg": None
    }
    async def _extract():
        try:
            video_url = download_reel_url(str(request.reel_url), RAPIDAPI_KEY)
            frames = extract_frames(video_url)
            tasks[task_id]["status"] = "completed"
            tasks[task_id]["frames"] = frames
        except Exception as e:
            tasks[task_id]["status"] = "error"
            tasks[task_id]["error_msg"] = str(e)
    asyncio.create_task(_extract())
    return FramesOnlyResponse(task_id=task_id, status="processing")

@app.get("/frames-only/{task_id}", response_model=FramesOnlyResponse)
async def get_frames(task_id: str):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task ID no encontrado")
    if task["status"] == "completed":
        return FramesOnlyResponse(
            task_id=task_id,
            status="completed",
            frames=task.get("frames")
        )
    elif task["status"] == "error":
        return FramesOnlyResponse(
            task_id=task_id,
            status="error",
            error=task["error_msg"]
        )
    else:
        return FramesOnlyResponse(task_id=task_id, status="processing")

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "rapidapi_configured": bool(RAPIDAPI_KEY),
        "assemblyai_configured": bool(ASSEMBLYAI_KEY)
    }