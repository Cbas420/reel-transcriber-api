import os
import time
import uuid
import asyncio
import base64
import tempfile
import requests
import cv2
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv
from typing import Dict, Optional, Union, List

load_dotenv()

app = FastAPI(title="Reel Transcriber API", version="2.0.0")

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY", "")
ASSEMBLYAI_KEY = os.getenv("ASSEMBLYAI_KEY", "")

tasks: Dict[str, dict] = {}

# Modelos
class ReelRequest(BaseModel):
    reel_url: HttpUrl

class TaskResponse(BaseModel):
    task_id: str
    status: str

class TranscriptionResult(BaseModel):
    task_id: str
    status: str
    transcription: Optional[str] = None
    frames: Optional[List[str]] = None
    error: Optional[str] = None

class FramesOnlyResponse(BaseModel):
    task_id: str
    status: str
    frames: Optional[List[str]] = None
    error: Optional[str] = None

# Funciones auxiliares
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
    # Descargar video a archivo temporal
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
        video_path = tmp_file.name
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(video_url, headers=headers, stream=True, timeout=60)
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=8192):
            tmp_file.write(chunk)
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("No se pudo abrir el video")
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        if duration <= 0:
            raise Exception("Duración inválida")
        times = []
        if duration <= 1:
            times = [duration * 0.5]
        else:
            step = duration / (num_frames + 1)
            times = [step * i for i in range(1, num_frames + 1)]
        frames_b64 = []
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, frame = cap.read()
            if not ret:
                # intentar por frame index
                frame_idx = int(t * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
            if ret and frame is not None:
                _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                frames_b64.append(base64.b64encode(buffer).decode('utf-8'))
        cap.release()
        return frames_b64
    finally:
        os.unlink(video_path)

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

# Endpoints (iguales que antes)
@app.post("/transcribe", response_model=Union[TaskResponse, TranscriptionResult])
async def start_transcription(
    request: ReelRequest,
    sync: bool = Query(False),
    frames: bool = Query(True)
):
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "processing", "result": None, "frames": None, "error_msg": None}
    asyncio.create_task(process_transcription(task_id, str(request.reel_url), include_frames=frames))
    if sync:
        timeout = 120
        start_time = time.time()
        while time.time() - start_time < timeout:
            task = tasks[task_id]
            if task["status"] == "completed":
                return TranscriptionResult(task_id=task_id, status="completed", transcription=task["result"], frames=task.get("frames"))
            elif task["status"] == "error":
                return TranscriptionResult(task_id=task_id, status="error", error=task["error_msg"])
            await asyncio.sleep(1)
        tasks[task_id]["status"] = "error"
        tasks[task_id]["error_msg"] = "Timeout"
        return TranscriptionResult(task_id=task_id, status="error", error="Timeout")
    else:
        return TaskResponse(task_id=task_id, status="processing")

@app.get("/transcribe/{task_id}", response_model=TranscriptionResult)
async def get_transcription(task_id: str):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task ID no encontrado")
    if task["status"] == "completed":
        return TranscriptionResult(task_id=task_id, status="completed", transcription=task["result"], frames=task.get("frames"))
    elif task["status"] == "error":
        return TranscriptionResult(task_id=task_id, status="error", error=task["error_msg"])
    else:
        return TranscriptionResult(task_id=task_id, status="processing")

@app.post("/frames-only", response_model=FramesOnlyResponse)
async def extract_only_frames(request: ReelRequest):
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "processing", "frames": None, "error_msg": None}
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
        return FramesOnlyResponse(task_id=task_id, status="completed", frames=task.get("frames"))
    elif task["status"] == "error":
        return FramesOnlyResponse(task_id=task_id, status="error", error=task["error_msg"])
    else:
        return FramesOnlyResponse(task_id=task_id, status="processing")

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "rapidapi_configured": bool(RAPIDAPI_KEY),
        "assemblyai_configured": bool(ASSEMBLYAI_KEY)
    }