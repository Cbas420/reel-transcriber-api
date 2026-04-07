import os
import time
import uuid
import asyncio
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv
from typing import Dict, Optional, Union

load_dotenv()

app = FastAPI(title="Reel Transcriber API", version="2.0.0")

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY", "")
ASSEMBLYAI_KEY = os.getenv("ASSEMBLYAI_KEY", "")

tasks: Dict[str, dict] = {}

class ReelRequest(BaseModel):
    reel_url: HttpUrl

class TaskResponse(BaseModel):
    task_id: str
    status: str

class TranscriptionResult(BaseModel):
    task_id: str
    status: str
    transcription: Optional[str] = None
    error: Optional[str] = None

# ------------------------------------------------------------
# Funciones auxiliares (igual que antes, lanzan Exception)
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

async def process_transcription(task_id: str, reel_url: str):
    try:
        video_url = download_reel_url(reel_url, RAPIDAPI_KEY)
        transcript = transcribe_audio(video_url, ASSEMBLYAI_KEY)
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["result"] = transcript
    except Exception as e:
        tasks[task_id]["status"] = "error"
        tasks[task_id]["error_msg"] = str(e)

# ------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------
@app.post("/transcribe", response_model=Union[TaskResponse, TranscriptionResult])
async def start_transcription(request: ReelRequest, sync: bool = False):
    """
    Si sync=False (por defecto): inicia tarea asíncrona y devuelve task_id.
    Si sync=True: espera a que termine y devuelve la transcripción directamente.
    """
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "status": "processing",
        "result": None,
        "error_msg": None
    }
    # Lanzar tarea en segundo plano
    asyncio.create_task(process_transcription(task_id, str(request.reel_url)))
    
    if sync:
        # Esperar hasta que la tarea termine o timeout de 120 segundos
        timeout = 120
        start_time = time.time()
        while time.time() - start_time < timeout:
            task = tasks[task_id]
            if task["status"] == "completed":
                return TranscriptionResult(
                    task_id=task_id,
                    status="completed",
                    transcription=task["result"]
                )
            elif task["status"] == "error":
                return TranscriptionResult(
                    task_id=task_id,
                    status="error",
                    error=task["error_msg"]
                )
            await asyncio.sleep(1)
        # Timeout
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
            transcription=task["result"]
        )
    elif task["status"] == "error":
        return TranscriptionResult(
            task_id=task_id,
            status="error",
            error=task["error_msg"]
        )
    else:
        return TranscriptionResult(task_id=task_id, status="processing")

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "rapidapi_configured": bool(RAPIDAPI_KEY),
        "assemblyai_configured": bool(ASSEMBLYAI_KEY)
    }