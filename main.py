import os
import time
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv

# Cargar variables de entorno (opcional)
load_dotenv()

app = FastAPI(
    title="Reel Transcriber API",
    description="Recibe un enlace de Instagram Reel y devuelve la transcripción del audio usando AssemblyAI.",
    version="1.0.0"
)

# Claves API desde variables de entorno
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY", "")
ASSEMBLYAI_KEY = os.getenv("ASSEMBLYAI_KEY", "")

if not RAPIDAPI_KEY or not ASSEMBLYAI_KEY:
    print("⚠️ Advertencia: Faltan claves API. Configura RAPIDAPI_KEY y ASSEMBLYAI_KEY como variables de entorno.")

class ReelRequest(BaseModel):
    reel_url: HttpUrl

class TranscriptionResponse(BaseModel):
    transcription: str
    success: bool = True

# ------------------------------------------------------------
# 1. Obtener URL directa del video desde Instagram (RapidAPI)
# ------------------------------------------------------------
def download_reel_url(reel_url: str, rapid_key: str) -> str:
    """
    Usa la API de RapidAPI 'instagram-reels-downloader-api' para obtener la URL del video.
    """
    if not rapid_key:
        raise HTTPException(status_code=400, detail="Falta la clave de RapidAPI. Configúrala en el servidor.")

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
        raise HTTPException(status_code=502, detail=f"Error al contactar RapidAPI: {str(e)}")

    # Validar la estructura de la respuesta (basada en el código JS original)
    medias = data.get("data", {}).get("medias")
    if not medias or not isinstance(medias, list):
        raise HTTPException(status_code=500, detail="La respuesta de RapidAPI no contiene la lista de medios esperada.")

    video_item = next((item for item in medias if item.get("type") == "video"), None)
    if not video_item or not video_item.get("url"):
        raise HTTPException(status_code=500, detail="No se encontró URL de video en la respuesta de RapidAPI.")

    return video_item["url"]

# ------------------------------------------------------------
# 2. Transcribir audio/video con AssemblyAI
# ------------------------------------------------------------
def transcribe_audio(video_url: str, assembly_key: str) -> str:
    """
    Envía la URL del video a AssemblyAI, espera la transcripción y la devuelve.
    """
    if not assembly_key:
        raise HTTPException(status_code=400, detail="Falta la clave de AssemblyAI. Configúrala en el servidor.")

    # 1. Enviar solicitud de transcripción
    headers = {
        "Authorization": assembly_key,
        "Content-Type": "application/json"
    }
    payload = {
        "audio_url": video_url,
        "language_detection": True,
        "speech_models": [
            "universal-3-pro",
            "universal-2"
        ]
        # Nota: Si quieres usar modelos específicos, añade "speech_models": ["universal-3-pro", "universal-2"]
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
            raise HTTPException(status_code=500, detail="AssemblyAI no devolvió un ID de transcripción.")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Error al enviar a AssemblyAI: {str(e)}")

    # 2. Polling hasta que esté completado (máximo 2 minutos)
    poll_url = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"
    for _ in range(40):  # 40 intentos * 3s = 120 segundos
        time.sleep(3)
        try:
            poll_resp = requests.get(poll_url, headers=headers, timeout=10)
            poll_resp.raise_for_status()
            data = poll_resp.json()
            status = data.get("status")
            if status == "completed":
                return data.get("text", "")
            elif status == "error":
                error_msg = data.get("error", "Error desconocido en AssemblyAI")
                raise HTTPException(status_code=500, detail=f"AssemblyAI error: {error_msg}")
        except requests.exceptions.RequestException:
            continue  # reintentar si falla la conexión

    raise HTTPException(status_code=504, detail="La transcripción tardó demasiado. Inténtalo de nuevo más tarde.")

# ------------------------------------------------------------
# 3. Endpoint principal
# ------------------------------------------------------------
@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_reel(request: ReelRequest):
    """
    Recibe la URL de un reel de Instagram y devuelve la transcripción del audio.
    """
    try:
        # Paso 1: obtener URL directa del video
        video_url = download_reel_url(str(request.reel_url), RAPIDAPI_KEY)

        # Paso 2: transcribir con AssemblyAI
        transcript = transcribe_audio(video_url, ASSEMBLYAI_KEY)

        return TranscriptionResponse(transcription=transcript)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno inesperado: {str(e)}")

# ------------------------------------------------------------
# Endpoint de verificación de salud (opcional)
# ------------------------------------------------------------
@app.get("/health")
async def health_check():
    return {"status": "ok", "rapidapi_configured": bool(RAPIDAPI_KEY), "assemblyai_configured": bool(ASSEMBLYAI_KEY)}