from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Baad mein apna Netlify URL daal dena
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")


@app.get("/")
def root():
    return {"status": "BolToText API is running ✅"}


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form("whisper-large-v3-turbo"),
    language: str = Form(""),
):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="API key not configured on server.")

    # File size check (25MB limit)
    file_bytes = await file.read()
    if len(file_bytes) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Max 25MB allowed.")

    form_data = {"model": model, "response_format": "verbose_json"}
    if language:
        form_data["language"] = language

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                files={"file": (file.filename, file_bytes, file.content_type or "audio/mpeg")},
                data=form_data,
            )

        if response.status_code != 200:
            error = response.json().get("error", {})
            raise HTTPException(
                status_code=response.status_code,
                detail=error.get("message", "Groq API error")
            )

        return response.json()

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Request timed out. File bahut badi ho sakti hai.")
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Groq se connect nahi ho paya: {str(e)}")
