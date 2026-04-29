from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import tempfile
import os
import subprocess
import traceback
from groq import Groq
from fpdf import FPDF

app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

def transcribe_audio(audio_path: str, model: str = "whisper-large-v3"):
    """Transcribe audio using Groq Whisper"""
    with open(audio_path, "rb") as audio_file:
        response = groq_client.audio.transcriptions.create(
            file=audio_file,
            model=model,
            response_format="verbose_json",
            language="hi"
        )
    return response

def convert_to_format(text: str, target_format: str) -> str:
    """Convert text to Hindi/Hinglish/English using Groq LLaMA"""
    if target_format == "hindi":
        prompt = f"Convert the following text to pure Hindi (Devanagari script). Only output the converted text, no explanations:\n\n{text}"
    elif target_format == "hinglish":
        prompt = f"Convert the following text to Hinglish (Hindi words in Roman/English script). Only output the converted text, no explanations:\n\n{text}"
    elif target_format == "english":
        prompt = f"Translate the following Hindi text to English. Only output the translation, no explanations:\n\n{text}"
    else:
        return text

    response = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        max_tokens=4000
    )
    
    return response.choices[0].message.content.strip()

@app.post("/transcribe")
async def transcribe_file(
    file: UploadFile = File(...),
    language: str = Form("hindi"),
    mode: str = Form("fast")
):
    """Transcribe uploaded audio/video file"""
    try:
        print(f"[DEBUG] Starting transcription - Language: {language}, Mode: {mode}")
        print(f"[DEBUG] File: {file.filename}, Content-Type: {file.content_type}")
        
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        print(f"[DEBUG] Saved to: {tmp_path}")

        # Choose model based on mode
        model = "whisper-large-v3" if mode == "accurate" else "whisper-large-v3-turbo"
        print(f"[DEBUG] Using model: {model}")

        # Transcribe
        print(f"[DEBUG] Calling transcribe_audio...")
        result = transcribe_audio(tmp_path, model)
        print(f"[DEBUG] Transcription complete")
        
        # Clean up
        os.unlink(tmp_path)

        # Convert format if needed
        original_text = result.text
        if language != "hindi":
            print(f"[DEBUG] Converting to {language}...")
            converted_text = convert_to_format(original_text, language)
        else:
            converted_text = original_text

        return {
            "text": original_text,
            "converted_text": converted_text,
            "language": language,
            "segments": [{"start": s.start, "end": s.end, "text": s.text} for s in result.segments] if hasattr(result, 'segments') else []
        }

    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"[ERROR] Transcription failed: {str(e)}")
        print(f"[ERROR] Traceback:\n{error_trace}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.post("/transcribe-url")
async def transcribe_url(request: dict):
    """Transcribe video from URL (YouTube, etc)"""
    try:
        url = request.get("url")
        language = request.get("language", "hindi")
        mode = request.get("mode", "fast")

        if not url:
            raise HTTPException(status_code=400, detail="URL required")

        print(f"[DEBUG] Starting URL transcription - URL: {url}")

        # Download audio using yt-dlp
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "audio.mp3")
            
            print(f"[DEBUG] Downloading audio...")
            subprocess.run([
                "yt-dlp",
                "-x",
                "--audio-format", "mp3",
                "--audio-quality", "0",
                "-o", output_path,
                url
            ], check=True, capture_output=True)

            print(f"[DEBUG] Download complete")

            # Choose model
            model = "whisper-large-v3" if mode == "accurate" else "whisper-large-v3-turbo"

            # Transcribe
            print(f"[DEBUG] Transcribing...")
            result = transcribe_audio(output_path, model)
            print(f"[DEBUG] Transcription complete")

        # Convert format
        original_text = result.text
        if language != "hindi":
            print(f"[DEBUG] Converting to {language}...")
            converted_text = convert_to_format(original_text, language)
        else:
            converted_text = original_text

        return {
            "text": original_text,
            "converted_text": converted_text,
            "language": language,
            "segments": [{"start": s.start, "end": s.end, "text": s.text} for s in result.segments] if hasattr(result, 'segments') else []
        }

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Download failed: {e.stderr}")
        raise HTTPException(status_code=400, detail="Failed to download video")
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"[ERROR] URL transcription failed: {str(e)}")
        print(f"[ERROR] Traceback:\n{error_trace}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/export/srt")
async def export_srt(data: dict):
    """Export transcription as SRT subtitle file"""
    try:
        segments = data.get("segments", [])
        
        srt_content = ""
        for i, seg in enumerate(segments, 1):
            start = format_srt_time(seg["start"])
            end = format_srt_time(seg["end"])
            text = seg["text"].strip()
            
            srt_content += f"{i}\n{start} --> {end}\n{text}\n\n"

        return Response(
            content=srt_content,
            media_type="text/plain",
            headers={"Content-Disposition": "attachment; filename=transcription.srt"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def format_srt_time(seconds: float) -> str:
    """Convert seconds to SRT time format (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

@app.post("/export/pdf")
async def export_pdf(data: dict):
    """Export transcription as PDF"""
    try:
        text = data.get("converted_text") or data.get("text", "")
        
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        # Split text into lines
        for line in text.split('\n'):
            pdf.multi_cell(0, 10, line)

        pdf_content = pdf.output(dest='S').encode('latin-1')

        return Response(
            content=pdf_content,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=transcription.pdf"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug")
async def debug_info():
    """Debug endpoint to check configuration"""
    return {
        "status": "ok",
        "groq_key_exists": bool(GROQ_API_KEY),
        "groq_key_length": len(GROQ_API_KEY) if GROQ_API_KEY else 0
    }

@app.get("/")
async def root():
    return {"status": "BolToText API - Simple Version", "version": "2.0"}
