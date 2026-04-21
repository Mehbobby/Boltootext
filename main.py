from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx
import os
import tempfile
import hmac
import hashlib
import subprocess
from datetime import datetime
from typing import Optional

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
RAZORPAY_KEY_ID = os.environ.get("RAZORPAY_KEY_ID", "")
RAZORPAY_KEY_SECRET = os.environ.get("RAZORPAY_KEY_SECRET", "")

PLAN_LIMITS = {
    "free": 10,
    "starter": 120,
    "pro": 500,
}

def supabase_headers():
    return {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }

async def get_user_from_token(token: str):
    """Verify JWT token and get user info from Supabase"""
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{SUPABASE_URL}/auth/v1/user",
                headers={
                    "apikey": SUPABASE_SERVICE_KEY,
                    "Authorization": f"Bearer {token}"
                },
                timeout=10
            )
            if r.status_code == 200:
                return r.json()
    except Exception as e:
        print(f"Token verification error: {e}")
    return None

async def get_usage(user_id: str) -> float:
    """Get current month usage for user in minutes"""
    try:
        async with httpx.AsyncClient() as client:
            # Get current month start
            now = datetime.utcnow()
            month_start = f"{now.year}-{now.month:02d}-01T00:00:00"
            
            r = await client.get(
                f"{SUPABASE_URL}/rest/v1/usage",
                params={
                    "user_id": f"eq.{user_id}",
                    "select": "minutes_used,month"
                },
                headers=supabase_headers(),
                timeout=10
            )
            print(f"Get usage response: {r.status_code} - {r.text}")
            
            if r.status_code == 200:
                data = r.json()
                if data and len(data) > 0:
                    return float(data[0].get("minutes_used", 0))
        return 0.0
    except Exception as e:
        print(f"Get usage error: {e}")
        return 0.0

async def add_usage(user_id: str, minutes: float):
    """Add minutes to user usage - create row if not exists"""
    try:
        async with httpx.AsyncClient() as client:
            # First try to get existing row
            r = await client.get(
                f"{SUPABASE_URL}/rest/v1/usage",
                params={"user_id": f"eq.{user_id}", "select": "id,minutes_used"},
                headers=supabase_headers(),
                timeout=10
            )
            print(f"Existing usage check: {r.status_code} - {r.text}")
            
            existing = r.json() if r.status_code == 200 else []
            
            if existing and len(existing) > 0:
                # Update existing row
                current = float(existing[0].get("minutes_used", 0))
                new_total = current + minutes
                row_id = existing[0]["id"]
                
                upd = await client.patch(
                    f"{SUPABASE_URL}/rest/v1/usage",
                    params={"id": f"eq.{row_id}"},
                    json={"minutes_used": new_total},
                    headers=supabase_headers(),
                    timeout=10
                )
                print(f"Usage update: {upd.status_code} - {upd.text}")
                return new_total
            else:
                # Insert new row
                ins = await client.post(
                    f"{SUPABASE_URL}/rest/v1/usage",
                    json={"user_id": user_id, "minutes_used": minutes},
                    headers=supabase_headers(),
                    timeout=10
                )
                print(f"Usage insert: {ins.status_code} - {ins.text}")
                return minutes
    except Exception as e:
        print(f"Add usage error: {e}")
        return 0.0

async def get_subscription(user_id: str) -> str:
    """Get user's current subscription plan"""
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{SUPABASE_URL}/rest/v1/subscriptions",
                params={"user_id": f"eq.{user_id}", "select": "plan,status"},
                headers=supabase_headers(),
                timeout=10
            )
            if r.status_code == 200:
                data = r.json()
                if data and len(data) > 0 and data[0].get("status") == "active":
                    return data[0].get("plan", "free")
        return "free"
    except Exception as e:
        print(f"Get subscription error: {e}")
        return "free"

async def transcribe_with_groq(audio_bytes: bytes, filename: str, language: str = "auto") -> dict:
    """Send audio to Groq Whisper API and return segments + text"""
    
    prompt_map = {
        "hi": "यह हिंदी और हिंगलिश की बातचीत है। शब्दों को देवनागरी में लिखें। अंग्रेजी शब्दों का उच्चारण देवनागरी में करें, अनुवाद न करें।",
        "hinglish": "This is a Hinglish conversation. Transcribe exactly what is spoken.",
        "en": "Transcribe the audio in English."
    }
    
    files = {"file": (filename, audio_bytes, "audio/mpeg")}
    data = {
        "model": "whisper-large-v3-turbo",
        "response_format": "verbose_json",
        "temperature": "0",
    }
    
    if language == "hi":
        data["language"] = "hi"
        data["prompt"] = prompt_map["hi"]
    elif language == "en":
        data["language"] = "en"
        data["prompt"] = prompt_map["en"]
    else:
        data["prompt"] = prompt_map["hinglish"]
    
    async with httpx.AsyncClient() as client:
        r = await client.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            files=files,
            data=data,
            timeout=180
        )
    
    if r.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Groq error: {r.text}")
    
    result = r.json()
    return {
        "text": result.get("text", ""),
        "segments": result.get("segments", []),
        "duration": result.get("duration", 0)
    }

async def convert_text(text: str, segments: list, target: str) -> dict:
    """Convert Hindi transcript to Hinglish or English using Groq LLM"""
    
    if target == "hinglish":
        system = "Convert this Hindi/Devanagari text to Hinglish (Roman script mix of Hindi and English). Keep the meaning same, write Hindi words in Roman script. Return only the converted text, nothing else."
    else:
        system = "Translate this Hindi text to English accurately. Return only the translated text, nothing else."
    
    async with httpx.AsyncClient() as client:
        r = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": text}
                ],
                "max_tokens": 4000,
                "temperature": 0.3
            },
            timeout=60
        )
    
    converted_text = text
    if r.status_code == 200:
        converted_text = r.json()["choices"][0]["message"]["content"].strip()
    
    # Convert segments too if available
    converted_segments = []
    if segments:
        seg_texts = "\n".join([f"[{i}] {s.get('text','')}" for i, s in enumerate(segments)])
        async with httpx.AsyncClient() as client:
            r2 = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [
                        {"role": "system", "content": system + " Each line starts with [number], keep those numbers."},
                        {"role": "user", "content": seg_texts}
                    ],
                    "max_tokens": 4000,
                    "temperature": 0.3
                },
                timeout=60
            )
        
        if r2.status_code == 200:
            converted_lines = r2.json()["choices"][0]["message"]["content"].strip().split("\n")
            for i, seg in enumerate(segments):
                converted_text_seg = seg.get("text", "")
                for line in converted_lines:
                    if line.startswith(f"[{i}]"):
                        converted_text_seg = line[len(f"[{i}]"):].strip()
                        break
                converted_segments.append({
                    "start": seg.get("start", 0),
                    "end": seg.get("end", 0),
                    "text": converted_text_seg
                })
        else:
            converted_segments = segments
    
    return {"text": converted_text, "segments": converted_segments}

@app.get("/")
async def root():
    return {"status": "BolToText API running"}

@app.get("/me")
async def get_me(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Token required")
    
    token = authorization.replace("Bearer ", "")
    user = await get_user_from_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    user_id = user["id"]
    plan = await get_subscription(user_id)
    used = await get_usage(user_id)
    limit = PLAN_LIMITS.get(plan, 10)
    
    return {
        "email": user.get("email", ""),
        "name": user.get("user_metadata", {}).get("full_name", ""),
        "plan": plan,
        "used_minutes": round(used, 2),
        "limit_minutes": limit,
        "remaining_minutes": max(0, round(limit - used, 2))
    }

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = "auto",
    authorization: Optional[str] = Header(None)
):
    # Determine user
    user_id = None
    plan = "guest"
    used = 0.0
    limit = 4.0  # Guest limit in minutes
    
    if authorization and authorization.startswith("Bearer "):
        token = authorization.replace("Bearer ", "")
        user = await get_user_from_token(token)
        if user:
            user_id = user["id"]
            plan = await get_subscription(user_id)
            used = await get_usage(user_id)
            limit = PLAN_LIMITS.get(plan, 10)
    
    # Check limit
    if used >= limit:
        if user_id is None:
            raise HTTPException(status_code=403, detail="GUEST_LIMIT_REACHED")
        else:
            raise HTTPException(status_code=403, detail="LIMIT_REACHED")
    
    # Read file
    audio_bytes = await file.read()
    file_size_mb = len(audio_bytes) / (1024 * 1024)
    
    if file_size_mb > 50:
        raise HTTPException(status_code=400, detail="File too large. Max 50MB.")
    
    # Transcribe
    result = await transcribe_with_groq(audio_bytes, file.filename or "audio.mp3", language)
    
    duration_minutes = result["duration"] / 60.0
    
    # Update usage
    new_used = used
    if user_id:
        new_used = await add_usage(user_id, duration_minutes)
        print(f"Usage updated for {user_id}: +{duration_minutes:.2f} min, total: {new_used:.2f}")
    
    return {
        "text": result["text"],
        "segments": result["segments"],
        "duration_minutes": round(duration_minutes, 2),
        "usage": {
            "used_minutes": round(new_used, 2),
            "limit_minutes": limit,
            "plan": plan
        }
    }

@app.post("/transcribe-url")
async def transcribe_url(
    request: dict,
    authorization: Optional[str] = Header(None)
):
    url = request.get("url", "").strip()
    language = request.get("language", "auto")
    
    if not url:
        raise HTTPException(status_code=400, detail="URL required")
    
    # Determine user
    user_id = None
    plan = "guest"
    used = 0.0
    limit = 4.0
    
    if authorization and authorization.startswith("Bearer "):
        token = authorization.replace("Bearer ", "")
        user = await get_user_from_token(token)
        if user:
            user_id = user["id"]
            plan = await get_subscription(user_id)
            used = await get_usage(user_id)
            limit = PLAN_LIMITS.get(plan, 10)
    
    if used >= limit:
        if user_id is None:
            raise HTTPException(status_code=403, detail="GUEST_LIMIT_REACHED")
        else:
            raise HTTPException(status_code=403, detail="LIMIT_REACHED")
    
    # Download audio using yt-dlp
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "audio.mp3")
        try:
            subprocess.run(
                ["yt-dlp", "-x", "--audio-format", "mp3",
                 "--audio-quality", "5",
                 "-o", output_path, url],
                capture_output=True, text=True, timeout=120, check=True
            )
        except subprocess.CalledProcessError as e:
            raise HTTPException(status_code=400, detail=f"Could not download video: {e.stderr[:200]}")
        except FileNotFoundError:
            raise HTTPException(status_code=500, detail="yt-dlp not installed on server")
        
        if not os.path.exists(output_path):
            # yt-dlp sometimes adds extension
            import glob
            files = glob.glob(os.path.join(tmpdir, "audio.*"))
            if not files:
                raise HTTPException(status_code=400, detail="Could not extract audio from URL")
            output_path = files[0]
        
        with open(output_path, "rb") as f:
            audio_bytes = f.read()
    
    result = await transcribe_with_groq(audio_bytes, "audio.mp3", language)
    duration_minutes = result["duration"] / 60.0
    
    new_used = used
    if user_id:
        new_used = await add_usage(user_id, duration_minutes)
        print(f"URL usage updated for {user_id}: +{duration_minutes:.2f} min, total: {new_used:.2f}")
    
    return {
        "text": result["text"],
        "segments": result["segments"],
        "duration_minutes": round(duration_minutes, 2),
        "usage": {
            "used_minutes": round(new_used, 2),
            "limit_minutes": limit,
            "plan": plan
        }
    }

@app.post("/convert")
async def convert(request: dict):
    text = request.get("text", "")
    segments = request.get("segments", [])
    target = request.get("target", "en")
    
    if not text:
        raise HTTPException(status_code=400, detail="Text required")
    
    result = await convert_text(text, segments, target)
    return result

@app.post("/create-order")
async def create_order(request: dict, authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Token required")
    
    token = authorization.replace("Bearer ", "")
    user = await get_user_from_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    plan = request.get("plan", "starter")
    amount_map = {"starter": 9900, "pro": 29900}  # paise
    amount = amount_map.get(plan, 9900)
    
    async with httpx.AsyncClient() as client:
        r = await client.post(
            "https://api.razorpay.com/v1/orders",
            auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET),
            json={"amount": amount, "currency": "INR", "receipt": f"{user['id'][:8]}_{plan}"},
            timeout=10
        )
    
    if r.status_code != 200:
        raise HTTPException(status_code=500, detail="Could not create order")
    
    return r.json()

@app.post("/verify-payment")
async def verify_payment(request: dict, authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Token required")
    
    token = authorization.replace("Bearer ", "")
    user = await get_user_from_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    order_id = request.get("razorpay_order_id", "")
    payment_id = request.get("razorpay_payment_id", "")
    signature = request.get("razorpay_signature", "")
    plan = request.get("plan", "starter")
    
    expected = hmac.new(
        RAZORPAY_KEY_SECRET.encode(),
        f"{order_id}|{payment_id}".encode(),
        hashlib.sha256
    ).hexdigest()
    
    if not hmac.compare_digest(expected, signature):
        raise HTTPException(status_code=400, detail="Invalid payment signature")
    
    user_id = user["id"]
    async with httpx.AsyncClient() as client:
        # Upsert subscription
        check = await client.get(
            f"{SUPABASE_URL}/rest/v1/subscriptions",
            params={"user_id": f"eq.{user_id}"},
            headers=supabase_headers(),
            timeout=10
        )
        
        sub_data = {"user_id": user_id, "plan": plan, "status": "active", "payment_id": payment_id}
        
        if check.status_code == 200 and check.json():
            await client.patch(
                f"{SUPABASE_URL}/rest/v1/subscriptions",
                params={"user_id": f"eq.{user_id}"},
                json=sub_data,
                headers=supabase_headers(),
                timeout=10
            )
        else:
            await client.post(
                f"{SUPABASE_URL}/rest/v1/subscriptions",
                json=sub_data,
                headers=supabase_headers(),
                timeout=10
            )
    
    return {"success": True, "plan": plan}
