from fastapi import FastAPI, File, UploadFile, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
import os
import hmac
import hashlib
import json
from typing import Optional

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
RAZORPAY_KEY_ID = os.environ.get("RAZORPAY_KEY_ID", "")
RAZORPAY_KEY_SECRET = os.environ.get("RAZORPAY_KEY_SECRET", "")

PLAN_LIMITS = {
    "free": 10 * 60,       # 10 min in seconds
    "starter": 120 * 60,   # 120 min
    "pro": 500 * 60        # 500 min
}

async def get_user_from_token(authorization: str) -> Optional[dict]:
    """Verify token with Supabase and return user info"""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    
    token = authorization.replace("Bearer ", "").strip()
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Use Supabase's /auth/v1/user endpoint to verify token
            resp = await client.get(
                f"{SUPABASE_URL}/auth/v1/user",
                headers={
                    "Authorization": f"Bearer {token}",
                    "apikey": SUPABASE_SERVICE_KEY,
                }
            )
            if resp.status_code == 200:
                return resp.json()
            else:
                print(f"Token verify failed: {resp.status_code} {resp.text}")
                return None
    except Exception as e:
        print(f"Token verify error: {e}")
        return None

async def get_user_usage(user_id: str) -> dict:
    """Get user's current usage and plan from Supabase"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Get subscription
            sub_resp = await client.get(
                f"{SUPABASE_URL}/rest/v1/subscriptions?user_id=eq.{user_id}&select=*",
                headers={
                    "apikey": SUPABASE_SERVICE_KEY,
                    "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                }
            )
            plan = "free"
            if sub_resp.status_code == 200:
                subs = sub_resp.json()
                if subs:
                    plan = subs[0].get("plan", "free")

            # Get usage this month
            from datetime import datetime
            month_start = datetime.now().strftime("%Y-%m-01")
            usage_resp = await client.get(
                f"{SUPABASE_URL}/rest/v1/usage?user_id=eq.{user_id}&created_at=gte.{month_start}&select=duration_seconds",
                headers={
                    "apikey": SUPABASE_SERVICE_KEY,
                    "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                }
            )
            used_seconds = 0
            if usage_resp.status_code == 200:
                records = usage_resp.json()
                used_seconds = sum(r.get("duration_seconds", 0) for r in records)

            return {
                "plan": plan,
                "used_seconds": used_seconds,
                "limit_seconds": PLAN_LIMITS.get(plan, PLAN_LIMITS["free"])
            }
    except Exception as e:
        print(f"Usage fetch error: {e}")
        return {"plan": "free", "used_seconds": 0, "limit_seconds": PLAN_LIMITS["free"]}

async def record_usage(user_id: str, duration_seconds: int):
    """Record transcription usage"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(
                f"{SUPABASE_URL}/rest/v1/usage",
                headers={
                    "apikey": SUPABASE_SERVICE_KEY,
                    "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                    "Content-Type": "application/json",
                },
                json={"user_id": user_id, "duration_seconds": duration_seconds}
            )
    except Exception as e:
        print(f"Record usage error: {e}")

async def do_transcription(file_bytes: bytes, filename: str) -> dict:
    """Call Groq Whisper API"""
    async with httpx.AsyncClient(timeout=180.0) as client:
        files = {"file": (filename, file_bytes, "audio/mpeg")}
        data = {
            "model": "whisper-large-v3-turbo",
            "response_format": "verbose_json",
            "temperature": "0",
        }
        resp = await client.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            files=files,
            data=data,
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Groq error: {resp.text}")
        return resp.json()

@app.get("/me")
async def get_me(authorization: str = Header(None)):
    user = await get_user_from_token(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    usage = await get_user_usage(user["id"])
    return {
        "id": user["id"],
        "email": user.get("email", ""),
        **usage
    }

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), authorization: str = Header(None)):
    user = await get_user_from_token(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    usage = await get_user_usage(user["id"])
    remaining = usage["limit_seconds"] - usage["used_seconds"]
    
    if remaining <= 0:
        raise HTTPException(status_code=403, detail="limit_exceeded")
    
    file_bytes = await file.read()
    result = await do_transcription(file_bytes, file.filename)
    
    duration = int(result.get("duration", 60))
    await record_usage(user["id"], duration)
    
    return {
        "text": result.get("text", ""),
        "segments": result.get("segments", []),
        "duration": duration,
        "usage": {
            "used_seconds": usage["used_seconds"] + duration,
            "limit_seconds": usage["limit_seconds"],
            "plan": usage["plan"]
        }
    }

@app.post("/transcribe-guest")
async def transcribe_guest(file: UploadFile = File(...)):
    file_bytes = await file.read()
    result = await do_transcription(file_bytes, file.filename)
    return {
        "text": result.get("text", ""),
        "segments": result.get("segments", []),
        "duration": int(result.get("duration", 60))
    }

@app.post("/convert")
async def convert_text(request: dict):
    text = request.get("text", "")
    target = request.get("target", "hinglish")
    
    if target == "hindi":
        return {"converted": text}
    
    if target == "hinglish":
        prompt = f"""Convert this Hindi text to Hinglish (Hindi words written in Roman/English script, keeping English words as-is). 
Write EXACTLY as spoken, phonetically in Roman script. Do not translate meaning, just transliterate.
Example: "यह बहुत अच्छा है" → "Yeh bahut accha hai"
Text: {text}
Output only the Hinglish text, nothing else."""
    else:  # english
        prompt = f"""Translate this Hindi/Hinglish text to English naturally.
Text: {text}
Output only the English translation, nothing else."""
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 4000
            }
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=500, detail="Conversion failed")
        result = resp.json()
        converted = result["choices"][0]["message"]["content"].strip()
        return {"converted": converted}

@app.post("/convert-segments")
async def convert_segments(request: dict):
    segments = request.get("segments", [])
    target = request.get("target", "hinglish")
    
    if target == "hindi" or not segments:
        return {"segments": segments}
    
    # Join all segment texts for batch conversion
    texts = [s.get("text", "") for s in segments]
    combined = "\n---\n".join(texts)
    
    if target == "hinglish":
        prompt = f"""Convert each Hindi text segment to Hinglish (Roman script phonetically). Keep English words as-is.
Each segment is separated by ---
Convert each one and return them separated by ---
Segments:
{combined}
Output only converted segments separated by ---, nothing else."""
    else:
        prompt = f"""Translate each Hindi/Hinglish segment to English naturally.
Each segment is separated by ---
Translate each one and return them separated by ---
Segments:
{combined}
Output only translated segments separated by ---, nothing else."""
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 4000
            }
        )
        if resp.status_code != 200:
            return {"segments": segments}
        
        result = resp.json()
        converted_text = result["choices"][0]["message"]["content"].strip()
        converted_parts = converted_text.split("---")
        
        converted_segments = []
        for i, seg in enumerate(segments):
            converted_seg = dict(seg)
            if i < len(converted_parts):
                converted_seg["text"] = converted_parts[i].strip()
            converted_segments.append(converted_seg)
        
        return {"segments": converted_segments}

@app.post("/create-order")
async def create_order(request: dict):
    plan = request.get("plan", "starter")
    prices = {"starter": 9900, "pro": 29900}  # in paise
    amount = prices.get(plan, 9900)
    
    import base64
    auth = base64.b64encode(f"{RAZORPAY_KEY_ID}:{RAZORPAY_KEY_SECRET}".encode()).decode()
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            "https://api.razorpay.com/v1/orders",
            headers={"Authorization": f"Basic {auth}", "Content-Type": "application/json"},
            json={"amount": amount, "currency": "INR", "receipt": f"order_{plan}"}
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=500, detail="Order creation failed")
        return resp.json()

@app.post("/verify-payment")
async def verify_payment(request: dict, authorization: str = Header(None)):
    user = await get_user_from_token(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
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
        raise HTTPException(status_code=400, detail="Invalid signature")
    
    # Update subscription
    from datetime import datetime
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(
                f"{SUPABASE_URL}/rest/v1/subscriptions",
                headers={
                    "apikey": SUPABASE_SERVICE_KEY,
                    "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                    "Content-Type": "application/json",
                    "Prefer": "resolution=merge-duplicates"
                },
                json={
                    "user_id": user["id"],
                    "plan": plan,
                    "updated_at": datetime.now().isoformat()
                }
            )
    except Exception as e:
        print(f"Subscription update error: {e}")
    
    return {"success": True, "plan": plan}

@app.get("/health")
async def health():
    return {"status": "ok"}
