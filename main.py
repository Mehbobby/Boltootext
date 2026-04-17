from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Header
import tempfile, os, subprocess, re
from fastapi.middleware.cors import CORSMiddleware
import httpx, os, hmac, hashlib
from supabase import create_client
from datetime import datetime, timedelta, timezone

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

GROQ_API_KEY         = os.environ.get("GROQ_API_KEY", "")
SUPABASE_URL         = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
RAZORPAY_KEY_ID      = os.environ.get("RAZORPAY_KEY_ID", "")
RAZORPAY_KEY_SECRET  = os.environ.get("RAZORPAY_KEY_SECRET", "")

PLAN_LIMITS = {"free": 10*60, "starter": 120*60, "pro": 99999*60}
PLAN_PRICES = {"starter": 9900, "pro": 29900}

def sb():
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

def cur_month():
    return datetime.now().strftime("%Y-%m")

def get_user(token):
    try:
        return sb().auth.get_user(token).user
    except:
        raise HTTPException(401, "Login karo pehle!")

def get_plan(uid):
    try:
        r = sb().table("subscriptions").select("plan,valid_until").eq("user_id", uid).single().execute()
        if r.data:
            v = r.data.get("valid_until")
            if v and datetime.fromisoformat(v.replace("Z","+00:00")) < datetime.now(timezone.utc):
                return "free"
            return r.data["plan"]
    except:
        pass
    return "free"

def get_usage(uid):
    try:
        r = sb().table("usage").select("seconds_used").eq("user_id", uid).eq("month_year", cur_month()).single().execute()
        return r.data["seconds_used"] if r.data else 0
    except:
        return 0

def add_usage(uid, secs):
    m = cur_month()
    try:
        ex = sb().table("usage").select("id,seconds_used").eq("user_id", uid).eq("month_year", m).single().execute()
        if ex.data:
            sb().table("usage").update({
                "seconds_used": ex.data["seconds_used"] + secs,
                "updated_at": datetime.now().isoformat()
            }).eq("id", ex.data["id"]).execute()
        else:
            sb().table("usage").insert({
                "user_id": uid, "seconds_used": secs, "month_year": m
            }).execute()
    except Exception as e:
        print(f"Usage err: {e}")

def verify_razorpay_signature(order_id, payment_id, signature):
    msg = f"{order_id}|{payment_id}"
    expected = hmac.new(RAZORPAY_KEY_SECRET.encode(), msg.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


@app.get("/")
def root():
    return {"status": "BolToText API running ✅"}


@app.get("/me")
async def me(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Token nahi mila")
    user = get_user(authorization.split(" ")[1])
    uid  = str(user.id)
    plan  = get_plan(uid)
    used  = get_usage(uid)
    limit = PLAN_LIMITS[plan]
    return {
        "email": user.email,
        "plan": plan,
        "usage_seconds": used,
        "limit_seconds": limit,
        "remaining_seconds": max(0, limit - used)
    }


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form("whisper-large-v3-turbo"),
    language: str = Form(""),
    authorization: str = Header(None),
):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Login karo pehle!")

    user = get_user(authorization.split(" ")[1])
    uid  = str(user.id)
    plan  = get_plan(uid)
    used  = get_usage(uid)
    limit = PLAN_LIMITS[plan]

    if used >= limit:
        raise HTTPException(429, f"LIMIT_EXCEEDED|{plan}|{used}|{limit}")

    fb = await file.read()
    if len(fb) > 25 * 1024 * 1024:
        raise HTTPException(413, "25MB se badi file nahi chalegi!")

    try:
        data = await call_groq_whisper(fb, file.filename, file.content_type, model, language)
        dur  = int(data.get("duration", 0)) + 1
        add_usage(uid, dur)
        new_used = used + dur
        return {
            **data,
            "usage": {
                "used": new_used,
                "limit": limit,
                "plan": plan,
                "remaining": max(0, limit - new_used)
            }
        }
    except httpx.TimeoutException:
        raise HTTPException(504, "Timeout! Dobara try karo.")


@app.post("/create-order")
async def create_order(plan: str = Form(...), authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Login required")
    if plan not in PLAN_PRICES:
        raise HTTPException(400, "Invalid plan")

    user = get_user(authorization.split(" ")[1])

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.razorpay.com/v1/orders",
            auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET),
            json={
                "amount": PLAN_PRICES[plan],
                "currency": "INR",
                "notes": {"user_id": str(user.id), "plan": plan}
            }
        )
    if resp.status_code != 200:
        raise HTTPException(500, "Order create nahi hua")

    order = resp.json()
    return {
        "order_id": order["id"],
        "amount": PLAN_PRICES[plan],
        "currency": "INR",
        "email": user.email
    }


@app.post("/verify-payment")
async def verify_payment(
    razorpay_order_id: str   = Form(...),
    razorpay_payment_id: str = Form(...),
    razorpay_signature: str  = Form(...),
    plan: str                = Form(...),
    authorization: str       = Header(None),
):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Login required")

    user = get_user(authorization.split(" ")[1])
    uid  = str(user.id)

    if not verify_razorpay_signature(razorpay_order_id, razorpay_payment_id, razorpay_signature):
        raise HTTPException(400, "Payment verify nahi hua!")

    valid_until = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
    try:
        ex = sb().table("subscriptions").select("id").eq("user_id", uid).single().execute()
        if ex.data:
            sb().table("subscriptions").update({
                "plan": plan,
                "valid_until": valid_until,
                "razorpay_payment_id": razorpay_payment_id
            }).eq("user_id", uid).execute()
        else:
            sb().table("subscriptions").insert({
                "user_id": uid,
                "plan": plan,
                "valid_until": valid_until,
                "razorpay_payment_id": razorpay_payment_id
            }).execute()
    except Exception as e:
        raise HTTPException(500, str(e))

    return {"success": True, "plan": plan}


async def call_groq_whisper(fb: bytes, filename: str, content_type: str, model: str, language: str) -> dict:
    """Call Groq Whisper API and return result"""
    fd = {
        "model": model,
        "response_format": "verbose_json",
        "temperature": "0",
    }
    # Only set language if user explicitly chose one
    # For Hinglish (auto), let Whisper detect — forced "hi" causes translation issues
    if language:
        fd["language"] = language

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            files={"file": (filename, fb, content_type or "audio/mpeg")},
            data=fd,
        )
    if resp.status_code != 200:
        err = resp.json().get("error", {})
        raise HTTPException(resp.status_code, err.get("message", "Groq error"))
    return resp.json()


@app.post("/transcribe-guest")
async def transcribe_guest(
    file: UploadFile = File(...),
    model: str = Form("whisper-large-v3-turbo"),
    language: str = Form(""),
):
    """No auth needed — guest gets 4 min, tracked client-side"""
    fb = await file.read()
    if len(fb) > 25*1024*1024:
        raise HTTPException(413, "25MB se badi file nahi chalegi!")
    try:
        data = await call_groq_whisper(fb, file.filename, file.content_type, model, language)
        return data
    except httpx.TimeoutException:
        raise HTTPException(504, "Timeout! Try again.")


@app.post("/transcribe-url")
async def transcribe_url(
    url: str = Form(...),
    model: str = Form("whisper-large-v3-turbo"),
    language: str = Form(""),
    authorization: str = Header(None),
):
    """Transcribe from YouTube/Instagram/Facebook URL"""

    # Validate URL
    allowed = ['youtube.com', 'youtu.be', 'instagram.com', 'facebook.com', 'fb.watch', 'fb.com']
    if not any(d in url for d in allowed):
        raise HTTPException(400, "Only YouTube, Instagram, and Facebook links are supported.")

    # Check auth & usage
    if authorization and authorization.startswith("Bearer "):
        try:
            user = get_user(authorization.split(" ")[1])
            uid  = str(user.id)
            plan = get_plan(uid)
            used = get_usage(uid)
            limit = PLAN_LIM_SEC.get(plan, 600)
            if used >= limit:
                raise HTTPException(429, f"LIMIT_EXCEEDED|{plan}|{used}|{limit}")
            is_logged_in = True
        except HTTPException:
            raise
        except:
            is_logged_in = False
            uid = None
    else:
        is_logged_in = False
        uid = None

    # Download audio using yt-dlp
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, 'audio.%(ext)s')
        cmd = [
            'yt-dlp',
            '--extract-audio',
            '--audio-format', 'mp3',
            '--audio-quality', '5',
            '--max-filesize', '25m',
            '--output', out_path,
            '--no-playlist',
            '--quiet',
            url
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                err = result.stderr.lower()
                if 'private' in err or 'login' in err:
                    raise HTTPException(400, "This video is private or requires login.")
                if 'not available' in err or 'unavailable' in err:
                    raise HTTPException(400, "Video not available. Check the link.")
                raise HTTPException(400, "Could not download video. Try a different link.")
        except subprocess.TimeoutExpired:
            raise HTTPException(504, "Download timed out. Try a shorter video.")

        # Find downloaded file
        audio_file = None
        for f in os.listdir(tmpdir):
            if f.startswith('audio.'):
                audio_file = os.path.join(tmpdir, f)
                break

        if not audio_file:
            raise HTTPException(500, "Audio extraction failed.")

        # Check file size
        fsize = os.path.getsize(audio_file)
        if fsize > 25 * 1024 * 1024:
            raise HTTPException(413, "Audio too large (max 25MB). Try a shorter video.")

        with open(audio_file, 'rb') as f:
            fb = f.read()

        # Transcribe
        data = await call_groq_whisper(fb, 'audio.mp3', 'audio/mpeg', model, language)
        dur = int(data.get("duration", 0)) + 1

        if is_logged_in and uid:
            add_usage(uid, dur)
            plan = get_plan(uid)
            used = get_usage(uid)
            limit = PLAN_LIM_SEC.get(plan, 600)
            return {**data, "usage": {"used": used, "limit": limit, "plan": plan, "remaining": max(0, limit-used)}}

        return data


PLAN_LIM_SEC = {"free": 600, "starter": 7200, "pro": 30000}


@app.post("/convert")
async def convert_text(
    text: str = Form(...),
    target: str = Form(...),  # "hinglish" or "english"
):
    """Convert Hindi transcript to Hinglish or English using LLaMA"""

    if target == "hinglish":
        prompt = f"""Convert this Hindi transcript to Hinglish (Roman script). 
Rules:
- Write exactly as an Indian would speak casually in Roman letters
- Mix Hindi and English naturally like real conversation
- Do NOT translate meaning, just convert the script/style
- Keep English words as English
- Example: "मैं कल घर जाऊंगा" → "Main kal ghar jaunga"
- Example: "यह बहुत अच्छा है" → "Yeh bahut accha hai"

Hindi transcript:
{text}

Hinglish output (only output the converted text, nothing else):"""

    elif target == "english":
        prompt = f"""Translate this Hindi/Hinglish transcript to natural English.
- Keep the meaning accurate
- Use natural conversational English
- Do not add extra words or change meaning

Transcript:
{text}

English translation (only output the translation, nothing else):"""
    else:
        raise HTTPException(400, "Invalid target. Use 'hinglish' or 'english'")

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 4000,
                }
            )
        if resp.status_code != 200:
            raise HTTPException(500, "Conversion failed")
        
        data = resp.json()
        converted = data["choices"][0]["message"]["content"].strip()
        return {"text": converted, "target": target}

    except httpx.TimeoutException:
        raise HTTPException(504, "Conversion timed out. Try again.")
