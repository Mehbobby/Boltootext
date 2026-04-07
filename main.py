from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Header
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

    fd = {"model": model, "response_format": "verbose_json", "prompt": """You are an advanced transcription AI.

Your task is to transcribe the given video/audio into Hindi (Devanagari script) with STRICT rules:

CORE INSTRUCTIONS:
- Do NOT translate the meaning under any condition.
- Transcribe EXACTLY what is spoken.
- Preserve original wording, sentence structure, and speaking style.

HINGLISH HANDLING:
- If the speaker uses English words, write them in Hindi based on pronunciation (phonetic transcription).
- Do NOT convert English words into pure Hindi meanings.

STYLE RULES:
- Keep the output natural, casual, and conversational.
- Do NOT formalize or improve grammar.
- Preserve fillers and pauses such as: "uh", "hmm", "matlab", "like", etc.
- Maintain the speaker’s tone and emotion.

EXAMPLES:
- "please is computer ko thik kardo" → "प्लीज़ इस कंप्यूटर को ठीक कर दो"
- "guys aaj hum marketing strategy seekhenge" → "गाइज आज हम मार्केटिंग स्ट्रेटेजी सीखेंगे"

STRICTLY AVOID:
- ❌ कृपया (instead of प्लीज़)
- ❌ translating or rephrasing
- ❌ correcting grammar

IMPORTANT:
- If unsure about a word, prioritize phonetic accuracy over meaning.

OUTPUT FORMAT:
- Output ONLY the final transcription in Hindi (Devanagari script).
- Do not add explanations, notes, or extra text.
"""}
    if language:
        fd["language"] = language

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                files={"file": (file.filename, fb, file.content_type or "audio/mpeg")},
                data=fd,
            )
        if resp.status_code != 200:
            err = resp.json().get("error", {})
            raise HTTPException(resp.status_code, err.get("message", "Groq error"))

        data = resp.json()
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
    fd = {"model": model, "response_format": "verbose_json", "prompt": """You are an advanced transcription AI.

Your task is to transcribe the given video/audio into Hindi (Devanagari script) with STRICT rules:

CORE INSTRUCTIONS:
- Do NOT translate the meaning under any condition.
- Transcribe EXACTLY what is spoken.
- Preserve original wording, sentence structure, and speaking style.

HINGLISH HANDLING:
- If the speaker uses English words, write them in Hindi based on pronunciation (phonetic transcription).
- Do NOT convert English words into pure Hindi meanings.

STYLE RULES:
- Keep the output natural, casual, and conversational.
- Do NOT formalize or improve grammar.
- Preserve fillers and pauses such as: "uh", "hmm", "matlab", "like", etc.
- Maintain the speaker’s tone and emotion.

EXAMPLES:
- "please is computer ko thik kardo" → "प्लीज़ इस कंप्यूटर को ठीक कर दो"
- "guys aaj hum marketing strategy seekhenge" → "गाइज आज हम मार्केटिंग स्ट्रेटेजी सीखेंगे"

STRICTLY AVOID:
- ❌ कृपया (instead of प्लीज़)
- ❌ translating or rephrasing
- ❌ correcting grammar

IMPORTANT:
- If unsure about a word, prioritize phonetic accuracy over meaning.

OUTPUT FORMAT:
- Output ONLY the final transcription in Hindi (Devanagari script).
- Do not add explanations, notes, or extra text.
"""}
    if language: fd["language"] = language
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                files={"file": (file.filename, fb, file.content_type or "audio/mpeg")},
                data=fd,
            )
        if resp.status_code != 200:
            err = resp.json().get("error", {})
            raise HTTPException(resp.status_code, err.get("message", "Groq error"))
        return resp.json()
    except httpx.TimeoutException:
        raise HTTPException(504, "Timeout! Try again.")
