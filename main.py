from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
import httpx, os, razorpay
from supabase import create_client, Client
from datetime import datetime, timedelta, timezone

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

GROQ_API_KEY         = os.environ.get("GROQ_API_KEY")
SUPABASE_URL         = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "")
RAZORPAY_KEY_ID      = os.environ.get("RAZORPAY_KEY_ID")
RAZORPAY_KEY_SECRET  = os.environ.get("RAZORPAY_KEY_SECRET")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
rz = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))

PLAN_LIMITS = {"free": 10*60, "starter": 120*60, "pro": 99999*60}
PLAN_PRICES = {"starter": 9900, "pro": 29900}

def cur_month(): return datetime.now().strftime("%Y-%m")

def get_user(token):
    try: return supabase.auth.get_user(token).user
    except: raise HTTPException(401, "Login karo pehle!")

def get_plan(uid):
    try:
        r = supabase.table("subscriptions").select("plan,valid_until").eq("user_id", uid).single().execute()
        if r.data:
            v = r.data.get("valid_until")
            if v and datetime.fromisoformat(v.replace("Z","+00:00")) < datetime.now(timezone.utc):
                return "free"
            return r.data["plan"]
    except: pass
    return "free"

def get_usage(uid):
    try:
        r = supabase.table("usage").select("seconds_used").eq("user_id", uid).eq("month_year", cur_month()).single().execute()
        return r.data["seconds_used"] if r.data else 0
    except: return 0

def add_usage(uid, secs):
    m = cur_month()
    try:
        ex = supabase.table("usage").select("id,seconds_used").eq("user_id", uid).eq("month_year", m).single().execute()
        if ex.data:
            supabase.table("usage").update({"seconds_used": ex.data["seconds_used"]+secs, "updated_at": datetime.now().isoformat()}).eq("id", ex.data["id"]).execute()
        else:
            supabase.table("usage").insert({"user_id": uid, "seconds_used": secs, "month_year": m}).execute()
    except Exception as e: print(f"Usage err: {e}")


@app.get("/")
def root(): return {"status": "BolToText API running"}

@app.get("/me")
async def me(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Token nahi mila")
    user = get_user(authorization.split(" ")[1])
    uid  = str(user.id)
    plan = get_plan(uid); used = get_usage(uid); limit = PLAN_LIMITS[plan]
    return {"email": user.email, "plan": plan, "usage_seconds": used, "limit_seconds": limit, "remaining_seconds": max(0, limit-used)}

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
    plan = get_plan(uid); used = get_usage(uid); limit = PLAN_LIMITS[plan]
    if used >= limit:
        raise HTTPException(429, f"LIMIT_EXCEEDED|{plan}|{used}|{limit}")
    fb = await file.read()
    if len(fb) > 25*1024*1024:
        raise HTTPException(413, "25MB se badi file nahi chalegi!")
    fd = {"model": model, "response_format": "verbose_json"}
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
        data = resp.json()
        dur  = int(data.get("duration", 0)) + 1
        add_usage(uid, dur)
        new_used = used + dur
        return {**data, "usage": {"used": new_used, "limit": limit, "plan": plan, "remaining": max(0, limit-new_used)}}
    except httpx.TimeoutException:
        raise HTTPException(504, "Timeout! Dobara try karo.")

@app.post("/create-order")
async def create_order(plan: str = Form(...), authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Login required")
    if plan not in PLAN_PRICES: raise HTTPException(400, "Invalid plan")
    user  = get_user(authorization.split(" ")[1])
    order = rz.order.create({"amount": PLAN_PRICES[plan], "currency": "INR", "notes": {"user_id": str(user.id), "plan": plan}})
    return {"order_id": order["id"], "amount": PLAN_PRICES[plan], "currency": "INR", "email": user.email}

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
    user = get_user(authorization.split(" ")[1]); uid = str(user.id)
    try:
        rz.utility.verify_payment_signature({"razorpay_order_id": razorpay_order_id, "razorpay_payment_id": razorpay_payment_id, "razorpay_signature": razorpay_signature})
    except: raise HTTPException(400, "Payment verify nahi hua!")
    valid_until = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
    try:
        ex = supabase.table("subscriptions").select("id").eq("user_id", uid).single().execute()
        if ex.data:
            supabase.table("subscriptions").update({"plan": plan, "valid_until": valid_until, "razorpay_payment_id": razorpay_payment_id}).eq("user_id", uid).execute()
        else:
            supabase.table("subscriptions").insert({"user_id": uid, "plan": plan, "valid_until": valid_until, "razorpay_payment_id": razorpay_payment_id}).execute()
    except Exception as e: raise HTTPException(500, str(e))
    return {"success": True, "plan": plan}
