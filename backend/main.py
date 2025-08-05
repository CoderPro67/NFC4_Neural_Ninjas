from fastapi import UploadFile, File, Form
from logic import handle_uploaded_file
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)

# --- MODIFICATION IS HERE ---
@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    mode: str = Form("DPDP"),
    level: float = Form(0.75),
    # 1. Accept 'aggressive' as a string from the form.
    aggressive_str: str = Form("false") 
):
    """
    Receives a file and redaction parameters from the frontend.
    """
    # 2. Reliably convert the string ('true'/'false') to a real boolean.
    # This is much safer than relying on automatic type conversion.
    aggressive = aggressive_str.lower() in ('true', '1', 't')

    # 3. Pass the clean boolean to the logic handler.
    return await handle_uploaded_file(file, compliance_mode=mode, agentic_level=level, aggressive=aggressive)


@app.get("/")
def read_root():
    return {"message": "Welcome to the AI Redaction API. Use the /upload endpoint to process files."}