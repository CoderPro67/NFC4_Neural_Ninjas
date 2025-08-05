# main.py
from fastapi import UploadFile, File, Form, HTTPException
from logic import handle_uploaded_file
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="AI Redaction API", version="1.0.0")

# FIXED: Added more origins and your specific port
origins = [
    "http://localhost",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    # Add any other ports you might be using for your frontend
    "*"  # For development only - remove in production
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# FIXED: Changed endpoint from /upload to /redact to match your frontend
@app.post("/redact")
async def redact_file(
    file: UploadFile = File(...),
    mode: str = Form("DPDP"),
    level: float = Form(0.75),
    aggressive: str = Form("false")
):
    """
    Receives a file and redaction parameters from the frontend.
    """
    try:
        # Convert string to boolean safely
        aggressive_bool = aggressive.lower() in ('true', '1', 't', 'yes')
        
        # Process the file
        result = await handle_uploaded_file(
            file, 
            compliance_mode=mode, 
            agentic_level=level, 
            aggressive=aggressive_bool
        )
        
        return result
        
    except Exception as e:
        print(f"Error in redact_file endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# OPTIONAL: Keep the upload endpoint as well for backward compatibility
@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    mode: str = Form("DPDP"),
    level: float = Form(0.75),
    aggressive: str = Form("false")
):
    """
    Alternative endpoint - redirects to redact_file
    """
    return await redact_file(file, mode, level, aggressive)

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the AI Redaction API", 
        "endpoints": {
            "/redact": "POST - Upload and redact files",
            "/upload": "POST - Alternative upload endpoint",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "AI Redaction API"}

# ADDED: Direct run capability
if __name__ == "__main__":
    print("Starting AI Redaction API server...")
    print("Frontend should connect to: http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")