# main.py

import os
import io
import docx
import fitz  # PyMuPDF
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Initialization ---
load_dotenv()
app = FastAPI(title="Document Redaction API")

# Allow CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Variables & Model Loading ---
LLM = None
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env file.")
    LLM = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=api_key)
    print("✅ Gemini model initialized successfully.")
except Exception as e:
    print(f"❌ Critical Error: Could not initialize Gemini model. {e}")

COMPLIANCE_MAP = {
    "gdpr": ["names", "emails", "phones", "physical mailing addresses", "IP addresses", "social security numbers", "passport numbers"],
    "hipaa": ["names", "dates", "phone numbers", "emails", "social security numbers", "medical record numbers"],
    "dpdp": ["names", "emails", "phones", "addresses", "Aadhaar numbers", "PAN numbers", "financial data"]
}

# --- Core Redaction Logic ---
def get_redaction_chain(entity_types: list):
    """Creates a LangChain chain for a given set of PII types."""
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are an AI assistant that redacts Personally Identifiable Information (PII).
        - Detect these PII types: {entity_types}
        - Replace each PII instance with "[REDACTED]".
        - Return only the redacted text, preserving original structure and line breaks."""),
        ("human", "Document Text:\n---\n{document_text}\n---")
    ])
    return prompt_template | LLM | StrOutputParser()

# --- API Endpoint ---
@app.post("/redact-file")
async def redact_file(
    compliance_mode: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Receives a file, redacts it based on its type and compliance mode,
    and returns the redacted file.
    """
    if not LLM:
        raise HTTPException(status_code=500, detail="LLM not initialized.")

    entity_types = COMPLIANCE_MAP.get(compliance_mode)
    if not entity_types:
        raise HTTPException(status_code=400, detail="Invalid compliance mode.")

    chain = get_redaction_chain(entity_types)
    file_content = await file.read()
    file_extension = os.path.splitext(file.filename)[1].lower()

    # --- Process based on file type ---
    if file_extension == ".pdf":
        # True PDF redaction: add black boxes
        try:
            doc = fitz.open(stream=file_content, filetype="pdf")
            for page in doc:
                text = page.get_text("text")
                if text.strip():
                    # To find PII to redact, we first get a list from the LLM
                    # A more advanced approach would be to get PII and its context
                    redacted_page_text = chain.invoke({"document_text": text, "entity_types": ", ".join(entity_types)})
                    # This is a simplified approach: find text differences
                    # A robust solution would use fuzzy matching or coordinate-based redaction
                    # For this example, we'll redact the whole page if PII is found.
                    # This part can be significantly improved.
                    pii_words = [word for word in text.split() if word not in redacted_page_text]
                    for word in pii_words:
                         areas = page.search_for(word)
                         for rect in areas:
                            page.add_redact_annot(rect, fill=(0, 0, 0))
                page.apply_redactions()
            
            output_buffer = io.BytesIO()
            doc.save(output_buffer)
            output_buffer.seek(0)
            media_type = "application/pdf"

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"PDF processing error: {e}")

    elif file_extension == ".docx":
        # Edit DOCX paragraphs
        try:
            doc = docx.Document(io.BytesIO(file_content))
            for para in doc.paragraphs:
                if para.text.strip():
                    redacted_text = chain.invoke({"document_text": para.text, "entity_types": ", ".join(entity_types)})
                    para.text = redacted_text
            
            output_buffer = io.BytesIO()
            doc.save(output_buffer)
            output_buffer.seek(0)
            media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"DOCX processing error: {e}")

    elif file_extension in [".txt", ".json"]:
        # Simple text redaction
        try:
            text = file_content.decode('utf-8')
            redacted_text = chain.invoke({"document_text": text, "entity_types": ", ".join(entity_types)})
            output_buffer = io.BytesIO(redacted_text.encode('utf-8'))
            media_type = "text/plain"

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Text processing error: {e}")

    else:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    return StreamingResponse(output_buffer, media_type=media_type, headers={
        "Content-Disposition": f"attachment; filename=redacted_{file.filename}"
    })

@app.get("/")
def read_root():
    return {"status": "Redaction API is running"}