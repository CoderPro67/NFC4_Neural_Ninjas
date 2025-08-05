# backend/logic.py

async def handle_uploaded_file(file):
    content = await file.read()
    text = content.decode("utf-8", errors="ignore")

    redacted = text
    for word in ['email', 'phone', 'name', 'address', 'SSN', 'credit card']:
        redacted = redacted.replace(word, '[REDACTED]')

    return {
        "filename": file.filename,
        "original_text": text,
        "redacted_text": redacted,
        "message": "File uploaded and redacted successfully!",
        "status": "success"
    }
