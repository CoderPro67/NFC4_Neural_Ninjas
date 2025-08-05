#logic.py

import io
import json
from pathlib import Path

import docx
import fitz  # PyMuPDF
from fastapi import UploadFile

# --- All the logic from your redaction_engine.py goes here ---
import re
from transformers import pipeline

class PIIRedactor:
    def __init__(self, model_name="Jean-Baptiste/roberta-large-ner-english"):
        print(f"Loading NER model ({model_name})... This might take a moment.")
        # IMPORTANT: Specify device=0 to use GPU if available, otherwise it will use CPU.
        # For CPU-only, you can use device=-1 or remove the argument.
        try:
            self.ner_pipeline = pipeline("ner", model=model_name, grouped_entities=True, device=0)
            print("NER model loaded on GPU.")
        except Exception:
            print("GPU not available or CUDA not set up correctly. Loading NER model on CPU.")
            self.ner_pipeline = pipeline("ner", model=model_name, grouped_entities=True)
            print("NER model loaded on CPU.")


        phone_regex = r"""
            \b
            (?:(?:\+91|0)[\s-]?)?[6-9]\d{2}[\s-]?\d{3}[\s-]?\d{4}\b|
            \b0\d{2,4}[\s-]?\d{6,8}\b
        """
        self.regex_patterns = {
            "EMAIL": re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'),
            "PHONE": re.compile(phone_regex, re.VERBOSE),
            "AADHAAR": re.compile(r'\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b'),
            "PAN_CARD": re.compile(r'\b[A-Z]{5}\d{4}[A-Z]{1}\b'),
            "ACCOUNT_NO": re.compile(r'\b\d{9,18}\b'),
        }
        self.entity_priorities = {
            'AADHAAR': 1, 'PAN_CARD': 1, 'PHONE': 2, 'EMAIL': 2, 'PER': 3,
            'ORG': 3, 'LOC': 4, 'DATE': 4, 'ADDRESS': 5, 'PINCODE': 5,
            'ACCOUNT_NO': 99,
        }
        self.compliance_map = {
            "GDPR": ["PER", "LOC", "EMAIL", "PHONE", "DATE"],
            "HIPAA": ["PER", "PHONE", "DATE", "LOC", "AGE", "ID"],
            "DPDP": ["PER", "EMAIL", "PHONE", "ACCOUNT_NO", "AADHAAR", "PAN_CARD", "LOC", "PINCODE"],
            "FULL_REDACTION": ["PER", "ORG", "LOC", "EMAIL", "PHONE", "ACCOUNT_NO", "AADHAAR", "PAN_CARD", "DATE", "PINCODE"]
        }

    def _resolve_overlaps(self, entities):
        if not entities: return []
        entities.sort(key=lambda x: (x['start'], -(x['end'] - x['start'])))
        resolved = []
        last_entity = None
        for current_entity in entities:
            if last_entity is None:
                last_entity = current_entity
                continue
            if current_entity['start'] < last_entity['end']:
                last_priority = self.entity_priorities.get(last_entity['entity_group'], 100)
                current_priority = self.entity_priorities.get(current_entity['entity_group'], 100)
                if current_priority < last_priority:
                    last_entity = current_entity
            else:
                resolved.append(last_entity)
                last_entity = current_entity
        if last_entity is not None:
            resolved.append(last_entity)
        return resolved

    def detect_pii(self, text):
        ner_results = self.ner_pipeline(text)
        for entity in ner_results: entity['score'] = float(entity['score'])
        regex_results = []
        for entity_type, pattern in self.regex_patterns.items():
            for match in pattern.finditer(text):
                regex_results.append({
                    'entity_group': entity_type, 'score': 1.0, 'word': match.group(0),
                    'start': match.start(), 'end': match.end()
                })
        all_entities = ner_results + regex_results
        resolved_entities = self._resolve_overlaps(all_entities)
        resolved_entities.sort(key=lambda x: x['start'])
        return resolved_entities

    def _stitch_address_entities(self, entities, text, max_gap=15):
        stitched_entities = []
        i = 0
        address_components = {"LOC", "PINCODE"}
        while i < len(entities):
            current_entity = entities[i]
            if current_entity['entity_group'] in address_components:
                address_block = [current_entity]
                j = i + 1
                while j < len(entities):
                    next_entity = entities[j]
                    if next_entity['entity_group'] in address_components:
                        gap_text = text[address_block[-1]['end']:next_entity['start']]
                        if len(gap_text) < max_gap and all(c in ',- \n\t' for c in gap_text):
                            address_block.append(next_entity)
                            j += 1
                        else: break
                    else: break
                if len(address_block) > 1:
                    start_char, end_char = address_block[0]['start'], address_block[-1]['end']
                    stitched_entities.append({
                        'entity_group': 'ADDRESS', 'score': min(e['score'] for e in address_block),
                        'word': text[start_char:end_char], 'start': start_char, 'end': end_char
                    })
                    i = j
                else:
                    stitched_entities.append(current_entity)
                    i += 1
            else:
                stitched_entities.append(current_entity)
                i += 1
        return stitched_entities

    def redact(self, text, compliance_mode="DPDP", agentic_level=0.75, aggressive=False):
        raw_entities = self.detect_pii(text)
        processed_entities = self._stitch_address_entities(raw_entities, text)
        entities_to_redact_types = self.compliance_map.get(compliance_mode, []).copy()
        if "LOC" in entities_to_redact_types or "PINCODE" in entities_to_redact_types:
            if "ADDRESS" not in entities_to_redact_types:
                entities_to_redact_types.append("ADDRESS")
        filtered_entities = [e for e in processed_entities if e['entity_group'] in entities_to_redact_types]
        filtered_entities.sort(key=lambda x: x['start'], reverse=True)
        redacted_text = text
        for entity in filtered_entities:
            start, end = entity['start'], entity['end']
            entity_type, score = entity['entity_group'], entity['score']
            if aggressive or score >= agentic_level:
                redaction_marker = f"[{entity_type}]"
            else:
                redaction_marker = f"[NEEDS_REVIEW: {entity['word']} ({entity_type})]"
            redacted_text = redacted_text[:start] + redaction_marker + redacted_text[end:]
        return redacted_text

    def redact_json_recursively(self, data, compliance_mode, agentic_level, aggressive):
        if isinstance(data, dict):
            return {k: self.redact_json_recursively(v, compliance_mode, agentic_level, aggressive) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.redact_json_recursively(i, compliance_mode, agentic_level, aggressive) for i in data]
        elif isinstance(data, str):
            return self.redact(data, compliance_mode, agentic_level, aggressive)
        else:
            return data

# --- IMPORTANT: Instantiate the model ONCE at the module level ---
# This ensures the heavy ML model is loaded only when the server starts.
redactor = PIIRedactor()


# The PIIRedactor class remains the same...

# ... (scroll down to the handle_uploaded_file function)

# MODIFIED: The function now accepts the redaction parameters
async def handle_uploaded_file(file: UploadFile, compliance_mode: str, agentic_level: float, aggressive: bool):
    """
    Reads an uploaded file, extracts text, performs redaction using dynamic parameters,
    and returns the original and redacted content.
    """
    filename = file.filename
    content = await file.read()
    file_suffix = Path(filename).suffix.lower()

    original_text = ""
    redacted_text = ""
    
    try:
        if file_suffix == ".json":
            original_data = json.loads(content)
            # MODIFIED: Use the passed-in parameters
            redacted_data = redactor.redact_json_recursively(
                original_data,
                compliance_mode=compliance_mode,
                agentic_level=agentic_level,
                aggressive=aggressive
            )
            original_text = json.dumps(original_data, indent=2)
            redacted_text = json.dumps(redacted_data, indent=2)
        
        elif file_suffix in [".txt", ".docx", ".pdf"]:
            if file_suffix == ".txt":
                original_text = content.decode("utf-8", errors="ignore")
            elif file_suffix == ".docx":
                doc = docx.Document(io.BytesIO(content))
                original_text = "\n".join([para.text for para in doc.paragraphs])
            elif file_suffix == ".pdf":
                with fitz.open(stream=content, filetype="pdf") as doc:
                    original_text = "".join(page.get_text() for page in doc)
            
            # MODIFIED: Use the passed-in parameters for all text-based files
            redacted_text = redactor.redact(
                original_text,
                compliance_mode=compliance_mode,
                agentic_level=agentic_level,
                aggressive=aggressive
            )
            
        else:
            return {
                "filename": filename,
                "error": f"Unsupported file type: {file_suffix}",
                "status": "error"
            }

        return {
            "filename": filename,
            "original_text": original_text,
            "redacted_text": redacted_text,
            "message": f"File redacted successfully with mode: {compliance_mode}",
            "status": "success"
        }

    except Exception as e:
        print(f"Error processing file {filename}: {e}")
        return {
            "filename": filename,
            "error": f"An error occurred while processing the file: {str(e)}",
            "status": "error"
        }