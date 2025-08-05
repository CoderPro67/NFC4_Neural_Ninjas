# main.py

import argparse
from pathlib import Path
import docx
import fitz  # PyMuPDF
import json
from redaction_engine import PIIRedactor

def read_text_from_unstructured_file(filepath):
    """Reads raw text content from TXT, DOCX, and PDF files."""
    path = Path(filepath)
    text = ""
    if path.suffix == ".txt":
        text = path.read_text(encoding='utf-8')
    elif path.suffix == ".docx":
        doc = docx.Document(filepath)
        text = "\n".join([para.text for para in doc.paragraphs])
    elif path.suffix == ".pdf":
        with fitz.open(filepath) as doc:
            text = "".join(page.get_text() for page in doc)
    return text

def redact_json_recursively(data, redactor, compliance_mode, agentic_level, aggressive):
    """Recursively traverses a JSON object and redacts string values."""
    if isinstance(data, dict):
        return {k: redact_json_recursively(v, redactor, compliance_mode, agentic_level, aggressive) for k, v in data.items()}
    elif isinstance(data, list):
        return [redact_json_recursively(i, redactor, compliance_mode, agentic_level, aggressive) for i in data]
    elif isinstance(data, str):
        return redactor.redact(data, compliance_mode, agentic_level, aggressive)
    else:
        return data

def main():
    parser = argparse.ArgumentParser(description="AI-Powered Data Privacy Redaction Tool")
    parser.add_argument("input_file", type=str, help="Path to the input file (txt, docx, pdf, json).")
    parser.add_argument("-o", "--output_file", type=str, help="Path to save the redacted output file.")
    parser.add_argument(
        "-m", "--mode",
        type=str,
        default="DPDP",
        choices=["GDPR", "HIPAA", "DPDP", "FULL_REDACTION"],
        help="Compliance mode for redaction. Defaults to DPDP."
    )
    parser.add_argument(
        "-l", "--level",
        type=float,
        default=0.75,
        help="Agentic confidence level (0.0 to 1.0). Lower to redact more. Default: 0.75"
    )
    parser.add_argument(
        "-a", "--aggressive",
        action="store_true",
        help="Enable aggressive mode. Redacts all detected entities regardless of confidence."
    )
    
    args = parser.parse_args()
    input_path = Path(args.input_file)

    if not input_path.exists():
        print(f"Error: Input file not found at {args.input_file}")
        return

    try:
        redactor = PIIRedactor()
        
        if input_path.suffix == ".json":
            print(f"Processing JSON file: {args.input_file}...")
            with open(input_path, 'r', encoding='utf-8') as f:
                original_data = json.load(f)
            
            redacted_data = redact_json_recursively(
                original_data, redactor, args.mode, args.level, args.aggressive
            )
            output_content = json.dumps(redacted_data, indent=2)

        elif input_path.suffix in [".txt", ".docx", ".pdf"]:
            print(f"Reading text from {args.input_file}...")
            original_text = read_text_from_unstructured_file(args.input_file)
            
            print(f"Performing redaction with mode='{args.mode}', level={args.level}, aggressive={args.aggressive}...")
            redacted_text = redactor.redact(
                original_text, compliance_mode=args.mode, agentic_level=args.level, aggressive=args.aggressive
            )
            output_content = redacted_text
        else:
            print(f"Error: Unsupported file type '{input_path.suffix}'.")
            return
            
        if args.output_file:
            Path(args.output_file).write_text(output_content, encoding='utf-8')
            print(f"\nRedacted output saved to: {args.output_file}")
        else:
            print("\n--- REDACTED OUTPUT ---")
            print(output_content)
            print("--- END OF OUTPUT ---")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()