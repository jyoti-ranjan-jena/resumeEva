import io
import os
import pdfplumber
import docx2txt
import tempfile

def extract_text_from_pdf_bytes(file_bytes):
    """Extract text from PDF bytes using pdfplumber."""
    text_parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)

def extract_text_from_docx_bytes(file_bytes):
    """Extract text from DOCX bytes using a temporary file for safe processing."""
    # Create a temporary file with .docx suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    # Extract text using docx2txt
    text = docx2txt.process(tmp_path)

    # Safely remove temporary file
    try:
        os.remove(tmp_path)
    except Exception:
        pass

    return text

def extract_text_from_uploaded_file(uploaded_file):
    """
    uploaded_file: a file-like object (e.g. from Streamlit uploader).
    Return: extracted text (str).
    """
    file_bytes = uploaded_file.read()
    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        return extract_text_from_pdf_bytes(file_bytes)
    if filename.endswith(".docx") or filename.endswith(".doc"):
        return extract_text_from_docx_bytes(file_bytes)
    if filename.endswith(".txt"):
        try:
            return file_bytes.decode("utf-8", errors="ignore")
        except Exception:
            return str(file_bytes)

    # Fallback: try PDF first, then decode as text
    try:
        return extract_text_from_pdf_bytes(file_bytes)
    except Exception:
        try:
            return file_bytes.decode("utf-8", errors="ignore")
        except Exception:
            return ""
