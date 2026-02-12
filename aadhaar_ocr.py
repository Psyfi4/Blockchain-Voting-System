# aadhaar_ocr.py (add or replace)
import re
import easyocr
from PIL import Image

# Initialize OCR reader ONCE
reader = easyocr.Reader(["en"], gpu=False)

def _clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def scan_aadhaar_image_from_pil(pil_image: Image.Image) -> dict:
    """
    Aadhaar OCR using EasyOCR (NO OpenCV, NO Tesseract)
    """
    results = reader.readtext(pil_image)

    full_text = " ".join([r[1] for r in results])
    full_text = _clean_text(full_text)

    aadhaar_match = re.search(r"\d{4}\s?\d{4}\s?\d{4}", full_text)
    dob_match = re.search(r"\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}", full_text)

    return {
        "aadhaar_number": aadhaar_match.group(0).replace(" ", "") if aadhaar_match else None,
        "date_of_birth": dob_match.group(0) if dob_match else None,
        "raw_text": full_text
    }
