import re
import easyocr
from PIL import Image

reader = easyocr.Reader(['en'], gpu=False)

def scan_aadhaar_image_from_pil(img: Image.Image):
    results = reader.readtext(img)
    text = " ".join([r[1] for r in results])

    aadhaar = re.search(r"\d{4}\s?\d{4}\s?\d{4}", text)
    dob = re.search(r"\d{2}[/-]\d{2}[/-]\d{4}", text)

    return {
        "aadhaar_number": aadhaar.group(0).replace(" ", "") if aadhaar else None,
        "date_of_birth": dob.group(0) if dob else None,
        "raw_text": text
    }
