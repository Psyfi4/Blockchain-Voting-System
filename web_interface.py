#!/usr/bin/env python3
# web_interface.py
"""
Web interface for AadhaarFaceSystem (Option A).
- Standardizes DB columns to: registered_face_path, aadhaar_photo_path
- Endpoints:
    /                 -> home page (template)
    /register         -> registration page (template)
    /recognize_live   -> live recognition page (template)
    /detect_preview   -> POST {image: dataURL} -> {faces: [...]}
    /register_api     -> POST {image: dataURL, aadhaar_number, name, ... , aadhaar_photo (optional dataURL)}
    /recognize_frame  -> POST {image: dataURL} -> recognition results (enriched)
    /recognize_stream -> POST {image: dataURL} -> JPEG overlay (optional preview)
    /diag             -> POST {image: dataURL} -> returns shape/dtype for debug
"""

import os
import io
import re
import time
import base64
import traceback
from functools import wraps
from typing import Optional

from flask import Flask, render_template, request, jsonify, Response
from PIL import Image, ExifTags
import numpy as np
import cv2

# Import your AadhaarFaceSystem. Ensure it's on PYTHONPATH and loads correctly.
from aadhaar_system import AadhaarFaceSystem

# ---- Flask app ----
app = Flask(__name__, template_folder="templates", static_folder="static")
os.makedirs("debug", exist_ok=True)
os.makedirs(os.path.join("static", "registered_faces"), exist_ok=True)
os.makedirs(os.path.join("static", "aadhaar_photos"), exist_ok=True)

# single global instance
system = AadhaarFaceSystem(debug=True)

# Regex to match data URLs
_DATAURL_RE = re.compile(r'data:(image/[^;]+);base64,(.*)$', re.I)

# -----------------------
# Utilities
# -----------------------
def json_api(fn):
    """Decorator to ensure JSON error responses and traceback logging."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"success": False, "message": f"Server error: {str(e)}", "trace": traceback.format_exc()}), 500
    return wrapper

def decode_base64_image(data_url: str) -> Optional[np.ndarray]:
    """
    Decode a data URL or plain base64 string to an RGB numpy array (HxWx3).
    Returns numpy array in RGB order.
    """
    if not isinstance(data_url, str):
        return None
    m = _DATAURL_RE.match(data_url.strip())
    if m:
        b64 = m.group(2)
    else:
        # possibly a bare base64 or "data:image/...;base64,..." variant - handle generically
        parts = data_url.split(",", 1)
        if len(parts) == 2 and "base64" in parts[0]:
            b64 = parts[1]
        else:
            # assume whole string is base64
            b64 = data_url.strip()
    try:
        b = base64.b64decode(b64)
        pil = Image.open(io.BytesIO(b))
        # fix orientation if EXIF exists
        try:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            exif = pil._getexif()
            if exif is not None:
                orient = exif.get(orientation)
                if orient == 3:
                    pil = pil.rotate(180, expand=True)
                elif orient == 6:
                    pil = pil.rotate(270, expand=True)
                elif orient == 8:
                    pil = pil.rotate(90, expand=True)
        except Exception:
            pass
        pil = pil.convert("RGB")
        arr = np.asarray(pil)  # RGB
        return arr
    except Exception as e:
        print("[decode_base64_image] failed:", e)
        return None

def encode_image_to_dataurl_from_bgr(bgr_arr: np.ndarray, fmt="JPEG") -> str:
    """
    Converts a BGR numpy array (HxWx3) to a data URL (base64).
    """
    # convert BGR->RGB for PIL
    rgb = bgr_arr[:, :, ::-1].astype("uint8")
    img = Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{b64}"

def save_debug_image_bgr(bgr_arr: np.ndarray, tag="debug"):
    ts = int(time.time())
    path = os.path.join("debug", f"{tag}_{ts}.jpg")
    try:
        cv2.imwrite(path, bgr_arr)
    except Exception as e:
        print("[save_debug_image_bgr] failed:", e)
    return path

def ensure_db_has_columns():
    """Ensure persons table has the standardized columns for Option A."""
    try:
        cur = system.conn.cursor()
        cur.execute("PRAGMA table_info(persons);")
        cols = [r[1] for r in cur.fetchall()]
        if "registered_face_path" not in cols:
            try:
                cur.execute("ALTER TABLE persons ADD COLUMN registered_face_path TEXT;")
                system.conn.commit()
                print("[MIGRATE] Added registered_face_path")
            except Exception as me:
                print("[MIGRATE] Failed to add registered_face_path:", me)
        if "aadhaar_photo_path" not in cols:
            try:
                cur.execute("ALTER TABLE persons ADD COLUMN aadhaar_photo_path TEXT;")
                system.conn.commit()
                print("[MIGRATE] Added aadhaar_photo_path")
            except Exception as me:
                print("[MIGRATE] Failed to add aadhaar_photo_path:", me)
        cur.close()
    except Exception as e:
        print("[MIGRATE] ensure columns failed:", e)

# Run migration at startup
ensure_db_has_columns()

# -----------------------
# Routes: pages
# -----------------------
@app.route("/")
def home():
    # simple landing, ensure you have templates/home.html
    return render_template("home.html")

@app.route("/register")
def register_page():
    return render_template("register.html")

@app.route("/recognize_live")
def recognize_page():
    return render_template("recognize_live.html")

# -----------------------
# API endpoints
# -----------------------
@app.route("/detect_preview", methods=["POST"])
@json_api
def detect_preview():
    """
    POST {image: dataURL}
    returns: { faces: [ { bbox:[x1,y1,x2,y2], w, h }, ... ] }
    """
    data = request.get_json(force=True, silent=True)
    if not data or "image" not in data:
        return jsonify({"faces": []}), 400
    rgb = decode_base64_image(data["image"])
    if rgb is None:
        return jsonify({"faces": []}), 400
    # convert RGB->BGR for insightface/opencv
    bgr = rgb[:, :, ::-1].copy()
    try:
        faces = system.app.get(bgr)
    except Exception as e:
        print("[detect_preview] insightface get failed:", e)
        return jsonify({"faces": [], "error": str(e)}), 500
    out = []
    for f in faces:
        try:
            x1, y1, x2, y2 = map(int, f.bbox)
            out.append({"bbox": [x1, y1, x2, y2], "w": int(x2 - x1), "h": int(y2 - y1)})
        except Exception:
            continue
    # optionally save a debug image
    try:
        save_debug_image_bgr(bgr, "detect_preview_in")
    except Exception:
        pass
    return jsonify({"faces": out}), 200

@app.route("/register_api", methods=["POST"])
@json_api
def register_api():
    """
    Register a person.
    Expects JSON:
       {
         image: dataURL (face crop captured from the camera preview),
         aadhaar_number, name, date_of_birth, gender, address,
         aadhaar_photo: optional dataURL of Aadhaar card uploaded by user
       }
    Flow:
      - decode image, save to temp file
      - call system.register_person(temp_path, person_data)
      - detect face bbox to crop and save registered_face_path
      - save Aadhaar card photo to aadhaar_photos/<aadhaar>.jpg and update DB column aadhaar_photo_path
      - update persons row with registered_face_path and aadhaar_photo_path
    """
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"success": False, "message": "No JSON body received"}), 400

    image_b64 = data.get("image")
    aadhaar_number = data.get("aadhaar_number")
    name = data.get("name")
    dob = data.get("date_of_birth")
    gender = data.get("gender")
    address = data.get("address")
    aadhaar_photo_b64 = data.get("aadhaar_photo")  # optional

    if not image_b64 or not aadhaar_number or not name:
        return jsonify({"success": False, "message": "Missing required fields: image/aadhaar_number/name"}), 400

    # decode captured face image (RGB)
    rgb = decode_base64_image(image_b64)
    if rgb is None:
        return jsonify({"success": False, "message": "Could not decode captured image"}), 400

    # Save temp file for register_person (BGR)
    ts = int(time.time())
    tmp_path = os.path.join("debug", f"register_capture_{aadhaar_number}_{ts}.jpg")
    try:
        cv2.imwrite(tmp_path, rgb[:, :, ::-1])  # save as BGR
    except Exception as e:
        return jsonify({"success": False, "message": "Failed to save temp image: " + str(e)}), 500

    # call register_person to store embedding/etc.
    try:
        ok, msg = system.register_person(tmp_path, {
            "aadhaar_number": aadhaar_number,
            "name": name,
            "date_of_birth": dob,
            "gender": gender,
            "address": address,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "message": "register_person raised exception: " + str(e)}), 500

    if not ok:
        # return error message from register_person (e.g., No face detected)
        return jsonify({"success": False, "message": msg}), 400

    # At this point embedding saved. Now detect face to crop and save registered_face_path:
    try:
        # Try to detect face again on the saved tmp (BGR)
        img_bgr = cv2.imread(tmp_path)
        faces = system.app.get(img_bgr)
        reg_face_path = None
        if faces and len(faces) >= 1:
            f = faces[0]
            try:
                x1, y1, x2, y2 = map(int, f.bbox)
                h, w = img_bgr.shape[:2]
                # clamp
                x1 = max(0, min(w - 1, x1))
                y1 = max(0, min(h - 1, y1))
                x2 = max(0, min(w, x2))
                y2 = max(0, min(h, y2))
                crop = img_bgr[y1:y2, x1:x2]
                os.makedirs(os.path.join("static", "registered_faces"), exist_ok=True)
                reg_face_path = os.path.join("static", "registered_faces", f"{aadhaar_number}.jpg")
                cv2.imwrite(reg_face_path, crop)
            except Exception as e:
                print("[register_api] failed to crop/save registered face:", e)
                reg_face_path = None
        else:
            print("[register_api] face detection after register_person returned no faces.")
    except Exception as e:
        print("[register_api] detection error:", e)
        reg_face_path = None

    # If Aadhaar card photo was provided, save it
    aadhaar_photo_path = None
    if aadhaar_photo_b64:
        try:
            aadhaar_rgb = decode_base64_image(aadhaar_photo_b64)
            if aadhaar_rgb is not None:
                os.makedirs(os.path.join("static", "aadhaar_photos"), exist_ok=True)
                aadhaar_photo_path = os.path.join("static", "aadhaar_photos", f"{aadhaar_number}.jpg")
                cv2.imwrite(aadhaar_photo_path, aadhaar_rgb[:, :, ::-1])
        except Exception as e:
            print("[register_api] failed to save aadhaar photo:", e)
            aadhaar_photo_path = None

    # Update DB with the two standardized columns
    try:
        cur = system.conn.cursor()
        if reg_face_path:
            cur.execute("UPDATE persons SET registered_face_path = ? WHERE aadhaar_number = ?", (reg_face_path, aadhaar_number))
        if aadhaar_photo_path:
            cur.execute("UPDATE persons SET aadhaar_photo_path = ? WHERE aadhaar_number = ?", (aadhaar_photo_path, aadhaar_number))
        system.conn.commit()
        cur.close()
    except Exception as e:
        print("[register_api] DB update failed:", e)

    return jsonify({"success": True, "message": msg, "registered_face_path": reg_face_path, "aadhaar_photo_path": aadhaar_photo_path}), 200

@app.route("/recognize_frame", methods=["POST"])
@json_api
def recognize_frame():
    """
    POST {image: dataURL}
    Returns recognition results enriched with registered_face and aadhaar_photo data URLs (if available).
    """
    data = request.get_json(force=True, silent=True)
    if not data or "image" not in data:
        return jsonify({"error": "missing image"}), 400

    rgb = decode_base64_image(data["image"])
    if rgb is None:
        return jsonify({"error": "could not decode"}), 400

    # Call recognizer (system.recognize_face expects numpy arrays RGB or BGR depending on your implementation).
    # We try RGB first (most of our helpers return RGB).
    try:
        results = system.recognize_face(rgb)
    except Exception:
        try:
            results = system.recognize_face(rgb[:, :, ::-1])
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": "recognition failed: " + str(e)}), 500

    enriched = []
    for r in results:
        item = dict(r)
        aad = item.get("aadhaar_number")
        # Attach registered face & aadhaar photo as data URLs if available in DB
        reg_dataurl = None
        aadhaar_dataurl = None
        if aad:
            try:
                details = system.get_person_details(aad)
                if details:
                    # Prefer standardized columns
                    reg_path = details.get("registered_face_path") or details.get("registered_face") or details.get("registered_face_path")
                    aad_path = details.get("aadhaar_photo_path") or details.get("aadhaar_photo") or details.get("photo_path")
                    if reg_path and os.path.exists(reg_path):
                        # load BGR and return dataURL
                        try:
                            bgr = cv2.imread(reg_path)
                            if bgr is not None:
                                reg_dataurl = encode_image_to_dataurl_from_bgr(bgr)
                        except Exception:
                            pass
                    if aad_path and os.path.exists(aad_path):
                        try:
                            bgr2 = cv2.imread(aad_path)
                            if bgr2 is not None:
                                aadhaar_dataurl = encode_image_to_dataurl_from_bgr(bgr2)
                        except Exception:
                            pass
            except Exception as e:
                print("[recognize_frame] enrichment error:", e)
        item["registered_face"] = reg_dataurl
        item["aadhaar_photo"] = aadhaar_dataurl
        enriched.append(item)

    return jsonify({"results": enriched}), 200

@app.route("/recognize_stream", methods=["POST"])
@json_api
def recognize_stream():
    """
    POST {image: dataURL} -> returns an image/jpeg that has bounding boxes/dims overlaid
    Useful for preview; frontend can stream this back into an <img>.
    """
    data = request.get_json(force=True, silent=True)
    if not data or "image" not in data:
        return Response("missing image", status=400)

    rgb = decode_base64_image(data["image"])
    if rgb is None:
        return Response("invalid image", status=400)

    bgr = rgb[:, :, ::-1].copy()
    try:
        results = system.recognize_face(rgb)
    except Exception:
        try:
            results = system.recognize_face(bgr)
        except Exception as e:
            print("[recognize_stream] recognition failed:", e)
            results = []

    display = bgr.copy()
    if results and len(results) > 0:
        face = results[0]
        top, right, bottom, left = face.get("location", (0, 0, 0, 0))
        h, w = display.shape[:2]
        top = max(0, min(h - 1, int(top)))
        left = max(0, min(w - 1, int(left)))
        bottom = max(0, min(h - 1, int(bottom)))
        right = max(0, min(w - 1, int(right)))
        cv2.rectangle(display, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(display, f"{right-left}x{bottom-top}", (left, max(10, top-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        label = face.get("name") or face.get("aadhaar_number") or ""
        if label:
            cv2.putText(display, label, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    else:
        cv2.putText(display, "No face detected", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

    save_debug_image_bgr(display, tag="stream_out")
    ok, jpg = cv2.imencode(".jpg", display)
    if not ok:
        return Response("encoding failed", status=500)
    return Response(jpg.tobytes(), mimetype="image/jpeg")

@app.route("/diag", methods=["POST"])
def diag():
    """
    Debug endpoint. POST {image: dataURL} -> returns shape/dtype the server sees.
    """
    try:
        data = request.get_json(force=True, silent=True)
        if not data or "image" not in data:
            return jsonify({"error": "missing image"}), 400
        arr = decode_base64_image(data["image"])
        if arr is None:
            return jsonify({"error": "could not decode"}), 400
        return jsonify({"ok": True, "shape": list(arr.shape), "dtype": str(arr.dtype)}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

# -----------------------
# Run server
# -----------------------
if __name__ == "__main__":
    host = os.environ.get("FLASK_RUN_HOST", "0.0.0.0")
    port = int(os.environ.get("FLASK_RUN_PORT", 5000))
    print(f"Starting Flask on http://{host}:{port}")
    app.run(host=host, port=port, debug=True)
