#!/usr/bin/env python3
"""
AadhaarFaceSystem – FINAL Pylance-Clean Version
- InsightFace only
- NumPy 1.26 compatible
- Headless / Codespaces safe
- ZERO invalid type expressions
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
import os
import sqlite3
import time

# ---------------- DEPENDENCY CHECKS ---------------- #

_missing = []

try:
    import numpy as np
except Exception:
    np = None
    _missing.append("numpy==1.26.4")

try:
    from PIL import Image
except Exception:
    Image = None
    _missing.append("Pillow")

try:
    from insightface.app import FaceAnalysis
except Exception:
    FaceAnalysis = None
    _missing.append("insightface + onnxruntime")

if _missing:
    raise RuntimeError(
        "Missing dependencies:\n"
        + ", ".join(_missing)
        + "\n\nInstall with:\n"
        "pip install numpy==1.26.4 pillow insightface onnxruntime"
    )

# ---------------- SYSTEM CLASS ---------------- #

class AadhaarFaceSystem:
    def __init__(self, db_path: str = "aadhaar_face_db.db"):
        self.db_path = db_path
        os.makedirs("static/registered_faces", exist_ok=True)

        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._create_tables()

        self.face_app = FaceAnalysis(name="buffalo_l")
        self.face_app.prepare(ctx_id=-1, det_size=(640, 640))

        self.known_embeddings: List[Any] = []
        self.known_details: List[Dict[str, str]] = []

        self._load_cache()

    # ---------------- DATABASE ---------------- #

    def _create_tables(self) -> None:
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                aadhaar_number TEXT UNIQUE,
                name TEXT,
                date_of_birth TEXT,
                gender TEXT,
                address TEXT,
                embedding BLOB,
                registered_face_path TEXT,
                registration_date TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS recognition_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                aadhaar_number TEXT,
                time TEXT,
                confidence REAL
            )
        """)
        self.conn.commit()
        cur.close()

    def _load_cache(self) -> None:
        cur = self.conn.cursor()
        cur.execute("SELECT aadhaar_number, name, embedding FROM persons")
        rows = cur.fetchall()

        self.known_embeddings.clear()
        self.known_details.clear()

        for aadhaar, name, blob in rows:
            if blob:
                emb = np.frombuffer(blob, dtype=np.float32)
                if emb.size == 512:
                    self.known_embeddings.append(emb)
                    self.known_details.append({
                        "aadhaar_number": aadhaar,
                        "name": name
                    })
        cur.close()

    # ---------------- REGISTRATION ---------------- #

    def register_person_from_pil(
        self,
        face_image: Any,
        person_data: Dict[str, Optional[str]]
    ) -> tuple[bool, str]:

        if not hasattr(face_image, "convert"):
            return False, "Invalid image object (expected PIL Image)"

        frame = np.array(face_image.convert("RGB"))
        faces = self.face_app.get(frame)

        if not faces:
            return False, "No face detected"
        if len(faces) > 1:
            return False, "Multiple faces detected"

        face = faces[0]
        emb = face.embedding.astype(np.float32)
        if emb.size != 512:
            return False, "Invalid embedding size"

        ts = int(time.time())
        x1, y1, x2, y2 = map(int, face.bbox)
        crop = frame[y1:y2, x1:x2]

        save_path = f"static/registered_faces/{person_data['aadhaar_number']}_{ts}.jpg"
        Image.fromarray(crop).save(save_path)

        cur = self.conn.cursor()
        cur.execute("""
            INSERT OR REPLACE INTO persons
            (aadhaar_number, name, date_of_birth, gender, address,
             embedding, registered_face_path, registration_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            person_data.get("aadhaar_number"),
            person_data.get("name"),
            person_data.get("date_of_birth"),
            person_data.get("gender"),
            person_data.get("address"),
            emb.tobytes(),
            save_path,
            datetime.utcnow().isoformat()
        ))
        self.conn.commit()
        cur.close()

        self._load_cache()
        return True, "Person registered successfully"

    # ---------------- RECOGNITION ---------------- #

    def recognize_face_from_pil(self, face_image: Any) -> List[Dict[str, Any]]:
        if not hasattr(face_image, "convert"):
            return []

        frame = np.array(face_image.convert("RGB"))
        faces = self.face_app.get(frame)

        results: List[Dict[str, Any]] = []

        for f in faces:
            emb = f.embedding.astype(np.float32)
            emb /= (np.linalg.norm(emb) + 1e-10)

            best_idx = None
            best_conf = 0.0

            for i, known in enumerate(self.known_embeddings):
                known_n = known / (np.linalg.norm(known) + 1e-10)
                sim = float(np.dot(emb, known_n))
                conf = (sim + 1.0) / 2.0
                if conf > best_conf:
                    best_conf = conf
                    best_idx = i

            matched = best_conf >= 0.44
            aadhaar = name = None

            if matched and best_idx is not None:
                detail = self.known_details[best_idx]
                aadhaar = detail["aadhaar_number"]
                name = detail["name"]
                self._log_recognition(aadhaar, best_conf)

            results.append({
                "name": name,
                "aadhaar_number": aadhaar,
                "confidence": best_conf,
                "matched": matched
            })

        return results

    # ---------------- UTILITIES ---------------- #

    def _log_recognition(self, aadhaar: str, confidence: float) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO recognition_logs (aadhaar_number, time, confidence) VALUES (?, ?, ?)",
            (aadhaar, datetime.utcnow().isoformat(), confidence)
        )
        self.conn.commit()
        cur.close()

    def close(self) -> None:
        self.conn.close()
