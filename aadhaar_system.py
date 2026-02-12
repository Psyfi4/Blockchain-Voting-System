#!/usr/bin/env python3
"""
AadhaarFaceSystem – Stable Recognition Version
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
import os
import sqlite3
import time
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis


class AadhaarFaceSystem:

    def __init__(self, db_path: str = "aadhaar_face_db.db"):
        self.db_path = db_path

        os.makedirs("static/registered_faces", exist_ok=True)

        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._create_tables()

        # CPU mode (Codespaces safe)
        self.face_app = FaceAnalysis(name="buffalo_l")
        self.face_app.prepare(ctx_id=-1, det_size=(640, 640))

        self.known_embeddings: List[np.ndarray] = []
        self.known_details: List[Dict[str, str]] = []

        self._load_cache()

    # ---------------- DATABASE ---------------- #

    def _create_tables(self):
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

    def _load_cache(self):
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

        print("Loaded embeddings:", len(self.known_embeddings))
        cur.close()

    # ---------------- REGISTRATION ---------------- #

    def register_person_from_pil(
        self,
        face_image: Any,
        person_data: Dict[str, Optional[str]]
    ):

        if not hasattr(face_image, "convert"):
            return False, "Invalid image"

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

        # Save crop
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

    def recognize_face_from_pil(self, face_image):

        if not hasattr(face_image, "convert"):
            return []

        frame = np.array(face_image.convert("RGB"))
        faces = self.face_app.get(frame)

        results = []

        for f in faces:

            emb = f.embedding.astype(np.float32)

            if emb.size != 512:
                continue

            best_idx = None
            best_sim = -1.0

            for i, known in enumerate(self.known_embeddings):

                sim = float(
                    np.dot(emb, known) /
                    ((np.linalg.norm(emb) * np.linalg.norm(known)) + 1e-10)
                )

                if sim > best_sim:
                    best_sim = sim
                    best_idx = i

            confidence = (best_sim + 1.0) / 2.0
            matched = best_sim > 0.40  # 🔥 tuned for webcam

            name = None
            aadhaar = None

            if matched and best_idx is not None:
                detail = self.known_details[best_idx]
                name = detail["name"]
                aadhaar = detail["aadhaar_number"]
                self._log_recognition(aadhaar, confidence)

            results.append({
                "name": name,
                "aadhaar_number": aadhaar,
                "confidence": confidence,
                "matched": matched
            })

        return results

    # ---------------- LOGGING ---------------- #

    def _log_recognition(self, aadhaar, confidence):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO recognition_logs (aadhaar_number, time, confidence) VALUES (?, ?, ?)",
            (aadhaar, datetime.utcnow().isoformat(), confidence)
        )
        self.conn.commit()
        cur.close()

    def close(self):
        self.conn.close()
