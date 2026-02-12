import os
import cv2
import numpy as np
import sqlite3
import face_recognition
import easyocr
from datetime import datetime

DB_PATH = "db/aadhaar.db"

class AadhaarFaceSystem:
    def __init__(self):
        os.makedirs("db", exist_ok=True)
        os.makedirs("data/faces", exist_ok=True)
        os.makedirs("data/aadhaar", exist_ok=True)

        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._create_tables()

        self.reader = easyocr.Reader(['en'], gpu=False)
        self.known_encodings = []
        self.known_meta = []
        self._load_known()

    def _create_tables(self):
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                aadhaar TEXT PRIMARY KEY,
                name TEXT,
                face_encoding BLOB,
                face_path TEXT,
                aadhaar_path TEXT,
                created_at TEXT
            )
        """)
        self.conn.commit()

    def _load_known(self):
        self.known_encodings.clear()
        self.known_meta.clear()
        cur = self.conn.cursor()
        cur.execute("SELECT aadhaar, name, face_encoding FROM persons")
        for aadhaar, name, blob in cur.fetchall():
            enc = np.frombuffer(blob, dtype=np.float64)
            self.known_encodings.append(enc)
            self.known_meta.append({"aadhaar": aadhaar, "name": name})

    # ---------- OCR ---------- #
    def extract_aadhaar(self, image):
        results = self.reader.readtext(image)
        text = " ".join([r[1] for r in results])
        aadhaar = None
        for part in text.split():
            if part.isdigit() and len(part) == 12:
                aadhaar = part
        return aadhaar

    # ---------- REGISTRATION ---------- #
    def register(self, face_img, aadhaar_img):
        rgb = face_img[:, :, ::-1]
        encs = face_recognition.face_encodings(rgb)
        if not encs:
            return False, "No face detected"

        enc = encs[0]

        aadhaar = self.extract_aadhaar(aadhaar_img)
        if not aadhaar:
            return False, "Aadhaar not detected"

        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        face_path = f"data/faces/{aadhaar}_{ts}.jpg"
        aadhaar_path = f"data/aadhaar/{aadhaar}_{ts}.jpg"

        cv2.imwrite(face_path, face_img)
        cv2.imwrite(aadhaar_path, aadhaar_img)

        cur = self.conn.cursor()
        cur.execute("""
            INSERT OR REPLACE INTO persons
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            aadhaar,
            "UNKNOWN",
            enc.tobytes(),
            face_path,
            aadhaar_path,
            datetime.now().isoformat()
        ))
        self.conn.commit()
        self._load_known()

        return True, f"Registered Aadhaar {aadhaar}"

    # ---------- RECOGNITION ---------- #
    def recognize(self, frame):
        rgb = frame[:, :, ::-1]
        locs = face_recognition.face_locations(rgb)
        encs = face_recognition.face_encodings(rgb, locs)

        results = []
        for (top, right, bottom, left), enc in zip(locs, encs):
            if not self.known_encodings:
                continue

            dists = face_recognition.face_distance(self.known_encodings, enc)
            idx = np.argmin(dists)

            if dists[idx] < 0.55:
                meta = self.known_meta[idx]
                label = f"{meta['aadhaar']}"
            else:
                label = "Unknown"

            results.append((top, right, bottom, left, label))

        return results
