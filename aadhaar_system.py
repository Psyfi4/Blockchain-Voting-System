import os
import re
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import easyocr
import face_recognition
import numpy as np

DB_PATH = "db/aadhaar.db"
_AADHAAR_RE = re.compile(r"\b\d{12}\b")


class AadhaarFaceSystem:
    """Unified system class used by Streamlit, Flask, and CLI entry points."""

    def __init__(self, db_path: str = DB_PATH, debug: bool = False):
        self.debug = debug
        self.db_path = db_path

        os.makedirs("db", exist_ok=True)
        os.makedirs("data/faces", exist_ok=True)
        os.makedirs("data/aadhaar", exist_ok=True)

        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._create_tables()

        # OCR init is expensive; keep one reader per process.
        self.reader = easyocr.Reader(["en"], gpu=False)

        self.known_encodings: List[np.ndarray] = []
        self.known_meta: List[Dict[str, Optional[str]]] = []
        self._known_matrix: np.ndarray = np.empty((0, 128), dtype=np.float64)
        self._load_known()

    # ---------- DATABASE ---------- #
    def _create_tables(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS persons (
                aadhaar_number TEXT PRIMARY KEY,
                name TEXT,
                date_of_birth TEXT,
                gender TEXT,
                address TEXT,
                face_encoding BLOB,
                face_path TEXT,
                aadhaar_path TEXT,
                registered_face_path TEXT,
                aadhaar_photo_path TEXT,
                created_at TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS recognition_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                aadhaar_number TEXT,
                recognition_time TEXT,
                confidence REAL
            )
            """
        )
        self.conn.commit()

        # Backward compatible migration if legacy table had `aadhaar` column.
        cur.execute("PRAGMA table_info(persons)")
        cols = [r[1] for r in cur.fetchall()]
        if "aadhaar" in cols and "aadhaar_number" not in cols:
            cur.execute("ALTER TABLE persons ADD COLUMN aadhaar_number TEXT")
            cur.execute("UPDATE persons SET aadhaar_number = aadhaar WHERE aadhaar_number IS NULL")
            self.conn.commit()
        for col_name, col_type in (
            ("registered_face_path", "TEXT"),
            ("aadhaar_photo_path", "TEXT"),
            ("date_of_birth", "TEXT"),
            ("gender", "TEXT"),
            ("address", "TEXT"),
            ("aadhaar_path", "TEXT"),
            ("face_path", "TEXT"),
            ("created_at", "TEXT"),
        ):
            if col_name not in cols:
                cur.execute(f"ALTER TABLE persons ADD COLUMN {col_name} {col_type}")
        self.conn.commit()

    def _load_known(self) -> None:
        self.known_encodings.clear()
        self.known_meta.clear()

        cur = self.conn.cursor()
        cur.execute("SELECT aadhaar_number, name, face_encoding FROM persons WHERE face_encoding IS NOT NULL")
        for aadhaar_number, name, blob in cur.fetchall():
            if not blob:
                continue
            enc = np.frombuffer(blob, dtype=np.float64)
            if enc.size != 128:
                continue
            self.known_encodings.append(enc)
            self.known_meta.append({"aadhaar_number": aadhaar_number, "name": name})

        if self.known_encodings:
            self._known_matrix = np.vstack(self.known_encodings)
        else:
            self._known_matrix = np.empty((0, 128), dtype=np.float64)

    # ---------- OCR ---------- #
    def extract_aadhaar(self, image: np.ndarray) -> Optional[str]:
        results = self.reader.readtext(image)
        text = " ".join([r[1] for r in results])
        match = _AADHAAR_RE.search(text)
        return match.group(0) if match else None

    # ---------- INTERNAL HELPERS ---------- #
    @staticmethod
    def _ensure_bgr(image: Any) -> Optional[np.ndarray]:
        if image is None:
            return None
        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[2] != 3:
            return None
        return arr

    @staticmethod
    def _first_face_encoding(bgr_img: np.ndarray) -> Optional[np.ndarray]:
        rgb = bgr_img[:, :, ::-1]
        encs = face_recognition.face_encodings(rgb)
        return encs[0] if encs else None

    def _upsert_person(self, payload: Dict[str, Any]) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO persons (
                aadhaar_number, name, date_of_birth, gender, address,
                face_encoding, face_path, aadhaar_path,
                registered_face_path, aadhaar_photo_path, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(aadhaar_number) DO UPDATE SET
                name=excluded.name,
                date_of_birth=excluded.date_of_birth,
                gender=excluded.gender,
                address=excluded.address,
                face_encoding=excluded.face_encoding,
                face_path=excluded.face_path,
                aadhaar_path=excluded.aadhaar_path,
                registered_face_path=excluded.registered_face_path,
                aadhaar_photo_path=excluded.aadhaar_photo_path,
                created_at=excluded.created_at
            """,
            (
                payload.get("aadhaar_number"),
                payload.get("name"),
                payload.get("date_of_birth"),
                payload.get("gender"),
                payload.get("address"),
                payload.get("face_encoding"),
                payload.get("face_path"),
                payload.get("aadhaar_path"),
                payload.get("registered_face_path"),
                payload.get("aadhaar_photo_path"),
                payload.get("created_at"),
            ),
        )
        self.conn.commit()

    # ---------- REGISTRATION ---------- #
    def register(self, face_img: np.ndarray, aadhaar_img: np.ndarray) -> Tuple[bool, str]:
        face_bgr = self._ensure_bgr(face_img)
        aadhaar_bgr = self._ensure_bgr(aadhaar_img)
        if face_bgr is None or aadhaar_bgr is None:
            return False, "Invalid image format"

        enc = self._first_face_encoding(face_bgr)
        if enc is None:
            return False, "No face detected"

        aadhaar_number = self.extract_aadhaar(aadhaar_bgr)
        if not aadhaar_number:
            return False, "Aadhaar not detected"

        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        face_path = f"data/faces/{aadhaar_number}_{ts}.jpg"
        aadhaar_path = f"data/aadhaar/{aadhaar_number}_{ts}.jpg"

        cv2.imwrite(face_path, face_bgr)
        cv2.imwrite(aadhaar_path, aadhaar_bgr)

        self._upsert_person(
            {
                "aadhaar_number": aadhaar_number,
                "name": "UNKNOWN",
                "face_encoding": enc.tobytes(),
                "face_path": face_path,
                "aadhaar_path": aadhaar_path,
                "created_at": datetime.now().isoformat(),
            }
        )

        self._load_known()
        return True, f"Registered Aadhaar {aadhaar_number}"

    def register_person(self, image_path: str, person_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Compatibility API used by Flask route /register_api."""
        bgr = cv2.imread(image_path)
        if bgr is None:
            return False, "Unable to read registration image"

        enc = self._first_face_encoding(bgr)
        if enc is None:
            return False, "No face detected"

        aadhaar_number = str(person_data.get("aadhaar_number") or "").strip()
        if not aadhaar_number:
            return False, "Missing aadhaar_number"

        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        face_path = f"data/faces/{aadhaar_number}_{ts}.jpg"
        cv2.imwrite(face_path, bgr)

        self._upsert_person(
            {
                "aadhaar_number": aadhaar_number,
                "name": person_data.get("name") or "UNKNOWN",
                "date_of_birth": person_data.get("date_of_birth"),
                "gender": person_data.get("gender"),
                "address": person_data.get("address"),
                "face_encoding": enc.tobytes(),
                "face_path": face_path,
                "registered_face_path": face_path,
                "created_at": datetime.now().isoformat(),
            }
        )

        self._load_known()
        return True, "Person registered successfully"

    # ---------- RECOGNITION ---------- #
    def recognize(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, str]]:
        bgr = self._ensure_bgr(frame)
        if bgr is None:
            return []

        rgb = bgr[:, :, ::-1]
        locs = face_recognition.face_locations(rgb)
        encs = face_recognition.face_encodings(rgb, locs)

        if self._known_matrix.size == 0:
            return [(top, right, bottom, left, "Unknown") for (top, right, bottom, left) in locs]

        results: List[Tuple[int, int, int, int, str]] = []
        for (top, right, bottom, left), enc in zip(locs, encs):
            dists = np.linalg.norm(self._known_matrix - enc, axis=1)
            idx = int(np.argmin(dists))

            if dists[idx] < 0.55:
                meta = self.known_meta[idx]
                label = str(meta.get("aadhaar_number") or meta.get("name") or "Unknown")
                self._log_recognition(str(meta.get("aadhaar_number") or ""), float(1.0 - min(1.0, dists[idx])))
            else:
                label = "Unknown"

            results.append((top, right, bottom, left, label))

        return results

    def recognize_face(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Compatibility API used by Flask route /recognize_frame and /recognize_stream."""
        tuple_results = self.recognize(frame)
        out: List[Dict[str, Any]] = []
        for top, right, bottom, left, label in tuple_results:
            matched = label != "Unknown"
            out.append(
                {
                    "location": (top, right, bottom, left),
                    "name": label if matched else "Unknown",
                    "aadhaar_number": label if matched else None,
                    "confidence": 1.0 if matched else 0.0,
                    "matched": matched,
                }
            )
        return out

    def get_person_details(self, aadhaar_number: str) -> Optional[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT aadhaar_number, name, date_of_birth, gender, address,
                   face_path, aadhaar_path, registered_face_path, aadhaar_photo_path, created_at
            FROM persons WHERE aadhaar_number = ?
            """,
            (aadhaar_number,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "aadhaar_number": row[0],
            "name": row[1],
            "date_of_birth": row[2],
            "gender": row[3],
            "address": row[4],
            "face_path": row[5],
            "aadhaar_path": row[6],
            "registered_face_path": row[7] or row[5],
            "aadhaar_photo_path": row[8] or row[6],
            "created_at": row[9],
        }

    def _log_recognition(self, aadhaar_number: str, confidence: float) -> None:
        if not aadhaar_number:
            return
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO recognition_logs (aadhaar_number, recognition_time, confidence) VALUES (?, ?, ?)",
            (aadhaar_number, datetime.utcnow().isoformat(), confidence),
        )
        self.conn.commit()

    # ---------- CLI HELPERS ---------- #
    def register_live(self) -> None:
        print("Live registration is not configured in this build. Use app.py or web_interface.py.")

    def recognize_live(self) -> None:
        print("Live recognition is not configured in this build. Use app.py or web_interface.py.")
