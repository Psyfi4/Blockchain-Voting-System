import os
import sqlite3
import numpy as np
from datetime import datetime
from PIL import Image
from insightface.app import FaceAnalysis

DB = "aadhaar_face_db.db"


class AadhaarFaceSystem:

    def __init__(self):
        os.makedirs("static/registered_faces", exist_ok=True)
        os.makedirs("static/aadhaar_photos", exist_ok=True)

        self.conn = sqlite3.connect(DB, check_same_thread=False)
        self._create_tables()

        # CPU mode → works in Codespaces
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=-1)

        self.known_embeddings = []
        self.known_ids = []

        self._load_known()

    def _create_tables(self):
        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS persons(
            aadhaar TEXT PRIMARY KEY,
            name TEXT,
            embedding BLOB,
            face_path TEXT,
            aadhaar_path TEXT
        )
        """)
        self.conn.commit()

    def _load_known(self):
        self.known_embeddings.clear()
        self.known_ids.clear()

        cur = self.conn.cursor()
        for aadhaar, blob in cur.execute("SELECT aadhaar, embedding FROM persons"):
            emb = np.frombuffer(blob, dtype=np.float32)
            self.known_embeddings.append(emb)
            self.known_ids.append(aadhaar)

        print("Loaded embeddings:", len(self.known_embeddings))

    # ---------------- REGISTER ---------------- #

    def register(self, face_img: Image.Image, aadhaar_img: Image.Image, aadhaar, name):

        frame = np.array(face_img.convert("RGB"))
        faces = self.app.get(frame)

        if not faces:
            return False, "No face detected"

        emb = faces[0].embedding.astype(np.float32)

        # SAVE FACE
        face_path = f"static/registered_faces/{aadhaar}.jpg"
        Image.fromarray(frame).save(face_path)

        # SAVE AADHAAR IMAGE
        aadhaar_path = f"static/aadhaar_photos/{aadhaar}.jpg"
        aadhaar_img.save(aadhaar_path)

        cur = self.conn.cursor()
        cur.execute("""
        INSERT OR REPLACE INTO persons VALUES (?,?,?,?,?)
        """, (aadhaar, name, emb.tobytes(), face_path, aadhaar_path))
        self.conn.commit()

        self._load_known()
        return True, "Registered Successfully"

    # ---------------- RECOGNIZE ---------------- #

    def recognize(self, frame: Image.Image):

        img = np.array(frame.convert("RGB"))
        faces = self.app.get(img)

        results = []

        for f in faces:
            emb = f.embedding.astype(np.float32)

            # Normalize (THIS FIXES YOUR ISSUE)
            emb = emb / (np.linalg.norm(emb) + 1e-10)

            best_sim = -1
            best_id = None

            for known, aid in zip(self.known_embeddings, self.known_ids):
                known_n = known / (np.linalg.norm(known) + 1e-10)
                sim = float(np.dot(emb, known_n))

                if sim > best_sim:
                    best_sim = sim
                    best_id = aid

            print("SIMILARITY:", best_sim)

            if best_sim > 0.38:
                results.append({"aadhaar": best_id, "confidence": best_sim})
            else:
                results.append({"aadhaar": "Unknown", "confidence": best_sim})

        return results
