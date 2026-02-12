# debug_match.py
import sqlite3, numpy as np, cv2, sys
from insightface.app import FaceAnalysis

DB = "aadhaar_face_db.db"
IMG = "/workspaces/aadhar_2ndtest/.venv/insightface/data/images/t1.jpg"  # registered image

app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640,640))

img = cv2.imread(IMG)
if img is None:
    print("ERROR: image not found:", IMG); sys.exit(1)

faces = app.get(img)
print("InsightFace detected:", len(faces))
if not faces:
    sys.exit(1)

emb = np.array(faces[0].embedding, dtype=np.float32)
print("captured emb shape:", emb.shape)

conn = sqlite3.connect(DB)
rows = conn.execute("SELECT id,aadhaar_number,name,embedding FROM persons").fetchall()
print("DB rows:", len(rows))
for r in rows:
    id_, aad, name, blob = r
    known = np.frombuffer(blob, dtype=np.float32)
    print("known shape:", known.shape, "dtype:", known.dtype)
    a = emb / (np.linalg.norm(emb)+1e-10)
    b = known / (np.linalg.norm(known)+1e-10)
    sim = float(np.dot(a,b))
    print(f"id={id_} aad={aad} name={name} sim={sim:.4f}")
