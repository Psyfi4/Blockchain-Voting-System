# ...existing code...
import cv2
import face_recognition
import sqlite3
from datetime import datetime
from typing import List, Dict, Optional, Callable, Any

def recognize_face(
    frame: Any,
    known_face_encodings: List,
    known_face_details: List[Dict],
    conn: sqlite3.Connection,
    draw_face_info: Callable[[Any, int, int, int, int, str, float], None],
    tolerance: float = 0.6,
) -> List[Dict]:
    """
    Recognize faces in a video frame.

    Args:
        frame: BGR OpenCV frame
        known_face_encodings: list of known face encodings
        known_face_details: parallel list of dicts with 'name' and 'aadhaar_number'
        conn: sqlite3 connection (used for logging / DB lookups)
        draw_face_info: callback(frame, top, right, bottom, left, name, confidence)
        tolerance: recognition tolerance (lower is stricter)
    Returns:
        list of recognized face dicts
    """
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find all face locations and encodings in current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    recognized_faces: List[Dict] = []

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(
            known_face_encodings, face_encoding, tolerance=tolerance
        )

        name = "Unknown"
        aadhaar_number = None
        confidence = 0.0

        if True in matches:
            first_match_index = matches.index(True)
            detail = known_face_details[first_match_index]
            name = detail.get('name', 'Unknown')
            aadhaar_number = detail.get('aadhaar_number')

            # Calculate confidence based on distance (clamped)
            face_distances = face_recognition.face_distance(
                known_face_encodings, face_encoding
            )
            try:
                raw_conf = 1.0 - float(face_distances[first_match_index])
            except Exception:
                raw_conf = 0.0
            confidence = max(0.0, min(1.0, raw_conf))

            # Log recognition and fetch additional details
            log_recognition(conn, aadhaar_number, confidence)
            person_details = get_person_details(conn, aadhaar_number)

            recognized_faces.append({
                'location': (top, right, bottom, left),
                'name': name,
                'aadhaar_number': aadhaar_number,
                'confidence': confidence,
                'details': person_details
            })

        # Draw bounding box and info using provided callback
        try:
            draw_face_info(frame, top, right, bottom, left, name, confidence)
        except Exception:
            # don't fail recognition if drawing fails
            pass

    return recognized_faces


def get_person_details(conn, aadhaar_number):
    cursor = conn.cursor()
    cursor.execute('SELECT name, date_of_birth, gender, address, aadhaar_photo_path FROM persons WHERE aadhaar_number = ?', (aadhaar_number,))
    row = cursor.fetchone()
    if row:
        return {'name': row[0], 'date_of_birth': row[1], 'gender': row[2], 'address': row[3], 'aadhaar_photo_path': row[4]}
    return None


def log_recognition(conn: sqlite3.Connection, aadhaar_number: str, confidence: float) -> None:
    """Log recognition events into recognition_logs table"""
    if not aadhaar_number:
        return
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO recognition_logs (aadhaar_number, recognition_time, confidence)
        VALUES (?, ?, ?)
    ''', (aadhaar_number, datetime.utcnow().isoformat(), float(confidence)))
    conn.commit()
# ...existing code...pip install opencv-python-headless==4.8.1.78

