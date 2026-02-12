import face_recognition
import cv2
import sqlite3
from datetime import datetime


def register_person(self, face_image_path, aadhaar_photo_path, person_data):
    """
    Register a new person with face and Aadhaar details
    
    Args:
        face_image_path: Path to clear face image
        aadhaar_photo_path: Path to Aadhaar card photo
        person_data: Dictionary containing person details
    """
    try:
        # Load and encode face
        face_image = face_recognition.load_image_file(face_image_path)
        face_encodings = face_recognition.face_encodings(face_image)

        if not face_encodings:
            return False, "No face detected in the image"

        face_encoding = face_encodings[0]

        # Convert encoding to blob for storage
        encoding_blob = face_encoding.tobytes()

        # Store in database
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO persons 
            (aadhaar_number, name, date_of_birth, gender, address, face_encoding, aadhaar_photo_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            person_data['aadhaar_number'],
            person_data['name'],
            person_data.get('date_of_birth'),
            person_data.get('gender'),
            person_data.get('address'),
            encoding_blob,
            aadhaar_photo_path
        ))

        self.conn.commit()

        # Update in-memory data
        self.known_face_encodings.append(face_encoding)
        self.known_face_details.append({
            'aadhaar_number': person_data['aadhaar_number'],
            'name': person_data['name']
        })

        return True, "Person registered successfully"

    except Exception as e:
        return False, f"Registration failed: {str(e)}"
