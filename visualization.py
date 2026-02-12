import cv2
import numpy as np
import os
from datetime import datetime


class FaceVisualizer:
    """Handles drawing and displaying face recognition results."""

    def draw_face_info(self, frame, top, right, bottom, left, name, confidence):
        """Draw bounding box and information on the frame"""
        # Draw bounding box
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # Create background for text
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        
        # Draw name and confidence
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, f"{name} ({confidence:.2f})", 
                    (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    def display_person_details(self, person_data):
        """Display detailed person information including Aadhaar photo"""
        if not person_data:
            print("No person data provided")
            return
        
        # Create a white window to display details
        details_window = np.ones((400, 600, 3), dtype=np.uint8) * 255
        
        # Load Aadhaar photo if available
        aadhaar_photo = None
        if 'aadhaar_photo_path' in person_data and os.path.exists(person_data['aadhaar_photo_path']):
            aadhaar_photo = cv2.imread(person_data['aadhaar_photo_path'])
            if aadhaar_photo is not None:
                aadhaar_photo = cv2.resize(aadhaar_photo, (200, 150))
        
        # Add text details
        y_offset = 30
        line_height = 25
        
        details = [
            f"Name: {person_data.get('name', 'N/A')}",
            f"Aadhaar Number: {person_data.get('aadhaar_number', 'N/A')}",
            f"Date of Birth: {person_data.get('date_of_birth', 'N/A')}",
            f"Gender: {person_data.get('gender', 'N/A')}",
            f"Address: {person_data.get('address', 'N/A')}",
            f"Recognition Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ]
        
        for i, detail in enumerate(details):
            cv2.putText(details_window, detail, (10, y_offset + i * line_height), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Add Aadhaar photo if available
        if aadhaar_photo is not None:
            h, w = aadhaar_photo.shape[:2]
            details_window[10:10+h, 380:380+w] = aadhaar_photo
            cv2.putText(details_window, "Aadhaar Card", (380, 170), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Display window
        cv2.imshow('Person Details', details_window)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
