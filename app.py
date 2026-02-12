import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from aadhaar_system import AadhaarFaceSystem

st.set_page_config(layout="wide")
system = AadhaarFaceSystem()

st.title("Aadhaar Face Recognition System")

mode = st.sidebar.selectbox(
    "Select Mode",
    ["Register", "Recognize"]
)

face_frame = None
aadhaar_frame = None

class VideoProcessor(VideoTransformerBase):
    def transform(self, frame):
        global face_frame
        img = frame.to_ndarray(format="bgr24")

        if mode == "Recognize":
            results = system.recognize(img)
            for top, right, bottom, left, label in results:
                cv2.rectangle(img, (left, top), (right, bottom), (0,255,0), 2)
                cv2.putText(img, label, (left, top-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        face_frame = img.copy()
        return img

ctx = webrtc_streamer(
    key="camera",
    video_transformer_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False}
)

if mode == "Register":
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Capture Face"):
            face_img = face_frame.copy()
            st.session_state["face"] = face_img
            st.success("Face captured")

    with col2:
        if st.button("Capture Aadhaar"):
            aadhaar_img = face_frame.copy()
            st.session_state["aadhaar"] = aadhaar_img
            st.success("Aadhaar captured")

    if st.button("Register Person"):
        if "face" not in st.session_state or "aadhaar" not in st.session_state:
            st.error("Capture both face and Aadhaar")
        else:
            ok, msg = system.register(
                st.session_state["face"],
                st.session_state["aadhaar"]
            )
            if ok:
                st.success(msg)
            else:
                st.error(msg)
