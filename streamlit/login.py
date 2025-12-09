import os
import sys

# --- Ensure project root is in sys.path so we can import face_recognition.py ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import base64
import cv2
import requests
import streamlit as st
import numpy as np


def login_page():
    st.title("User Login")

    st.write("### Select Login Method")
    option = st.selectbox("Choose an option", ["Webcam", "Upload Photo"])

    image_base64 = None

    # ------ Webcam Capture ------
    if option == "Webcam":
        st.write("### Capture your image")
        camera_image = st.camera_input("Take a picture")

        if camera_image:
            file_bytes = np.asarray(bytearray(camera_image.getvalue()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, 1)

            _, buffer = cv2.imencode('.jpg', frame)
            image_base64 = base64.b64encode(buffer).decode('utf-8')

    # ------ Manual Upload ------
    elif option == "Upload Photo":
        st.write("### Upload your image")
        uploaded_file = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, 1)

            _, buffer = cv2.imencode('.jpg', frame)
            image_base64 = base64.b64encode(buffer).decode('utf-8')

    # ------ Submit Button ------
    if st.button("Login"):
        if not image_base64:
            st.error("Please capture or upload an image first.")
            return

        try:
            response = requests.post(
                "http://localhost:9000/recognize",
                json={"image_base64": image_base64},
                timeout=20
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("user"):
                    st.success(f"Login successful! Welcome, {data['user']} (ID: {data['user_id']}).")
                else:
                    st.warning("User not recognized. Please sign up.")
            else:
                # 🔍 Show real backend error so we can debug
                try:
                    err = response.json()
                except Exception:
                    err = {"detail": response.text}

                st.error(f"Server error ({response.status_code}): {err}")

        except Exception as e:
            st.error(f"Error contacting API: {str(e)}")
