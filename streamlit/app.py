import streamlit as st
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
from ultralytics import YOLO
import numpy as np
import pickle
import os
import subprocess

# -----------------------------
# Model Loading Functions
# -----------------------------

@st.cache_resource
def load_yolo_model():
    """
    Load the YOLOv8 model for face detection.
    Returns:
        YOLO model object if successful, else None.
    """
    try:
        model = YOLO("../detection/weights/best.pt")
        st.success("YOLOv8 model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading YOLOv8 model: {e}")
        return None

@st.cache_resource
def load_facenet_models():
    """
    Load the MTCNN and InceptionResnetV1 models for face detection and embedding extraction.
    Returns:
        Tuple of (MTCNN model, InceptionResnetV1 model)
    """
    try:
        mtcnn = MTCNN(keep_all=True)
        resnet = InceptionResnetV1(pretrained='vggface2').eval()
        st.success("MTCNN and InceptionResnetV1 models loaded successfully.")
        return mtcnn, resnet
    except Exception as e:
        st.error(f"Error loading MTCNN/InceptionResnetV1: {e}")
        return None, None

@st.cache_resource
def load_known_embeddings():
    """
    Load precomputed face embeddings from a pickle file.
    Returns:
        Dictionary of known embeddings.
    """
    try:
        with open('../known_embeddings.pkl', 'rb') as f:
            known_embeddings = pickle.load(f)
            st.success("Known embeddings loaded successfully.")
            return known_embeddings
    except Exception as e:
        st.warning(f"Error loading known embeddings: {e}")
        return {}

# -----------------------------
# Helper Functions
# -----------------------------

def compare_embeddings(embedding, known_embeddings, threshold=0.8):
    """
    Compare the input embedding with known embeddings to identify the face.
    Args:
        embedding: Numpy array of the face embedding.
        known_embeddings: Dictionary of stored embeddings.
        threshold: Distance threshold for face matching.
    Returns:
        Name of the matched person or 'Unknown'.
    """
    min_dist = float('inf')
    match = "Unknown"

    for name, known_embedding_list in known_embeddings.items():
        for known_embedding in known_embedding_list:
            dist = np.linalg.norm(embedding - known_embedding)
            if dist < min_dist:
                min_dist = dist
                match = name if dist < threshold else "Unknown"

    st.write(f"Min distance: {min_dist}, Match: {match}")
    return match

def save_embedding(username, embedding):
    """
    Save the embedding for a new user.
    Args:
        username: Name of the user.
        embedding: Numpy array of the face embedding.
    """
    try:
        if os.path.exists('../known_embeddings.pkl'):
            with open('../known_embeddings.pkl', 'rb') as f:
                known_embeddings = pickle.load(f)
        else:
            known_embeddings = {}

        if username in known_embeddings:
            known_embeddings[username].append(embedding)
        else:
            known_embeddings[username] = [embedding]

        with open('../known_embeddings.pkl', 'wb') as f:
            pickle.dump(known_embeddings, f)
        st.success(f"Embedding saved for {username}.")
    except Exception as e:
        st.error(f"Error saving embedding: {e}")

# -----------------------------
# Streamlit App UI
# -----------------------------

st.title("Face Recognition System")

# Sidebar for navigation
option = st.sidebar.selectbox("Choose an option", ["Login", "Signup"])

# Load models
yolo_model = load_yolo_model()
mtcnn, resnet = load_facenet_models()
known_embeddings = load_known_embeddings()

if option == "Login":
    st.header("Login")
    login_option = st.radio("Choose login method", ["Webcam", "Upload Photo"])

    if login_option == "Webcam":
        st.write("Please allow access to your webcam.")
        webcam_image = st.camera_input("Take a picture")

        if webcam_image is not None and yolo_model and mtcnn and resnet:
            # Convert the image to OpenCV format
            file_bytes = np.asarray(bytearray(webcam_image.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            # Detect faces using YOLOv8
            results = yolo_model(image)
            boxes = results[0].boxes

            if boxes is None or len(boxes) == 0:
                st.warning("No faces detected by YOLOv8.")
            else:
                st.write(f"Number of faces detected: {len(boxes)}")

                faces = []
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                    face = image[y1:y2, x1:x2]
                    faces.append(face)
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                st.image(image, caption="Detected Faces", use_container_width=True)

                embeddings = []
                for face in faces:
                    # Convert face to RGB for MTCNN
                    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face_tensor = mtcnn(face_rgb)

                    if face_tensor is not None:
                        if face_tensor.dim() == 4:
                            face_embedding = resnet(face_tensor).detach().cpu().numpy().flatten()
                            embeddings.append(face_embedding)
                            st.write("Embedding extracted for a face.")

                # Compare extracted embeddings
                for embedding in embeddings:
                    match = compare_embeddings(embedding, known_embeddings)
                    if match != "Unknown":
                        st.success(f"Welcome to the system, {match}!")
                        break
                    else:
                        st.error("User not recognized. Please sign up.")

    elif login_option == "Upload Photo":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None and yolo_model and mtcnn and resnet:
            # Read the image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Detect faces using YOLOv8
            results = yolo_model(image)
            boxes = results[0].boxes

            if boxes is None or len(boxes) == 0:
                st.warning("No faces detected by YOLOv8.")
            else:
                st.write(f"Number of faces detected: {len(boxes)}")

                faces = []
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                    face = image[y1:y2, x1:x2]
                    faces.append(face)
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                st.image(image, caption="Detected Faces", use_container_width=True)

                embeddings = []
                for face in faces:
                    # Convert face to RGB for MTCNN
                    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face_tensor = mtcnn(face_rgb)

                    if face_tensor is not None:
                        if face_tensor.dim() == 4:
                            face_embedding = resnet(face_tensor).detach().cpu().numpy().flatten()
                            embeddings.append(face_embedding)
                            st.write("Embedding extracted for a face.")

                # Compare extracted embeddings
                for embedding in embeddings:
                    match = compare_embeddings(embedding, known_embeddings)
                    if match != "Unknown":
                        st.success(f"Welcome to the system, {match}!")
                        break
                    else:
                        st.error("User not recognized. Please sign up.")

elif option == "Signup":
    st.header("Signup")

    # Check if dataset folder exists
    if not os.path.exists("../dataset"):
        os.makedirs("../dataset")
        st.success("Dataset folder created.")

    username = st.text_input("Enter your username")
    if username:
        user_folder = os.path.join("../dataset", username)
        if not os.path.exists(user_folder):
            os.makedirs(user_folder)
            st.success(f"Folder created for user: {username}")

        uploaded_files = st.file_uploader("Upload your images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if uploaded_files:
            for uploaded_file in uploaded_files:
                # Save the uploaded file to the user's folder
                with open(os.path.join(user_folder, uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"Saved {uploaded_file.name} to {user_folder}")

            # Run the embedding script
            try:
                result = subprocess.run(
                    ["python", "generate_face_embeddings.py"],
                    capture_output=True, text=True, check=True
                )
                st.success("Embedding script executed successfully!")
                st.write(result.stdout)
                st.success("Signup complete! Redirecting to login page...")
            except subprocess.CalledProcessError as e:
                st.error("Error while running the embedding script.")
                st.write(e.stderr)