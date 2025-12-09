import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
from ultralytics import YOLO
import pickle
import numpy as np
from fastapi import FastAPI
import time

app = FastAPI()

# Global variables to store the recognized user and control the camera loop
recognized_user = None
camera_active = False
duration = 5
start_time = time.time()


def load_yolo_model(weight_path):
    try:
        model = YOLO(weight_path)
        print("YOLOv8 model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading YOLOv8 model: {e}")
        return None


def load_face_recognition_models():
    try:
        mtcnn = MTCNN(keep_all=True)
        resnet = InceptionResnetV1(pretrained='vggface2').eval()
        print("MTCNN and InceptionResnetV1 models loaded successfully.")
        return mtcnn, resnet
    except Exception as e:
        print(f"Error loading MTCNN/InceptionResnetV1: {e}")
        return None, None


def load_known_embeddings(file_path='known_embeddings.pkl'):
    try:
        with open(file_path, 'rb') as f:
            known_embeddings = pickle.load(f)
            print("Known embeddings loaded successfully.")
            return known_embeddings
    except Exception as e:
        print(f"Error loading known embeddings: {e}")
        return {}


def compare_embeddings(embedding, known_embeddings, threshold=0.8):
    if embedding is None or not known_embeddings:
        return "Unknown"

    min_dist = float('inf')
    match = "Unknown"

    for name, known_embedding_list in known_embeddings.items():
        for known_embedding in known_embedding_list:
            try:
                dist = np.linalg.norm(np.array(embedding) - np.array(known_embedding))
                if dist < min_dist:
                    min_dist = dist
                    match = name if dist < threshold else "Unknown"
            except Exception as e:
                print(f"Error comparing embeddings: {e}")
                continue

    print(f"Min distance: {min_dist}, Match: {match}")
    return match


def detect_eyes(face_image):
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return len(eyes) > 0


def run_camera(yolo_model, mtcnn, resnet, known_embeddings):
    global camera_active, recognized_user
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open the webcam.")
        return

    camera_active = True

    try:
        while camera_active:
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to read from the webcam.")
                break

            results = yolo_model(frame)
            boxes = results[0].boxes if results[0].boxes else []

            faces = []
            for box in boxes:
                try:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    face = frame[y1:y2, x1:x2]
                    faces.append((face, (x1, y1, x2, y2)))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                except Exception as e:
                    print(f"Error processing bounding box: {e}")
                    continue

            for face, (x1, y1, x2, y2) in faces:
                try:
                    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face_tensor = mtcnn(face_rgb)

                    if face_tensor is not None:
                        if face_tensor.ndim == 5:
                            face_tensor = face_tensor.squeeze(0)

                        face_embedding = resnet(face_tensor).detach().cpu().numpy().flatten()
                        match = compare_embeddings(face_embedding, known_embeddings)
                        print(f"Match: {match}")
                        print(detect_eyes(face))
                        if match != "Unknown" and detect_eyes(face):
                            recognized_user = match
                            camera_active = False
                            cap.release()
                            cv2.destroyWindow('Camera Feed')
                            cv2.destroyAllWindows()
                            break
                        else:
                            cap.release()
                            cv2.destroyWindow('Camera Feed')
                            cv2.destroyAllWindows()
                            break
                except Exception as e:
                    print(f"Error generating or comparing embeddings: {e}")
                    continue

            cv2.imshow('Camera Feed', frame)

            print(f"Recognized user: {recognized_user}")
            if recognized_user:  # Optional: Check if any key is pressed
                print("Key pressed. Exiting...")
                break

            # Check if the duration has passed
            if time.time() - start_time >= duration:
                print(f"{duration} seconds have passed. Exiting...")
                break

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    finally:
        print("hereereee")
        camera_active = False
        cap.release()
        cv2.destroyWindow('Camera Feed')
        cv2.destroyAllWindows()