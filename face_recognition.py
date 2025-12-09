import os
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
from starlette import status
from ultralytics import YOLO
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
import uvicorn
import base64
from pydantic import BaseModel
from typing import Optional

print(">>> LOADED LOCAL face_recognition.py API FILE <<<")
print(">>> __file__ =", __file__)

app = FastAPI()


# Load models and embeddings
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


def compare_embeddings(embedding, known_embeddings, threshold=2.0):
    """
    Compare a new embedding against known embeddings.
    For debugging/demo, we log distances and use a more relaxed threshold.
    """
    if embedding is None or not known_embeddings:
        print("compare_embeddings: No embedding or empty known_embeddings.")
        return "Unknown"

    min_dist = float("inf")
    best_match = "Unknown"

    for name, known_embedding_list in known_embeddings.items():
        for idx, known_embedding in enumerate(known_embedding_list):
            try:
                dist = np.linalg.norm(np.array(embedding) - np.array(known_embedding))
                print(f"[COMPARE] {name}[{idx}] distance = {dist}")
                if dist < min_dist:
                    min_dist = dist
                    best_match = name
            except Exception as e:
                print(f"Error computing distance for {name}: {e}")
                continue

    print(f"[COMPARE] best_match = {best_match}, min_dist = {min_dist}, threshold = {threshold}")

    # For demo: much more forgiving than before
    if min_dist < threshold:
        return best_match
    else:
        return "Unknown"


@app.on_event("startup")
async def startup_event():
    global yolo_model, mtcnn, resnet, known_embeddings
    yolo_model = load_yolo_model("detection/weights/best.pt")
    mtcnn, resnet = load_face_recognition_models()
    known_embeddings = load_known_embeddings()


class SignupResponse(BaseModel):
    status: str
    message: str
    file_path: Optional[str] = None


class RecognizeRequest(BaseModel):
    image_base64: str


@app.post("/recognize")
async def recognize(request: RecognizeRequest):
    global yolo_model, mtcnn, resnet, known_embeddings

    if yolo_model is None or mtcnn is None or resnet is None or not known_embeddings:
        raise HTTPException(status_code=500, detail="Models or embeddings not loaded correctly.")

    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image_base64)
        np_img = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image format. Please upload a valid image.")

        # Convert to RGB and detect faces
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = mtcnn(frame_rgb)

        if faces is None:
            return {"status": "failure", "message": "No face detected.", "status_code": status.HTTP_404_NOT_FOUND}

        # If faces detected, process each
        for face in faces:
            if face is None:
                continue

            face_embedding = resnet(face.unsqueeze(0)).detach().cpu().numpy().flatten()
            match = compare_embeddings(face_embedding, known_embeddings)

            if match != "Unknown":
                if "_" in match:
                    name, user_id = match.split("_",1)
                else:
                    name, user_id = match, ""
                match = match.split("_")
                return {"status": "success", "user": name, "user_id": user_id, "status_code": status.HTTP_200_OK}

        return {"status": "failure", "message": "Face not recognized. Please sign up.",
                "status_code": status.HTTP_404_NOT_FOUND}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during recognition: {str(e)}")


@app.post("/signup", response_model=SignupResponse)
async def signup(
        username: str = Form(...),
        user_id: str = Form(...),
        image_base64: str = Form(...)
):
    try:
        user_folder = os.path.join("dataset", f"{username}_{user_id}")
        os.makedirs(user_folder, exist_ok=True)

        # Decode base64 image
        image_data = base64.b64decode(image_base64)
        np_img = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image format. Please upload a valid image.")

        file_path = os.path.join(user_folder, f"{username}_{user_id}.jpg")
        cv2.imwrite(file_path, frame)

        return SignupResponse(status="success", message="User signed up successfully.", file_path=file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during signup: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
