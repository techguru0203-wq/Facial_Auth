# 🌟 Facial Recognition Authentication System

FastAPI + Streamlit based **facial recognition login system**.  
Users can **sign up** with a face image and later **log in** using only their face.

The backend uses:

- **YOLOv8** for face detection  
- **MTCNN + InceptionResnetV1 (FaceNet)** for face embeddings  
- Cosine/Euclidean distance to identify the closest user match  

The frontend is a simple, demo-friendly **Streamlit** app.

---

## 📌 Key Features

- 👤 **User Sign-Up**
  - Upload a face image during registration
  - Image is saved locally inside `dataset/<username>/`
  - Embeddings are generated and appended to `known_embeddings.pkl`

- 🔐 **Face-based Login**
  - Login using **Webcam** or **Image Upload**
  - Captured/uploaded face is matched against stored embeddings
  - Displays a success message if the user is recognized

- ⚙️ **Backend API (FastAPI)**
  - `POST /signup` → accepts user details + base64 image
  - `POST /recognize` → accepts base64 image and returns best match
  - Interactive docs at: **http://localhost:9000/docs**

- 🖥️ **Frontend UI (Streamlit)**
  - Simple 2-page UI: **Login** and **Signup**
  - Good for demos, POCs, and internal evaluation

---

## 🧰 Tech Stack

| Layer      | Technology                          |
|-----------|--------------------------------------|
| Language  | Python (3.12 recommended)           |
| Backend   | FastAPI + Uvicorn                   |
| Frontend  | Streamlit                           |
| Detection | YOLOv8 (Ultralytics)                |
| Embedding | MTCNN + InceptionResnetV1 (FaceNet) |
| Images    | OpenCV (cv2), NumPy                 |
| Config    | `.env`, python-dotenv               |
| Storage   | Local file system + `.pkl` files    |

> **Note:** Some libraries (e.g., `ultralytics`, `mediapipe`, `facenet-pytorch`) may not yet support Python 3.13+.  
> **Use Python 3.10–3.12** for best compatibility.

---

## 📂 Project Structure

```text
face_recognition/
├── .env                           # Environment configuration (LDAP/AD, etc.)
├── requirements.txt               # Python dependencies
├── face_recognition.py            # FastAPI backend application
├── generate_face_embeddings.py    # Script to generate/update face embeddings
├── known_embeddings.pkl           # Serialized face embeddings (generated)
├── processed_folders.txt          # Tracks which dataset folders have been processed
├── haarcascade_eye.xml            # Legacy OpenCV cascade (not critical for core flow)
│
├── dataset/                       # User image folders (each folder per user)
│   ├── <User1>/
│   │   └── user1_image.jpg
│   ├── <User2>/
│   │   └── user2_image.png
│   └── ...
│
├── detection/                     # YOLO model & related detection utilities
│   └── weights/
│       └── best.pt                # YOLOv8 weights for face detection
│
└── streamlit/                     # Frontend UI (Streamlit)
    ├── main.py                    # Entry point – navigation between pages
    ├── login.py                   # Login page – webcam/upload + POST /recognize
    └── signup.py                  # Signup page – upload + call embedding script
```
# Facial Recognition Authentication System — Setup & Usage Guide

This document contains all prerequisites, setup steps, commands, usage flow, and common troubleshooting notes for the Facial Recognition project.

---

# 📌 Prerequisites

- **Python:** 3.10 – 3.12 (recommended for package compatibility)
- **pip:** Latest version
- **Git:** Installed (optional for version control)
- **Operating System:** Windows / Linux / macOS
- **Internet Access:** Required only for initial dependency installations

---

# 🛠️ Setup Instructions

## 1️⃣ Clone the Repository (or open the existing project)

```bash
git clone https://github.com/techguru0203-wq/Facial_Recognition.git
```
```bash
cd Facial_Recognition/face_recognition
```

## Create & Activate a Virtual Environment
Windows (PowerShell / Git Bash):

```ps1
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

# MacOS
```terminal
source .venv/bin/activate
```

```terminal
python3 -m venv .venv
```

### Install Dependencies

```bash
pip install --upgrade pip
```

```bash
pip install -r requirements.txt
```

### (Optional) Configure .env for LDAP / AD Integration
```yaml
AD_SERVER=your_ad_server
AD_PORT=389
DOMAIN=your.domain
USERNAME=your_username
PASSWORD=your_password
BASE_DN=dc=your,dc=domain,dc=com
```

### If not required,
 leave ```.env``` as-is.

 ## 🧠 Generate Face Embeddings
 Run: -

 ```bash
 python generate_face_embeddings.py
```

This will:

Load YOLOv8 + MTCNN + InceptionResnetV1

Process each user folder inside dataset/

Extract embeddings and save as known_embeddings.pkl

Track processed folders in processed_folders.txt

If you add new user folders later, re-run the command above.

## 🚀 Run Commands

```bash
uvicorn face_recognition:app --reload --host 0.0.0.0 --port 9000
```

### 1.Open API docs:

👉 http://localhost:9000/docs

### 2.Start Streamlit Frontend

In a second terminal, with the venv active:
```bash
cd streamlit
streamlit run main.py
```

UI will open at:

👉 http://localhost:8501
---

🧪 Usage Guide

The system has two main flows: Signup and Login.

## ✳️ Signup Flow

### 1.Open Streamlit → Signup

### 2.Enter:
            Username
            (Optional) User ID

### 3.Upload a clear, front-facing photo of the user

### 4.Click Register

### 5.The system will:

                      Save the image into dataset/<username>/

                      Trigger embedding generation via subprocess

                      Update known_embeddings.pkl + processed_folders.txt

### 6.Success message appears in UI
If console shows “No valid faces found”, the image had no detectable face.
Use a clearer image and try again.

---

## ✳️ Login Flow

### 1.Open Streamlit → Login

### 2.Choose login method:

                          Webcam

                          Upload Photo

### 3.Provide a face image

### 4.Click Login

### 5.Backend:

                Detects face using YOLOv8

                Extracts embedding using MTCNN + ResNet

                Compares with stored embeddings

### 6.If recognized:

                    Shows: “Login successful! Welcome <username>.”

### 7.If not recognized:

                        Shows: “User not recognized. Please sign up.”

                        For testing: Using the same image you used during signup guarantees best recognition accuracy.
---

## Common Issues & Troubleshooting "Coming Soon"