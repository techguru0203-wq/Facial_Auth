from ldap3 import Server, Connection, ALL, SUBTREE
import os
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
from ultralytics import YOLO
import pickle
import torch
from tqdm import tqdm  # For progress bars
from dotenv import load_dotenv

load_dotenv()

# -------------------- AD Configuration -------------------- #

AD_SERVER = os.getenv("AD_SERVER")
AD_PORT = int(os.getenv("AD_PORT"))
DOMAIN = os.getenv("DOMAIN")
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")
BASE_DN = os.getenv("BASE_DN")


# Attributes to fetch for each user
ATTRIBUTES = [
    'cn',  # Common Name
    'sAMAccountName',  # Username
    'mail',  # Email
    'displayName',  # Display Name
    'givenName',  # First Name
    'sn',  # Last Name
    'title',  # Job Title
    'department',  # Department
    'telephoneNumber',  # Phone Number
    'thumbnailPhoto'  # Profile Picture (if available)
]

# Directory to save thumbnail images
IMAGE_FOLDER = 'dataset'

# -------------------- Helper Functions -------------------- #

def resize_image(img, target_size=640):
    """
    Resize the image while maintaining the aspect ratio.
    """
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    resized_img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return resized_img


# -------------------- Model Initialization -------------------- #

def initialize_models():
    """
    Initialize YOLOv8, MTCNN, and InceptionResnetV1 models.
    """
    try:
        # Load YOLOv8 face detection model (ensure it's trained for accessories)
        yolo_face_model = YOLO("detection/weights/best.pt")
        print("YOLOv8 face detection model loaded successfully.")

        # Load MTCNN for face alignment (optional, can be replaced with YOLO)
        mtcnn_model = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')

        # Load InceptionResnetV1 for face embeddings (fine-tuned for accessories)
        resnet_model = InceptionResnetV1(pretrained='vggface2').eval()
        if torch.cuda.is_available():
            resnet_model = resnet_model.to('cuda')
        print("MTCNN and InceptionResnetV1 models loaded successfully.")

        return yolo_face_model, mtcnn_model, resnet_model
    except Exception as e:
        print(f"Error initializing models: {e}")
        return None, None, None


# -------------------- Load Known Embeddings -------------------- #

def load_known_embeddings(filepath='known_embeddings.pkl'):
    """
    Load known face embeddings from a pickle file.
    """
    try:
        with open(filepath, 'rb') as file:
            embeddings = pickle.load(file)
            print(f"Known embeddings loaded successfully from {filepath}.")
            return embeddings
    except FileNotFoundError:
        print(f"No known embeddings found at {filepath}. Starting with an empty dictionary.")
        return {}
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return {}


# -------------------- Save Embeddings -------------------- #

def save_embeddings(embeddings, filepath='known_embeddings.pkl'):
    """
    Save known face embeddings to a pickle file.
    """
    try:
        with open(filepath, 'wb') as file:
            pickle.dump(embeddings, file)
            print(f"Known embeddings saved successfully to {filepath}.")
    except Exception as e:
        print(f"Error saving embeddings: {e}")


# -------------------- Load Processed Folders -------------------- #

def load_processed_folders(filepath='processed_folders.txt'):
    """
    Load the list of already processed folders.
    """
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as file:
                processed = file.read().splitlines()
                print(f"Processed folders loaded successfully from {filepath}.")
                return set(processed)
        return set()
    except Exception as e:
        print(f"Error loading processed folders: {e}")
        return set()


# -------------------- Save Processed Folders -------------------- #

def save_processed_folders(processed_folders, filepath='processed_folders.txt'):
    """
    Save the list of processed folders to a text file.
    """
    try:
        with open(filepath, 'w') as file:
            file.write("\n".join(processed_folders))
            print(f"Processed folders saved successfully to {filepath}.")
    except Exception as e:
        print(f"Error saving processed folders: {e}")


# -------------------- Extract and Save Embeddings -------------------- #

def save_embeddings_from_directory(directory_path, yolo_face_model, mtcnn_model, resnet_model, known_embeddings, processed_folders):
    """
    Extract face embeddings from images in a directory and save them.
    """
    if not os.path.isdir(directory_path):
        print(f"Error: '{directory_path}' is not a valid directory.")
        return

    for person_dir in tqdm(os.listdir(directory_path), desc="Checking directories"):
        if person_dir in processed_folders:
            print(f"'{person_dir}' already processed. Skipping.")
            continue

        person_path = os.path.join(directory_path, person_dir)
        if os.path.isdir(person_path):
            print(f"\nProcessing '{person_dir}' directory...")
            person_embeddings = []

            for filename in tqdm(os.listdir(person_path), desc=f"Processing images in {person_dir}"):
                if filename.lower().endswith(('.jpg', '.png', '.bmp', '.jpeg')):
                    image_path = os.path.join(person_path, filename)
                    img = cv2.imread(image_path)
                    if img is None:
                        print(f"Error: Unable to read image '{image_path}'.")
                        continue

                    # Resize image for better face detection
                    img_resized = resize_image(img, target_size=640)

                    # Detect faces with YOLO (trained for accessories)
                    face_results = yolo_face_model(img_resized)
                    face_boxes = face_results[0].boxes.xyxy.cpu().numpy()

                    if len(face_boxes) == 0:
                        print(f"No faces detected in '{filename}'. Skipping.")
                        continue

                    for box in face_boxes:
                        x1, y1, x2, y2 = map(int, box)
                        face = img_resized[y1:y2, x1:x2]

                        # Resize face to 160x160 for InceptionResnetV1
                        face_resized = cv2.resize(face, (160, 160))
                        face_tensor = torch.tensor(face_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0

                        if torch.cuda.is_available():
                            face_tensor = face_tensor.to('cuda')

                        # Get face embedding
                        embedding = resnet_model(face_tensor).detach().cpu().numpy().flatten()
                        person_embeddings.append(embedding)
                        print(f"Embedding extracted for '{filename}'.")

            if person_embeddings:
                # Store multiple embeddings for each user
                if person_dir not in known_embeddings:
                    known_embeddings[person_dir] = []
                known_embeddings[person_dir].extend(person_embeddings)
                processed_folders.add(person_dir)
                print(f"Embeddings saved for '{person_dir}'.")
            else:
                print(f"No valid faces found for '{person_dir}'. Skipping.")

    save_embeddings(known_embeddings)
    save_processed_folders(processed_folders)


# -------------------- Fetch Users from AD -------------------- #

def fetch_all_users():
    """
    Fetch all users from Active Directory, save their thumbnails, and write user details to a text file.
    """
    try:
        # Connect to the AD server
        server = Server(AD_SERVER, port=AD_PORT, get_info=ALL)
        conn = Connection(server, user=USERNAME, password=PASSWORD, auto_bind=True)
        if not conn.bind():
            print("Failed to authenticate, please check your credentials.")
            return

        print("Successfully connected to Active Directory!")

        # Search for all users in the AD
        search_filter = '(objectClass=person)'  # Filter to get all user objects
        conn.search(search_base=BASE_DN, search_filter=search_filter, search_scope=SUBTREE, attributes=ATTRIBUTES)

        # Create the images folder if it doesn't exist
        if not os.path.exists(IMAGE_FOLDER):
            os.makedirs(IMAGE_FOLDER)

        # Create or open a text file to save user details
        output_file = 'ad_users.txt'
        with open(output_file, 'w', encoding='utf-8') as file:
            # Print user details and save thumbnails
            print(f"Found {len(conn.entries)} users in Active Directory:")
            for entry in conn.entries:
                file.write("\nUser Details:\n")
                for attr in ATTRIBUTES:
                    if attr in entry:
                        file.write(f"{attr}: {entry[attr].value}\n")
                file.write("-" * 40 + "\n")

                # Save thumbnail photo if available
                if 'thumbnailPhoto' in entry and entry['thumbnailPhoto'].value:
                    username = entry['sAMAccountName'].value
                    display_name = entry['displayName'].value
                    thumbnail_data = entry['thumbnailPhoto'].value

                    # Create folder in the format 'displayName_username'
                    folder_name = f"{display_name}_{username}"
                    user_folder = os.path.join(IMAGE_FOLDER, folder_name)
                    if not os.path.exists(user_folder):
                        os.makedirs(user_folder)

                    # Save the thumbnail image
                    image_path = os.path.join(user_folder, f"{display_name}_{username}.jpg")
                    with open(image_path, 'wb') as img_file:
                        img_file.write(thumbnail_data)
                    print(f"Thumbnail saved for {username} at {image_path}")

        conn.unbind()
        print("Disconnected from Active Directory.")
        print(f"User details saved to {output_file}.")

    except Exception as e:
        print(f"Error occurred: {e}")


# -------------------- Main Execution -------------------- #

def main():
    """
    Main function to authenticate with AD, fetch users, and generate embeddings.
    """
    # Step 1: Authenticate with AD and fetch users
    fetch_all_users()

    # Step 2: Initialize models
    yolo_face_model, mtcnn_model, resnet_model = initialize_models()
    if yolo_face_model is None or mtcnn_model is None or resnet_model is None:
        print("Failed to initialize models. Exiting.")
        return

    # Step 3: Load known embeddings and processed folders
    known_embeddings = load_known_embeddings()
    processed_folders = load_processed_folders()

    # Step 4: Save embeddings from the dataset
    save_embeddings_from_directory(IMAGE_FOLDER, yolo_face_model, mtcnn_model, resnet_model, known_embeddings, processed_folders)


if __name__ == "__main__":
    main()