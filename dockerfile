# Use Python 3.10 as the base image
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files
COPY . .

# Expose the FastAPI default port
EXPOSE 8000

RUN python generate_face_embeddings.py
# Start FastAPI with uvicorn
CMD ["sh", "-c", "python generate_face_embeddings.py & uvicorn face_recognition:app --reload --host 0.0.0.0 --port 8000"]
