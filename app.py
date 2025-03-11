from flask import Flask, request, jsonify
import os
import shutil
from PIL import Image
from ultralytics import YOLO
import cloudinary
from cloudinary.uploader import upload
from dotenv import load_dotenv
from flask_cors import CORS

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Cloudinary Configuration
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
)

# Create necessary directories
os.makedirs("runs", exist_ok=True)

MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH).to("cpu")

@app.route('/')
def home():
    return jsonify({"message": "Bone Fracture Detection API is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    try:
        image = Image.open(file.stream)
    except Exception:
        return jsonify({"error": "Invalid image format"}), 400

    # Perform detection
    model.predict(image, project="runs/detect", save=True)

    # Find latest saved image in detection folder
    try:
        detect_folder = max(os.listdir("runs/detect"), key=lambda x: os.path.getctime(os.path.join("runs/detect", x)))
        detected_images = os.listdir(f"runs/detect/{detect_folder}")

        if not detected_images:
            return jsonify({"error": "No detection output found"}), 500
    except ValueError:
        return jsonify({"error": "Detection folder not found"}), 500

    detected_filename = detected_images[0]
    detected_file_path = os.path.join("runs/detect", detect_folder, detected_filename)

    # Upload to Cloudinary
    try:
        cloudinary_response = upload(detected_file_path)
        cloudinary_url = cloudinary_response.get("secure_url")
    except Exception as e:
        return jsonify({"error": "Failed to upload to Cloudinary", "details": str(e)}), 500

    # Delete detection folder after returning the result
    try:
        shutil.rmtree(f"runs/detect/{detect_folder}")
    except Exception as e:
        print(f"Warning: Could not delete folder {detect_folder}. Error: {e}")

    # Return image URL
    return jsonify({"message": "Success!", "image_url": cloudinary_url})


if __name__ == "__main__":
    app.run()
