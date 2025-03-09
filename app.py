import uuid
from flask import Flask, request, jsonify
import os
from PIL import Image
from ultralytics import YOLO
import cloudinary
from cloudinary.uploader import upload

app = Flask(__name__)

# Cloudinary Configuration
cloudinary.config(
    cloud_name="dsbpi1dsg",
    api_key="945415891443865",
    api_secret="W0rh0mYDvhM2GQTXv-_lVuoXwI4"
)

# Create necessary directories
os.makedirs("runs", exist_ok=True)

MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)

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

    # Generate unique filename
    # file_ext = file.filename.rsplit('.', 1)[-1].lower()
    # filename = f"{uuid.uuid4()}.{file_ext}"
    # file_path = os.path.join("uploads", filename)
    # image.save(file_path)

    # Perform detection
    model.predict(image, project="runs/detect", save=True)

    # Find latest saved image in detection folder
    detect_folder = max(os.listdir("runs/detect"), key=lambda x: os.path.getctime(os.path.join("runs/detect", x)))
    detected_images = os.listdir(f"runs/detect/{detect_folder}")

    if not detected_images:
        return jsonify({"error": "No detection output found"}), 500


    detected_filename = detected_images[0]
    detected_file_path = os.path.join("runs/detect", detect_folder, detected_filename)

    # Upload to Cloudinary
    try:
        cloudinary_response = upload(detected_file_path)
        cloudinary_url = cloudinary_response.get("secure_url")
    except Exception as e:
        return jsonify({"error": "Failed to upload to Cloudinary", "details": str(e)}), 500

    # Return image URL
    return jsonify({"message": "Success!", "image_url": cloudinary_url})


if __name__ == "__main__":
    app.run(debug=True)
