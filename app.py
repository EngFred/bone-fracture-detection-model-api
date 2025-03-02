import uuid
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
import torch
import os
from PIL import Image
import io
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

# if not os.path.exists('runs'):
#     os.makedirs('runs')

# if not os.path.exists('uploads'):
#     os.makedirs('uploads')

# Create necessary directories
os.makedirs("runs", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

# Load model once at startup (faster inference)
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
    

    # base_path = os.path.dirname(__file__)
    # file_path = os.path.join(base_path, 'uploads', file.filename)
    # file.save(file_path)

    # global img_path
    # predict.img_path = file.filename

    # file_ext =file.filename.rsplit('.', 1)[1].lower()

    # if(file_ext == 'jpg'):
    #     img = cv2.imread(file_path)
    #     frame = cv2.imencode('.jpg', cv2.UMat(img))[1].tobytes()

    #     image = Image.open(io.BytesIO(frame))
    
    # Read image using PIL (more robust)
    try:
        image = Image.open(file.stream)
    except Exception:
        return jsonify({"error": "Invalid image format"}), 400

    # Generate unique filename
    file_ext = file.filename.rsplit('.', 1)[-1].lower()
    filename = f"{uuid.uuid4()}.{file_ext}"
    file_path = os.path.join("uploads", filename)
    image.save(file_path)

    # Perform detection
    results = model.predict(image, project="runs/detect", save=True)

    # Find latest saved image in detection folder
    detect_folder = max(os.listdir("runs/detect"), key=lambda x: os.path.getctime(os.path.join("runs/detect", x)))
    detected_images = os.listdir(f"runs/detect/{detect_folder}")

    if not detected_images:
        return jsonify({"error": "No detection output found"}), 500

    detected_filename = detected_images[0]
    detected_file_path = os.path.join("runs/detect", detect_folder, detected_filename)

    # Return image URL
    image_url = url_for("serve_detected_image", subfolder=detect_folder, filename=detected_filename, _external=True)
    print(image_url)
    return jsonify({"message": "Success!", "image_url": image_url})


# @app.route('/<path:filename>')
# def display(filename):
#     folder_path = "runs/detect"


#     sub_folders = [ f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f)) ]

#     print(sub_folders)

#     if not sub_folders:
#         return jsonify({'error': 'No detected images found!'}), 404
    
#     latest_subfolder = max(sub_folders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
#     directory = f"{folder_path}/{latest_subfolder}"
#     print("Directory: ", directory)
#     files = os.listdir(directory)
#     latest_file = files[0]
#     print(latest_file)

#     filename = os.path.join(folder_path, latest_subfolder, latest_file)
#     file_ext = filename.rsplit('.', 1)[1].lower()



#     if( file_ext == 'jpg' ):
#         image_url = request.host_url + f"detect/{latest_subfolder}/{latest_file}"
#         return jsonify({
#         'message': 'Success!',
#         'image_url': image_url
#     })
#         #return send_from_directory(directory, latest_file)
#         #flask --app model run --host=0.0.0.0

#     else:
#         return 'Invalid file format!'
    

# Route to serve images
@app.route('/detect/<subfolder>/<filename>')
def serve_detected_image(subfolder, filename):
    directory = os.path.join("runs/detect", subfolder)
    return send_from_directory(directory, filename)

if __name__ == "__main__":
    app.run(debug=True)
