from flask import Flask, render_template, request, jsonify
import torch
from PIL import Image
import io
import numpy as np

app = Flask(__name__)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
model.conf = 0.5  # Confidence threshold

@app.route('/')
def index():
    return render_template('index1.html')  # Your frontend HTML page

@app.route('/detect', methods=['POST'])
def detect_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    try:
        # Get the image from the request
        image_file = request.files['image']
        image_bytes = image_file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Print image size for debugging
        print(f"Processing image of size: {img.size}")

        # Run YOLOv5 detection
        results = model(img)
        
        # Extract bounding box results - with explicit debugging
        pandas_results = results.pandas().xyxy[0]
        print(f"YOLOv5 found {len(pandas_results)} detections")
        
        # Use to_json() for debug print
        print(f"Raw detection data: {pandas_results.to_json(orient='records')}")
        
        # Convert to dict with explicit type conversion
        raw_detections = pandas_results.to_dict(orient='records')
        
        # Create formatted detections with explicit float conversion
        formatted_detections = []
        for det in raw_detections:
            print(f"Processing detection: {det}")
            formatted_det = {
                'xmin': float(det['xmin']),
                'ymin': float(det['ymin']),
                'xmax': float(det['xmax']),
                'ymax': float(det['ymax']),
                'confidence': float(det['confidence']),
                'name': str(det['name'])
            }
            print(f"Formatted detection: {formatted_det}")
            formatted_detections.append(formatted_det)

        print(f"Returning {len(formatted_detections)} detections")
        return jsonify(formatted_detections)
        
    except Exception as e:
        print(f"Error in detection: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5034, debug=True)