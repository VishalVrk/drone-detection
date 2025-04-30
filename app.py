from flask import Flask, render_template, Response
import cv2
import torch
import numpy as np
import threading
import time

app = Flask(__name__)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
model.conf = 0.5  # Confidence threshold

# Global variables
camera = None
output_frame = None
lock = threading.Lock()

def initialize_camera():
    global camera
    camera = cv2.VideoCapture(0)  # Use 0 for webcam
    return camera.isOpened()

def detect_drones():
    global camera, output_frame, lock
    
    while True:
        if camera is None or not camera.isOpened():
            if not initialize_camera():
                print("Error: Could not initialize camera")
                time.sleep(5)
                continue

        success, frame = camera.read()
        if not success:
            print("Error: Failed to read frame")
            time.sleep(1)
            continue

        # Create a copy of the frame for drawing
        display_frame = frame.copy()
        
        # Run inference with YOLOv5
        results = model(frame)
        
        # Process detections
        detections = results.xyxy[0].cpu().numpy()  # xmin, ymin, xmax, ymax, confidence, class
        
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Draw detection box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Add confidence label
            label = f"{conf*100:.1f}%"
            cv2.putText(display_frame, label, (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Update the output frame
        with lock:
            output_frame = display_frame.copy()
        
        # Small delay
        time.sleep(0.03)  # ~30 FPS

def generate_frames():
    global output_frame, lock
    
    while True:
        with lock:
            if output_frame is None:
                continue
            
            # Encode the frame
            (flag, encoded_image) = cv2.imencode(".jpg", output_frame)
            
            if not flag:
                continue
        
        # Yield the output frame in byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
              bytearray(encoded_image) + b'\r\n')

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    # Start the detection thread
    t = threading.Thread(target=detect_drones)
    t.daemon = True
    t.start()
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5030, debug=True, threaded=True, use_reloader=False)