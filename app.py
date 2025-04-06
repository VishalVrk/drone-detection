from flask import Flask, render_template, Response, jsonify, request
import cv2
import yolov5
import numpy as np
from datetime import datetime
import threading
import os
import csv
import time

app = Flask(__name__)
model = yolov5.load('best.pt')

# Initialize detection parameters
rectangle_coords = [(50, 50), (250, 50), (250, 250), (50, 250)]
detection_status = {"detected": False, "last_seen": None, "confidence": None, "in_restricted_zone": False}
confidence_threshold = 0.5
show_zone = True

# Statistics tracking
detection_history = []
start_time = datetime.now()

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, size=640)
        boxes = results.xyxy[0].numpy()
        detected = False
        max_conf = 0
        in_zone = False

        for result in boxes:
            x1, y1, x2, y2, conf, cls = result
            if conf > confidence_threshold:
                detected = True
                max_conf = max(max_conf, conf)
                
                # Draw the detection rectangle
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                
                # Add label with confidence
                label = f"{conf*100:.1f}%"
                cv2.putText(frame, label, (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Check if drone is in restricted zone
                drone_points = [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]
                for point in drone_points:
                    if (rectangle_coords[0][0] <= point[0] <= rectangle_coords[2][0] and
                        rectangle_coords[0][1] <= point[1] <= rectangle_coords[2][1]):
                        in_zone = True
                        break
                
                # Add warning label if in restricted zone
                if in_zone:
                    cv2.putText(frame, "⚠️ RESTRICTED AREA VIOLATION", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Update detection status
        if detected:
            detection_status["detected"] = True
            detection_status["last_seen"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            detection_status["confidence"] = f"{max_conf*100:.1f}"
            detection_status["in_restricted_zone"] = in_zone
            
            # Add to detection history if it's a new detection or zone status changed
            add_to_history(max_conf, in_zone)
        else:
            detection_status["detected"] = False

        # Draw zone if enabled
        if show_zone:
            # Draw the restricted zone
            for i in range(4):
                cv2.circle(frame, rectangle_coords[i], 5, (0, 255, 0), -1)
                cv2.line(frame, rectangle_coords[i], rectangle_coords[(i + 1) % 4], (0, 255, 0), 2)
                
            # Add zone label
            cv2.putText(frame, "RESTRICTED ZONE", 
                      (rectangle_coords[0][0], rectangle_coords[0][1] - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Add timestamp
        cv2.putText(frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 255, 255), 1)

        # Add system info
        uptime = datetime.now() - start_time
        hours, remainder = divmod(uptime.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        uptime_str = f"{hours:02}:{minutes:02}:{seconds:02}"
        cv2.putText(frame, f"SYSTEM UPTIME: {uptime_str}", 
                    (frame.shape[1] - 250, frame.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

def add_to_history(confidence, in_zone):
    # Add detection to history
    timestamp = datetime.now()
    detection_history.append({
        "timestamp": timestamp,
        "confidence": confidence,
        "in_restricted_zone": in_zone
    })
    
    # Keep only the last 1000 detections
    if len(detection_history) > 1000:
        detection_history.pop(0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return jsonify(detection_status)

@app.route('/stats')
def stats():
    total_detections = len(detection_history)
    zone_violations = sum(1 for det in detection_history if det["in_restricted_zone"])
    
    avg_confidence = 0
    if total_detections > 0:
        avg_confidence = sum(det["confidence"] for det in detection_history) / total_detections
    
    uptime = datetime.now() - start_time
    uptime_seconds = uptime.total_seconds()
    
    return jsonify({
        "total_detections": total_detections,
        "zone_violations": zone_violations,
        "avg_confidence": f"{avg_confidence*100:.1f}",
        "uptime_seconds": uptime_seconds
    })

@app.route('/update_zone', methods=['POST'])
def update_zone():
    try:
        data = request.json
        x = int(data.get('x', 50))
        y = int(data.get('y', 50))
        width = int(data.get('width', 200))
        height = int(data.get('height', 200))
        
        global rectangle_coords
        rectangle_coords = [
            (x, y),
            (x + width, y),
            (x + width, y + height),
            (x, y + height)
        ]
        
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/update_threshold', methods=['POST'])
def update_threshold():
    try:
        data = request.json
        global confidence_threshold
        confidence_threshold = float(data.get('threshold', 0.5))
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/toggle_zone', methods=['POST'])
def toggle_zone():
    try:
        global show_zone
        show_zone = not show_zone
        return jsonify({"success": True, "show_zone": show_zone})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/export_logs')
def export_logs():
    try:
        filename = f"drone_detection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(os.path.join('static', filename), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Timestamp', 'Confidence', 'In Restricted Zone'])
            
            for detection in detection_history:
                writer.writerow([
                    detection["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                    f"{detection['confidence']*100:.1f}%",
                    "Yes" if detection["in_restricted_zone"] else "No"
                ])
                
        return jsonify({"success": True, "filename": f"/static/{filename}"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == "__main__":
    # Create static directory if it doesn't exist
    if not os.path.exists('static'):
        os.makedirs('static')
        
    app.run(debug=True)