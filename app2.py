from flask import Flask, render_template, Response, jsonify, request
import os
import csv
from datetime import datetime
import time

app = Flask(__name__)

# Statistics tracking
detection_history = []
start_time = datetime.now()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/status', methods=['POST'])
def update_status():
    try:
        data = request.json
        if data and "timestamp" in data:
            # Add to detection history
            detection_history.append({
                "timestamp": datetime.now(),
                "confidence": data.get("confidence", 0),
                "in_restricted_zone": data.get("in_restricted_zone", False)
            })
            
            # Keep only the last 1000 detections
            if len(detection_history) > 1000:
                detection_history.pop(0)
                
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

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