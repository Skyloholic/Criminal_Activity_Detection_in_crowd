from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
import cv2
from ultralytics import YOLO
import random
import numpy as np
import os

# Flask app initialization
app = Flask(__name__, template_folder='templates')
socketio = SocketIO(app)

# Paths and constants
weights_path = r"D:\Model\Criminal_Activity_Detection_in_Crowd\Backend\weights\yolov8n.pt"
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"Weights file not found at {weights_path}")

# Load YOLO model
model = YOLO(weights_path)

# Load COCO classes
with open(r"D:\Model\Criminal_Activity_Detection_in_Crowd\Backend\utils\coco.txt", "r") as f:
    class_list = f.read().splitlines()

# Detection colors
detection_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in class_list]

# Movement detection parameters
movement_threshold = 30
alert_trigger_frames = 5
alert_counter = 0
alert_active = False
prev_boxes = None

# Frame dimensions
frame_wid, frame_hyt = 640, 480

# Real-time detection endpoint
def generate_frames():
    global prev_boxes, alert_counter, alert_active
    cap = cv2.VideoCapture("D:/Model/Criminal_Activity_Detection_in_Crowd/Backend/inference/videos/afriq0.MP4")  # Open webcam (or replace with video path for a pre-recorded video)
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream ended or could not receive frame.")
            break

        # Resize frame
        frame = cv2.resize(frame, (frame_wid, frame_hyt))

        # Run YOLO detection
        results = model.predict(source=frame, conf=0.45, verbose=False)
        boxes = []
        person_count = 0

        # Process detections
        for result in results[0].boxes:
            box = result.xyxy[0]
            clsID = int(result.cls)
            conf = float(result.conf)

            # Count persons (class ID 0)
            if clsID == 0:
                person_count += 1
                boxes.append((box, conf))

            # Draw bounding boxes
            bb = list(map(int, box))
            cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), detection_colors[clsID], 3)
            cv2.putText(frame, f"{class_list[clsID]} {conf:.2f}", (bb[0], bb[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Movement detection
        high_movement_detected = False
        if prev_boxes:
            for (box, _), (prev_box, _) in zip(boxes, prev_boxes):
                dx = (box[0] + box[2]) / 2 - (prev_box[0] + prev_box[2]) / 2
                dy = (box[1] + box[3]) / 2 - (prev_box[1] + prev_box[3]) / 2
                movement = np.sqrt(dx**2 + dy**2)

                color = (0, 0, 255) if movement > movement_threshold else (0, 255, 0)
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)

                if movement > movement_threshold:
                    high_movement_detected = True
        else:
            # Draw initial frame boxes in green
            for box, _ in boxes:
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

        # Alert management
        alert_counter = alert_counter + 1 if high_movement_detected else max(alert_counter - 1, 0)
        alert_active = alert_counter >= alert_trigger_frames

        if alert_active:
            socketio.emit("alert", {"message": "Movement Alert!", "person_count": person_count})
            cv2.putText(frame, "ALERT!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display person count
        cv2.putText(frame, f"Person count: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Encode frame for streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # Update previous boxes
        prev_boxes = boxes

    cap.release()

# Flask Routes
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
