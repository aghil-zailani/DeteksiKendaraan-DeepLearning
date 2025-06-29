from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import sqlite3
from datetime import datetime
from ultralytics import YOLO
from collections import defaultdict
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Load model
try:
    model = YOLO('model/best.pt') 
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    model = None

# Pastikan folder upload dan result ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

def process_video(input_path, output_path, model):
    """
    Processes a video file with a YOLO model for object detection.
    """
    if model is None:
        print("Model is not loaded. Cannot process video.")
        return 0
        
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 25
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"Error: Could not create VideoWriter for {output_path}")
        cap.release()
        return 0

    total_vehicle_count = 0
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        results = model(frame, conf=0.5, imgsz=640, verbose=False)
        annotated_frame = results[0].plot()
        
        num_detections_in_frame = len(results[0].boxes)
        total_vehicle_count += num_detections_in_frame
        
        if annotated_frame.shape[0] != height or annotated_frame.shape[1] != width:
            annotated_frame = cv2.resize(annotated_frame, (width, height))
            
        out.write(annotated_frame)
        
        frame_idx += 1
        print(f"[INFO] Processed and wrote frame {frame_idx}")
        
    cap.release()
    out.release()
    print("[INFO] Video processing complete. Output saved to:", output_path)
    
    return total_vehicle_count


@app.route('/history')
def history():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM detections ORDER BY timestamp DESC")
    records = cursor.fetchall()
    conn.close()

    grouped = defaultdict(list)
    for record in records:
        filename = record[1]
        grouped[filename].append({
            'id': record[0],
            'label': record[2],
            'confidence': record[3],
            'waktu': record[4]
        })
    
    return render_template('history.html', grouped=grouped, active_page='history')


@app.route('/')
def index():
    # This route now only renders the dashboard template.
    # It doesn't need to pass any specific data for now.
    return render_template("index.html", active_page='dashboard')


@app.route('/detect', methods=['GET', 'POST'])
def detect():
    # Initialize variables for the detection result section
    result_image = None
    result_video = None
    labels_detected = []
    detection_details = []
    total_detected = None
    
    # Connect to database and get initial stats for the 'detect' page
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    # Get total unique images uploaded
    cursor.execute("SELECT COUNT(DISTINCT filename) FROM detections WHERE filename LIKE '%.jpg' OR filename LIKE '%.jpeg' OR filename LIKE '%.png'")
    total_images = cursor.fetchone()[0]

    # Get label counts for the chart
    cursor.execute("SELECT label, COUNT(*) FROM detections GROUP BY label")
    label_data = cursor.fetchall()
    label_counts = {label: count for label, count in label_data}
    conn.close() # Close connection for GET request

    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)

        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)

        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            ext = os.path.splitext(filename)[1].lower()

            if ext in ['.jpg', '.jpeg', '.png']:
                # Process image
                results = model(filepath)
                annotated_image = results[0].plot()
                
                # Re-establish connection for POST request
                conn = sqlite3.connect('database.db')
                cursor = conn.cursor()

                for pred in results[0].boxes:
                    class_id = int(pred.cls.cpu().numpy())
                    label = model.names[class_id]
                    confidence = float(pred.conf.cpu().numpy())

                    labels_detected.append(label)
                    detection_details.append({
                        'label': label,
                        'confidence': confidence
                    })

                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    cursor.execute("INSERT INTO detections (filename, label, confidence, timestamp) VALUES (?, ?, ?, ?)",
                                   (filename, label, confidence, timestamp))
                
                conn.commit()
                conn.close()

                # Save the annotated image
                result_filename = 'result_' + filename
                result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
                cv2.imwrite(result_path, annotated_image)
                result_image = result_filename

            elif ext in ['.mp4', '.avi', '.mov']:
                # Process video
                base_name = os.path.splitext(filename)[0]
                timestamp_str = datetime.now().strftime("%Y%m%d%H%M%S")
                result_filename = f'result_{base_name}_{timestamp_str}.mp4'
                result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
                
                total_detected = process_video(filepath, result_path, model)
                result_video = result_filename
            
            # After a POST, re-fetch stats to update the page immediately
            conn = sqlite3.connect('database.db')
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(DISTINCT filename) FROM detections WHERE filename LIKE '%.jpg' OR filename LIKE '%.jpeg' OR filename LIKE '%.png'")
            total_images = cursor.fetchone()[0]
            cursor.execute("SELECT label, COUNT(*) FROM detections GROUP BY label")
            label_data = cursor.fetchall()
            label_counts = {label: count for label, count in label_data}
            conn.close()
                
    return render_template("detect.html",
                           total_images=total_images,
                           label_counts=label_counts or {},
                           result_image=result_image,
                           result_video=result_video,
                           labels_detected=labels_detected,
                           detection_details=detection_details,
                           total_detected=total_detected,
                           active_page='detect')


if __name__ == '__main__':
    # Initialize the database on startup
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS detections (id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT, label TEXT, confidence REAL, timestamp TEXT)")
    conn.close()
    
    app.run(debug=True, port=5001)