from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import sqlite3
from datetime import datetime
from ultralytics import YOLO
from collections import defaultdict

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Load model
model = YOLO('model/best.pt') 

# Pastikan folder upload dan result ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

def process_video(input_path, output_path, model):
    import numpy as np
    import cv2

    cap = cv2.VideoCapture(input_path)

    # Ambil info video
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 25  # fallback default

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output codec dan writer
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_vehicle_count = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize untuk input model
        resized_frame = cv2.resize(frame, (640, 640))

        # Deteksi
        results = model(resized_frame, conf=0.5, imgsz=640, verbose=False)

        # Copy frame asli untuk anotasi
        annotated_frame = frame.copy()

        for pred in results[0].boxes:
            # Ambil koordinat bounding box dari YOLO (dalam skala 640x640)
            x1_resized, y1_resized, x2_resized, y2_resized = map(int, pred.xyxy[0].cpu().numpy())

            # Hitung ulang ke ukuran asli frame
            x_scale = width / 640
            y_scale = height / 640

            x1 = int(x1_resized * x_scale)
            x2 = int(x2_resized * x_scale)
            y1 = int(y1_resized * y_scale)
            y2 = int(y2_resized * y_scale)

            class_id = int(pred.cls.cpu().numpy())
            label = model.names[class_id]
            confidence = float(pred.conf.cpu().numpy())

            # Gambar kotak dan label
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)


            total_vehicle_count += 1

        # Pastikan ukuran frame sesuai dengan VideoWriter
        annotated_frame = cv2.resize(annotated_frame, (width, height))

        out.write(annotated_frame)
        frame_count += 1
        print(f"[INFO] Menulis frame {frame_count}")

    cap.release()
    out.release()
    print("[INFO] Video selesai disimpan:", output_path)
    return total_vehicle_count



# def process_video(input_path, output_path, model, batch_size=8):
#     import numpy as np
#     cap = cv2.VideoCapture(input_path)
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#     frames = []
#     original_frames = []
#     total_vehicle_count = 0

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_resized = cv2.resize(frame, (640, 640))

#         frames.append(frame_resized)
#         original_frames.append(frame)

#         # Kalau frames sudah sebanyak batch_size ➔ deteksi sekaligus
#         if len(frames) == batch_size:
#             results = model(frames, conf=0.5, imgsz=640, verbose=False)

#             for i, preds in enumerate(results):
#                 for pred in preds.boxes:
#                     x1, y1, x2, y2 = map(int, pred.xyxy[0].cpu().numpy())
#                     class_id = int(pred.cls.cpu().numpy()[0])
#                     label = model.names[class_id]
#                     confidence = float(pred.conf.cpu().numpy()[0])

#                     x_scale = width / 640
#                     y_scale = height / 640
#                     x1 = int(x1 * x_scale)
#                     x2 = int(x2 * x_scale)
#                     y1 = int(y1 * y_scale)
#                     y2 = int(y2 * y_scale)

#                     cv2.rectangle(original_frames[i], (x1, y1), (x2, y2), (0, 255, 0), 2)
#                     cv2.putText(original_frames[i], f"{label} {confidence:.2f}", (x1, y1-10),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

#                     total_vehicle_count += 1

#                 out.write(original_frames[i])

#             frames = []
#             original_frames = []

#     # Kalau masih ada sisa frame yang belum diproses
#     if frames:
#         results = model(np.stack(frames), conf=0.5, imgsz=640, verbose=False)

#         for i, preds in enumerate(results):
#             for pred in preds.boxes:
#                 x1, y1, x2, y2 = map(int, pred.xyxy[0].cpu().numpy())
#                 class_id = int(pred.cls.cpu().numpy()[0])
#                 label = model.names[class_id]
#                 confidence = float(pred.conf.cpu().numpy()[0])

#                 x_scale = width / 640
#                 y_scale = height / 640
#                 x1 = int(x1 * x_scale)
#                 x2 = int(x2 * x_scale)
#                 y1 = int(y1 * y_scale)
#                 y2 = int(y2 * y_scale)

#                 cv2.rectangle(original_frames[i], (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(original_frames[i], f"{label} {confidence:.2f}", (x1, y1-10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

#                 total_vehicle_count += 1

#             out.write(original_frames[i])

#     cap.release()
#     out.release()

#     return total_vehicle_count



@app.route('/history')
def history():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM detections")
    records = cursor.fetchall()
    conn.close()

    grouped = {}
    for record in records:
        filename = record[1]
        if filename not in grouped:
            grouped[filename] = []
        grouped[filename].append({
            'id': record[0],
            'label': record[2],
            'confidence': record[3],
            'waktu': record[4]
        })

    return render_template('history.html', grouped=grouped)

@app.route('/', methods=['GET', 'POST'])
def index():
    ext = None
    total_detected = None
    result_image = None
    result_video = None
    labels_detected = []
    total_images = 0
    label_counts = {}

    # Ambil statistik data
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS detections (id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT, label TEXT, confidence REAL, timestamp TEXT)")

    cursor.execute("SELECT COUNT(DISTINCT filename) FROM detections")
    total_images = cursor.fetchone()[0]

    cursor.execute("SELECT label, COUNT(*) FROM detections GROUP BY label")
    label_data = cursor.fetchall()
    label_counts = {label: count for label, count in label_data}

    conn.close()

    detection_details = []

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
                # Proses gambar
                results = model(filepath)
                img = cv2.imread(filepath)

                conn = sqlite3.connect('database.db')
                cursor = conn.cursor()

                for pred in results[0].boxes:
                    x1, y1, x2, y2 = map(int, pred.xyxy[0].cpu().numpy())
                    class_id = int(pred.cls.cpu().numpy())
                    label = model.names[class_id]
                    confidence = float(pred.conf.cpu().numpy())

                    labels_detected.append(label)
                    detection_details.append({
                        'label': label,
                        'confidence': f"{round(confidence * 100)}%"
                    })

                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f"{label} {confidence:.2f}", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    cursor.execute("INSERT INTO detections (filename, label, confidence, timestamp) VALUES (?, ?, ?, ?)",
                                   (filename, label, confidence, timestamp))

                conn.commit()
                conn.close()

                result_path = os.path.join(app.config['RESULT_FOLDER'], 'result_' + filename)
                cv2.imwrite(result_path, img)
                result_image = 'result_' + filename

            elif ext in ['.mp4', '.avi']:
                # Proses video
                result_path = os.path.join(app.config['RESULT_FOLDER'], 'result_' + os.path.splitext(filename)[0] + '.mp4')
                # result_path = os.path.join(app.config['RESULT_FOLDER'], 'result_' + filename)
                total_detected = process_video(filepath, result_path, model)  #simpan hasil deteksi
                result_video = 'result_' + os.path.splitext(filename)[0] + '.mp4'


    return render_template("index.html", 
        result_image=result_image,
        result_video=result_video,
        labels_detected=labels_detected,
        detection_details=detection_details,
        total_detected=total_detected if ext in ['.mp4', '.avi'] else None,
        total_images=total_images,
        label_counts=label_counts or {})  # <- pastikan ini dictionary


if __name__ == '__main__':
    app.run(debug=True, port=5001)
