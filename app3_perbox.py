from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
import pandas as pd
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model_path = 'model/best.pt'
model = YOLO(model_path)

# List to store log data
log_data = []

def predict_frame(frame, frame_index):
    resized_frame = cv2.resize(frame, (640, 384))
    results = model(resized_frame)
    
    print(f"Processing frame {frame_index}")
    
    for result in results:
        # Print results for debugging
        print("Detection Results:", result)
        detections = result.boxes
        
        for detection in detections:
            print("Detection Object:", detection)
            
            # Ensure that detection.xyxy has the expected shape
            if detection.xyxy.shape[0] > 0:
                bbox = detection.xyxy[0]  # Assuming detection.xyxy is (N, 4)
                x1, y1, x2, y2 = bbox
                
                # Access class ID and confidence
                class_id = int(detection.cls)  # Ensure this index is valid
                confidence = float(detection.conf)
                
                # Retrieve class name from model's class names
                class_name = model.names[class_id] if class_id < len(model.names) else 'Unknown'

                log_entry = {
                    'Frame Index': frame_index,
                    'Class Name': class_name,
                    'Confidence': confidence,
                    'X1': x1,
                    'Y1': y1,
                    'X2': x2,
                    'Y2': y2,
                    'Inference Time (ms)': result.speed['inference']
                }
                log_data.append(log_entry)
            else:
                print(f"No bounding boxes found for detection in frame {frame_index}")

    return results

def annotate_frame(results, frame):
    annotated_frame = frame.copy()
    for result in results:
        annotated_frame = result.plot()
    return annotated_frame

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_path = video_path.replace('.mp4', '_annotated.avi')
    out = cv2.VideoWriter(out_path, fourcc, 30.0, (640, 384))

    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = predict_frame(frame, frame_index)
        annotated_frame = annotate_frame(results, frame)
        out.write(annotated_frame)
        
        frame_index += 1

    cap.release()
    out.release()

def save_logs_to_excel():
    df = pd.DataFrame(log_data)
    excel_path = os.path.join(app.config['UPLOAD_FOLDER'], 'detection_logs.xlsx')
    
    # Check if file exists and remove it
    if os.path.exists(excel_path):
        os.remove(excel_path)
    
    # Save the DataFrame to an Excel file
    df.to_excel(excel_path, index=False)
    # return excel_path
    return 'detection_logs.xlsx'


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            print('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            print('No selected file')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            if filename.lower().endswith(('.mp4', '.avi', '.mov')):
                process_video(filepath)
                annotated_filename = filename.replace('.mp4', '_annotated.avi')
                excel_file = save_logs_to_excel()

                return render_template('result.html', filename=annotated_filename, excel_file=excel_file)
            else:
                results = predict_frame(cv2.imread(filepath), 0)
                annotated_image = annotate_frame(results, cv2.imread(filepath))
                annotated_filename = 'annotated_' + filename
                Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)).save(os.path.join(app.config['UPLOAD_FOLDER'], annotated_filename))
                excel_file = save_logs_to_excel()
                print(f"EXCEL : {excel_file}") # debugging
                print(f"IMG : {annotated_filename}") # debugging
                return render_template('result.html', filename=annotated_filename, excel_file=excel_file)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
