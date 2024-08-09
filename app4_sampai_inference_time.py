from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
import pandas as pd
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import time

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model_path = 'model/best.pt'
model = YOLO(model_path)

# List to store log data and minute-based aggregation
log_data = []
minute_data = []

def predict_frame(frame, frame_index, timestamp):
    start_time = time.time()  # Start timing
    resized_frame = cv2.resize(frame, (640, 384))
    results = model(resized_frame)
    
    print(f"Processing frame {frame_index}")
    
    for result in results:
        detections = result.boxes
        
        for detection in detections:
            if detection.xyxy.shape[0] > 0:
                bbox = detection.xyxy[0]  # Assuming detection.xyxy is (N, 4)
                x1, y1, x2, y2 = bbox
                
                class_id = int(detection.cls)
                confidence = float(detection.conf)
                
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

                minute = int(timestamp // 60)  # assuming timestamp is in seconds
                label = class_name
                
                minute_entry = {
                    'Detected at minute': minute,
                    'Label': label,
                    'Count detected': 1
                }
                minute_data.append(minute_entry)
            else:
                print(f"No bounding boxes found for detection in frame {frame_index}")

    inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    return results, inference_time

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
    total_inference_time = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # get timestamp in seconds
        results, inference_time = predict_frame(frame, frame_index, timestamp)
        annotated_frame = annotate_frame(results, frame)
        out.write(annotated_frame)
        
        total_inference_time += inference_time
        frame_index += 1

    cap.release()
    out.release()

    return total_inference_time

def aggregate_minute_data():
    df = pd.DataFrame(minute_data)
    # if os.path.exists("complete_minute_data.xlsx"):
    #     os.remove("complete_minute_data.xlsx")
    # df.to_excel("complete_minute_data.xlsx", index=False)
    aggregated_data = df.groupby(['Detected at minute', 'Label']).size().reset_index(name='Count detected')
    return aggregated_data

def save_logs_to_excel(total_inference_time):
    df_logs = pd.DataFrame(log_data)
    logs_path = os.path.join(app.config['UPLOAD_FOLDER'], 'detection_logs.xlsx')
    
    try:
        os.remove(logs_path)
        print(f"File {logs_path} removed successfully.")
    except FileNotFoundError:
        print(f"File {logs_path} does not exist.")
    except PermissionError:
        print(f"Permission denied: Cannot remove file {logs_path}.")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    df_logs.to_excel(logs_path, index=False)
    
    aggregated_data = aggregate_minute_data()
    minute_data_path = os.path.join(app.config['UPLOAD_FOLDER'], 'minute_data_summary.xlsx')
    
    try:
        os.remove(minute_data_path)
        print(f"File {minute_data_path} removed successfully.")
    except FileNotFoundError:
        print(f"File {minute_data_path} does not exist.")
    except PermissionError:
        print(f"Permission denied: Cannot remove file {minute_data_path}.")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    aggregated_data.to_excel(minute_data_path, index=False)
    
    # Add total inference time to the end of the Excel file
    with pd.ExcelWriter(logs_path, mode='a', engine='openpyxl') as writer:
        pd.DataFrame({'Total Inference Time (ms)': [total_inference_time]}).to_excel(writer, sheet_name='Summary', index=False)
    
    return 'detection_logs.xlsx', 'minute_data_summary.xlsx'

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
                total_inference_time = round(process_video(filepath), 2)
                total_inference_time_sec = round(total_inference_time / 1000, 2)
                total_inference_time_min = round(total_inference_time_sec / 60, 2)
                annotated_filename = filename.replace('.mp4', '_annotated.avi')
                excel_logs, excel_minute_data = save_logs_to_excel(total_inference_time)
                print(f"VIDEO: {annotated_filename}") # debugging
                print(f"EXCEL LOGS: {excel_logs}") # debugging
                print(f"MINUTE DATA: {excel_minute_data}") # debugging
                return render_template('result.html', filename=annotated_filename, excel_logs=excel_logs, excel_minute_data=excel_minute_data, inference_time=total_inference_time, inference_time_sec=total_inference_time_sec, inference_time_min=total_inference_time_min)
            else:
                results, inference_time = predict_frame(cv2.imread(filepath), 0, 0)
                total_inference_time = round(inference_time, 2)
                total_inference_time_sec = round(total_inference_time / 1000, 2)
                total_inference_time_min = round(total_inference_time_sec / 60, 2)
                annotated_image = annotate_frame(results, cv2.imread(filepath))
                annotated_filename = 'annotated_' + filename
                Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)).save(os.path.join(app.config['UPLOAD_FOLDER'], annotated_filename))
                excel_logs, excel_minute_data = save_logs_to_excel(0)  # No inference time for images
                print(f"IMAGE: {annotated_filename}") # debugging
                print(f"EXCEL LOGS: {excel_logs}") # debugging
                print(f"MINUTE DATA: {excel_minute_data}") # debugging
                return render_template('result.html', filename=annotated_filename, excel_logs=excel_logs, excel_minute_data=excel_minute_data, inference_time=total_inference_time, inference_time_sec=total_inference_time_sec, inference_time_min=total_inference_time_min)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
