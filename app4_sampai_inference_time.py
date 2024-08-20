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
import psutil  # Untuk CPU dan RAM usage
import GPUtil  # Tambahkan pustaka GPUtil untuk GPU usage

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model_path = 'model/best.pt'
model = YOLO(model_path)

# List to store log data and minute-based aggregation
log_data = []
minute_data = []

def predict_frame(frame, frame_index, timestamp, threshold):
    start_time = time.time()  # Start timing
    start_cpu = psutil.cpu_percent(interval=None)  # Start CPU usage measurement
    start_memory = psutil.virtual_memory().used / 1024 / 1024  # Start RAM usage measurement (in MB)
    
    # Measure GPU usage at the start
    start_gpu = GPUtil.getGPUs()[0].memoryUsed if GPUtil.getGPUs() else 0

    resized_frame = cv2.resize(frame, (640, 384))
    results = model(resized_frame, conf=threshold)  # Apply threshold during prediction
    
    end_cpu = psutil.cpu_percent(interval=None)  # End CPU usage measurement
    end_memory = psutil.virtual_memory().used / 1024 / 1024  # End RAM usage measurement (in MB)
    
    # Measure GPU usage at the end
    end_gpu = GPUtil.getGPUs()[0].memoryUsed if GPUtil.getGPUs() else 0

    cpu_usage = end_cpu - start_cpu
    memory_usage = end_memory - start_memory
    gpu_usage = end_gpu - start_gpu
    
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
                    'Inference Time (ms)': result.speed['inference'],
                    'CPU Usage (%)': cpu_usage,
                    'RAM Usage (MB)': memory_usage,
                    'GPU Usage (MB)': gpu_usage  # Menambahkan GPU usage pada log
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
    return results, inference_time, cpu_usage, memory_usage, gpu_usage

def annotate_frame(results, frame):
    annotated_frame = frame.copy()
    for result in results:
        annotated_frame = result.plot()
    return annotated_frame

def process_video(video_path, threshold):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_path = video_path.replace('.mp4', '_annotated.avi')
    out = cv2.VideoWriter(out_path, fourcc, 30.0, (640, 384))

    frame_index = 0
    total_inference_time = 0
    total_cpu_usage = 0
    total_memory_usage = 0
    total_gpu_usage = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # get timestamp in seconds
        results, inference_time, cpu_usage, memory_usage, gpu_usage = predict_frame(frame, frame_index, timestamp, threshold)
        annotated_frame = annotate_frame(results, frame)
        out.write(annotated_frame)
        
        total_inference_time += inference_time
        total_cpu_usage += cpu_usage
        total_memory_usage += memory_usage
        total_gpu_usage += gpu_usage
        frame_index += 1

    cap.release()
    out.release()

    # Calculate average CPU, memory, and GPU usage
    avg_cpu_usage = total_cpu_usage / frame_index
    avg_memory_usage = total_memory_usage / frame_index
    avg_gpu_usage = total_gpu_usage / frame_index

    return total_inference_time, avg_cpu_usage, avg_memory_usage, avg_gpu_usage

def aggregate_minute_data():
    df = pd.DataFrame(minute_data)
    aggregated_data = df.groupby(['Detected at minute', 'Label']).size().reset_index(name='Count detected')
    return aggregated_data        

def save_logs_to_excel(total_inference_time, avg_cpu_usage, avg_memory_usage, avg_gpu_usage):
    df_logs = pd.DataFrame(log_data)
    logs_path = os.path.join('./', app.config['UPLOAD_FOLDER'], 'detection_logs.xlsx')
    # absolute_path = '/static/uploads/detection_logs.xlsx'
    # print(logs_path)
    # try:
    #     # os.remove(logs_path)
    #     os.remove(absolute_path)
    #     print(f"File {logs_path} removed successfully.")
    # except FileNotFoundError:
    #     print(f"File {logs_path} does not exist.")
    # except PermissionError:
    #     print(f"Permission denied: Cannot remove file {logs_path}.")
    # except Exception as e:
    #     print(f"An error occurred: {e}")
    
    df_logs.to_excel(logs_path, index=False)
    
    if minute_data:
        aggregated_data = aggregate_minute_data()
        minute_data_path = os.path.join(app.config['UPLOAD_FOLDER'], 'minute_data_summary.xlsx')        
        aggregated_data.to_excel(minute_data_path, index=False)
    
    # try:
    #     os.remove(minute_data_path)
    #     print(f"File {minute_data_path} removed successfully.")
    # except FileNotFoundError:
    #     print(f"File {minute_data_path} does not exist.")
    # except PermissionError:
    #     print(f"Permission denied: Cannot remove file {minute_data_path}.")
    # except Exception as e:
    #     print(f"An error occurred: {e}")
    
    
    # Add total inference time and resource usage to the end of the Excel file
    with pd.ExcelWriter(logs_path, mode='a', engine='openpyxl') as writer:
        summary_df = pd.DataFrame({
            'Total Inference Time (ms)': [total_inference_time],
            'Average CPU Usage (%)': [avg_cpu_usage],
            'Average RAM Usage (MB)': [avg_memory_usage],
            'Average GPU Usage (MB)': [avg_gpu_usage]  # Menambahkan GPU usage pada ringkasan
        })
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    return 'detection_logs.xlsx', 'minute_data_summary.xlsx'

def delete_logs_manually(): 
    # DELETE DETECTION LOGS
    logs_path = os.path.join(app.config['UPLOAD_FOLDER'], 'detection_logs.xlsx')
    detection_absolute_path = '/static/uploads/detection_logs.xlsx'
    
    try:
        # os.remove(logs_path)
        os.remove(detection_absolute_path)
        print(f"File {logs_path} removed successfully.")
    except FileNotFoundError:
        print(f"File {logs_path} does not exist.")
    except PermissionError:
        print(f"Permission denied: Cannot remove file {logs_path}.")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    # DELETE MINUTE DATA SUMMARY
    minute_data_path = os.path.join(app.config['UPLOAD_FOLDER'], 'minute_data_summary.xlsx')
    minute_absolute_path = '/static/uploads/minute_data_summary.xlsx'
    
    try:
        # os.remove(minute_data_path)
        os.remove(minute_absolute_path)
        print(f"File {minute_data_path} removed successfully.")
    except FileNotFoundError:
        print(f"File {minute_data_path} does not exist.")
    except PermissionError:
        print(f"Permission denied: Cannot remove file {minute_data_path}.")
    except Exception as e:
        print(f"An error occurred: {e}")

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
            log_data.clear()
            minute_data.clear()
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            threshold = float(request.form['threshold'])
            if filename.lower().endswith(('.mp4', '.avi', '.mov')):
                delete_logs_manually()
                total_inference_time, avg_cpu_usage, avg_memory_usage, avg_gpu_usage = process_video(filepath, threshold)
                total_inference_time_sec = round(total_inference_time / 1000, 2)
                total_inference_time_min = round(total_inference_time_sec / 60, 2)
                annotated_filename = filename.replace('.mp4', '_annotated.avi')
                excel_logs, excel_minute_data = save_logs_to_excel(total_inference_time, avg_cpu_usage, avg_memory_usage, avg_gpu_usage)
                return render_template('result.html', filename=annotated_filename, excel_logs=excel_logs, excel_minute_data=excel_minute_data, inference_time=total_inference_time, inference_time_sec=total_inference_time_sec, inference_time_min=total_inference_time_min, cpu_usage=avg_cpu_usage, memory_usage=avg_memory_usage, gpu_usage=avg_gpu_usage)
            else:
                delete_logs_manually()
                results, inference_time, cpu_usage, memory_usage, gpu_usage = predict_frame(cv2.imread(filepath), 0, 0, threshold)
                total_inference_time = round(inference_time, 2)
                total_inference_time_sec = round(total_inference_time / 1000, 2)
                total_inference_time_min = round(total_inference_time_sec / 60, 2)
                annotated_image = annotate_frame(results, cv2.imread(filepath))
                annotated_filename = 'annotated_' + filename
                cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], annotated_filename), annotated_image)
                excel_logs, excel_minute_data = save_logs_to_excel(total_inference_time, cpu_usage, memory_usage, gpu_usage)
                return render_template('result.html', filename=annotated_filename, excel_logs=excel_logs, excel_minute_data=excel_minute_data, inference_time=total_inference_time, inference_time_sec=total_inference_time_sec, inference_time_min=total_inference_time_min, cpu_usage=cpu_usage, 
                memory_usage=memory_usage, 
                gpu_usage=gpu_usage)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)