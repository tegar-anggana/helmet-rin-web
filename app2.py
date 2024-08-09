from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model_path = 'model/best.pt'  # Ganti dengan nama file model Anda
model = YOLO(model_path)

def predict_frame(frame):
    resized_frame = cv2.resize(frame, (640, 384))
    results = model(resized_frame)
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
    out = cv2.VideoWriter(out_path, fourcc, 30.0, (640, 384))  # Adjust frame size as needed

    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        
        results = predict_frame(frame)
        annotated_frame = annotate_frame(results, frame)
        out.write(annotated_frame)

    cap.release()
    out.release()

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
            else:
                results = predict_frame(cv2.imread(filepath))
                annotated_image = annotate_frame(results, cv2.imread(filepath))
                annotated_filename = 'annotated_' + filename
                Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)).save(os.path.join(app.config['UPLOAD_FOLDER'], annotated_filename))
            
            return render_template('result.html', filename=annotated_filename)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
