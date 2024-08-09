from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import io
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model_path = 'model/best.pt'  # Ganti dengan nama file model Anda
model = YOLO(model_path)

def predict(img_path):
    img = cv2.imread(img_path)
    resized_img = cv2.resize(img, (640, 384))
    results = model(resized_img)
    return results

def annotate_image(results, img_path):
    img = cv2.imread(img_path)
    resized_img = cv2.resize(img, (640, 384))
    for result in results:
        annotated_frame = result.plot()
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    return annotated_frame_rgb

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
            results = predict(filepath)
            annotated_image = annotate_image(results, filepath)
            annotated_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'annotated_' + filename)
            Image.fromarray(annotated_image).save(annotated_image_path)
            return render_template('result.html', filename='annotated_' + filename)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
