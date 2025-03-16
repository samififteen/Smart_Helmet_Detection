from flask import Flask, request, render_template, send_file
import cv2
import math
import cvzone
import os
from ultralytics import YOLO

app = Flask(__name__)

yolo_model = YOLO("Weights/best.pt")

class_labels = ['With Helmet', 'Without Helmet']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']
    
    if file.filename == '':
        return "No selected file"

    return process_image(file)

def process_image(image_file):
    img_path = os.path.join('uploads', image_file.filename)
    image_file.save(img_path)

    img = cv2.imread(img_path)
    results = yolo_model(img)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

            if conf > 0.1:
                cvzone.putTextRect(img, f'{class_labels[cls]} {conf}', (x1, y1 - 10), scale=0.8, thickness=1, colorR=(255, 0, 0))

    output_path = os.path.join('uploads', 'output_' + image_file.filename)
    cv2.imwrite(output_path, img)
    return send_file(output_path, mimetype='image/jpeg')

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)