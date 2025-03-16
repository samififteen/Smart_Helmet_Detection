from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from static.database.databases import get_db
import cv2
import math
import cvzone
from ultralytics import YOLO
import os
import logging

app = Flask("ItzSimplyJoe")
app.secret_key = 'superawesomesecretkey1010001'

yolo_model = YOLO("Weights/best.pt")
class_labels = ['With Helmet', 'Without Helmet']

logging.basicConfig(level=logging.INFO)

@app.route("/")
def home():
    return render_template("signup.html")

@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/signup")
def signup():
    return render_template("signup.html")

@app.route('/success')
def success():
    return render_template("success.html")

@app.route('/image_detection')
def image_detection():
    return render_template("image_detection.html")

@app.route('/video_detection')
def video_detection():
    return render_template("video_detection.html")

@app.route("/createaccount", methods=["POST"])
def create_account():
    username = request.form["username"]
    password = request.form["password"]
    db = get_db()
    others = db.fetch(username)
    if others:
        flash("Username already exists", "error")
        return redirect(url_for("signup"))
    if len(password) < 8:
        flash("Password must be at least 8 characters long", "error")
        return redirect(url_for("signup"))
    try:
        db.insert(username, password)
        flash("Account created successfully, please login", "success")
        return redirect(url_for("login"))
    except Exception as e:
        logging.error(f"Error creating account: {e}")
        flash("An error occurred", "error")
        return redirect(url_for("signup"))

@app.route("/login_account", methods=["POST"])
def login_account():
    username = request.form["username"]
    password = request.form["password"]
    db = get_db()
    user = db.login(username, password)
    if user:
        return redirect(url_for("success"))
    flash("Incorrect username or password", "error")
    return redirect(url_for("login"))

@app.route("/upload_image", methods=["POST"])
def upload_image():
    return handle_file_upload("image", "image_detection", process_image, "image/jpeg")

@app.route("/upload_video", methods=["POST"])
def upload_video():
    return handle_file_upload("video", "video_detection", process_video, "video/mp4")

def handle_file_upload(file_key, redirect_route, process_function, mimetype):
    if file_key not in request.files:
        flash("No file part", "error")
        return redirect(url_for(redirect_route))

    file = request.files[file_key]
    if file.filename == '':
        flash("No selected file", "error")
        return redirect(url_for(redirect_route))

    try:
        input_path = os.path.join('uploads', file.filename)
        output_path = os.path.join('uploads', f"output_{file.filename}")
        file.save(input_path)
        process_function(input_path, output_path)
        return send_file(output_path, mimetype=mimetype)
    except Exception as e:
        logging.error(f"Error processing file: {e}")
        flash("An error occurred while processing the file", "error")
        return redirect(url_for(redirect_route))

def process_image(input_path, output_path):
    image = cv2.imread(input_path)
    results = yolo_model(image)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(image, (x1, y1, w, h))
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            if conf > 0.1:
                cvzone.putTextRect(image, f'{class_labels[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

    cv2.imwrite(output_path, image)

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    frame_skip = 3
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 640))
        if frame_count % frame_skip == 0:
            results = yolo_model(frame)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1
                    cvzone.cornerRect(frame, (x1, y1, w, h))
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    if conf > 0.1:
                        cvzone.putTextRect(frame, f'{class_labels[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

            out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

if __name__ == "__main__":
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)