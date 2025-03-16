from flask import Flask, request, render_template, send_file
import cv2
import math
import cvzone
from ultralytics import YOLO
import os

app = Flask(__name__)

# Load YOLO model
yolo_model = YOLO("Weights/best.pt")

# Class labels for helmet detection
class_labels = ['With Helmet', 'Without Helmet']

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Check if file is present in the request
    if 'video' not in request.files:
        return "No file part"
    
    file = request.files['video']
    
    # Check if the file has a valid filename
    if file.filename == '':
        return "No selected file"
    
    # Create uploads folder if it doesn't exist
    os.makedirs('uploads', exist_ok=True)

    # Save the uploaded file
    video_path = os.path.join('uploads', file.filename)
    file.save(video_path)

    # Output file path
    output_path = os.path.join('uploads', 'output_' + file.filename)

    # Process the video
    try:
        process_video(video_path, output_path)
        print(f"Video successfully processed: {output_path}")
        return send_file(output_path, mimetype='video/mp4')
    except Exception as e:
        print(f"Error during video processing: {str(e)}")
        return "An error occurred during video processing."

def process_video(input_path, output_path):
    # Open video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise Exception("Could not open input video file.")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video properties - FPS: {fps}, Width: {width}, Height: {height}")

    # Initialize VideoWriter for output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_skip = 3  # Process every third frame
    frame_count = 0
    processed_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or failed to read frame.")
            break

        # Only process every third frame
        if frame_count % frame_skip == 0:
            frame = cv2.resize(frame, (640, 640))
            results = yolo_model(frame)

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Draw bounding box and labels
                    w, h = x2 - x1, y2 - y1
                    cvzone.cornerRect(frame, (x1, y1, w, h))
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    if conf > 0.1:
                        cvzone.putTextRect(frame, f'{class_labels[cls]} {conf}', 
                                           (max(0, x1), max(35, y1)), scale=1, thickness=1)

            out.write(frame)
            processed_frames += 1

        frame_count += 1

    # Release resources
    cap.release()
    out.release()

    print(f"Processed frames: {processed_frames}. Video saved to: {output_path}")

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)