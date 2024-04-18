from flask import Flask, render_template, request
from ultralytics import YOLO
import web_cam
import videos
import images

app = Flask(__name__ ,static_url_path='/static')
# Load your pre-trained YOLOv8 model
model_path = "best.pt"
model = YOLO(model_path)

# Define class names (modify according to your classes)
class_names = ["default", "Fire", "smoke"]

# Initialize webcam
# cap = cv2.VideoCapture(0)  # Use 0 for the default camera (usually built-in webcam)


@app.route('/')
def index():
    return render_template('front.html')


@app.route('/webcam', methods=['POST'])
def detect_using_web_cam():
    web_cam.webcam()


@app.route('/video', methods=['POST'])
def detect_in_a_video():
    if 'video' not in request.files:
        return "No file uploaded"
    video_file = request.files['video']
    # Save video file to local storage
    video_file.save('uploads/' + video_file.filename)

    videos.video_detect('uploads/' + video_file.filename)


@app.route('/image', methods=['POST'])
def detect_in_a_image():
    if 'image' not in request.files:
        return "No file uploaded"
    image_file = request.files['image']
    # Save image file to local storage
    image_file.save('uploads/' + image_file.filename)
    # Call your detection function with the saved image file
    # detection_result = detect_in_image('uploads/' + image_file.filename)
    images.image_detect('uploads/' + image_file.filename)


if __name__ == '__main__':
    app.run(debug=True)
