from flask import Flask, request, render_template
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as transforms
import os

app = Flask(__name__)

# Set the upload folder for images
UPLOAD_FOLDER = 'uploaded_images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your object detection and classification model
# Replace with your specific model loading code
# detection_model = load_object_detection_model()
# classification_model = load_classification_model()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return "No image part"
        img = request.files['image']
        if img.filename == '':
            return "No selected image"
        if img:
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], img.filename)
            img.save(img_path)

            # Perform object detection and classification
            # Replace the following lines with code to use your models
            image = Image.open(img_path)

            # Example bounding boxes and labels for illustration
            # In practice, replace this with real object detection results
            detection_results = [
                [50, 50, 200, 200, 'right'],
                [100, 100, 250, 250, 'left']
            ]

            # Display the bounding boxes and labels on the image
            image_with_boxes = image.copy()
            draw = ImageDraw.Draw(image_with_boxes)

            for box in detection_results:
                left, top, right, bottom, label = box
                draw.rectangle([left, top, right, bottom], outline="red")
                draw.text((left, top), label, fill="red")

            return render_template('result.html', image_path=img_path, image_with_boxes=image_with_boxes)
    return render_template('Model.html')

if __name__ == '__main__':
    app.run(debug=True)