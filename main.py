from flask import Flask, request, render_template, jsonify
import os
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as T
import base64
import io

app = Flask(__name__)

model = torch.load('best.pt', map_location=torch.device('cpu'))
model.eval()

def detect_and_classify_image(image_path):
    # Load and preprocess the image
    img = Image.open(image_path)
    transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)

    # Perform detection
    detection_result = model.detect(img_tensor)

    # Perform classification
    classification_result = model.classify(img_tensor)

    return img, detection_result, classification_result


@app.route('/')
def home():
    return render_template('TrafficSign.html')


@app.route('/predict', methods=['GET'])
def model():
    return render_template('Model.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image part"
    img = request.files['image']
    if img.filename == '':
        return "No selected image"

    img_path = 'uploads/' + img.filename
    img.save(img_path)

    img, detection_result, classification_result = detect_and_classify_image(
        img_path)

    # Process detection results and draw bounding boxes on the image
    img_with_boxes = img.copy()
    draw = ImageDraw.Draw(img_with_boxes)

    for box in detection_result:
        left, top, right, bottom, _ = box.tolist()
        draw.rectangle([left, top, right, bottom], outline="red")

    # Encode the image in base64 format
    buffered = io.BytesIO()
    img_with_boxes.save(buffered, format="JPEG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Update the results on the HTML page using JavaScript
    response_data = {
        "image": encoded_image,
        "classification": classification_result
    }

    return jsonify(response_data)


if __name__ == '__main__':
    app.run(debug=True)
