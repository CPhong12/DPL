from flask import Flask, request, render_template
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import io

app = Flask(__name__)
model = torch.load("best.pt")

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

    img_path = 'test/' + img.filename
    img.save(img_path)

    img, detection_result, classification_result = detect_and_classify_image(img_path)

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
