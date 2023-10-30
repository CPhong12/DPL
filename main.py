from flask import Flask, request, render_template
import torch
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
import torchvision

app = Flask(__name__)
model = None

# Load your PyTorch model here
def load_model():
    global model
    model = torch.load("static/model.pt", map_location=torch.device('cpu'))  # Adjust 'cpu' to the appropriate device
load_model()

def predict_image(image):
    image_tensor = F.to_tensor(image).unsqueeze(0)  # Convert PIL image to a PyTorch tensor
    with torch.no_grad():
        prediction = model(image_tensor)
    return prediction

@app.route('/')
def home():
    return render_template('TrafficSign.html')

@app.route('/model')
def man():
    return render_template('Model.html')

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        if "image" not in request.files:
            return "No file part"
        image_file = request.files["image"]
        if image_file.filename == "":
            return "No selected file"

        if image_file:
            img = Image.open(image_file)
            prediction = predict_image(img)

            draw = ImageDraw.Draw(img)

            # Extract bounding box coordinates and labels from the prediction
            boxes = prediction[0]['boxes'].tolist()
            labels = prediction[0]['labels'].tolist()

            for box, label in zip(boxes, labels):
                box = [int(i) for i in box]  # Convert to integers
                draw.rectangle(box, outline="red", width=3)
                draw.text((box[0], box[1]), f"Label: {label}", fill="red")

            img.save("static/predicted_image.jpg")
            return render_template("TrafficSign.html", image_url="static/predicted_image.jpg")

if __name__ == '__main__':
    app.run(debug=True, port=8080)
