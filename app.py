import torch
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from flask import Flask, request, render_template
from PIL import Image
import os

app = Flask(__name__)

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
weights = ResNet18_Weights.DEFAULT
num_classes = 9
model = resnet18(weights=weights)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)
model.load_state_dict(torch.load("best_skin_disease_model.pth", map_location=device))
model.eval()

# Image Transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Class labels
classes = [
    'Actinic keratosis', 'Atopic Dermatitis', 'Benign keratosis',
    'Dermatofibroma', 'Melanocytic nevus', 'Melanoma',
    'Squamous cell carcinoma', 'Tinea Ringworm Candidiasis', 'Vascular lesion'
]


def is_skin_image(image):
    """
    Basic check to ensure the uploaded image likely contains skin region.
    """
    image = image.convert("RGB")
    pixels = list(image.getdata())

    skin_pixels = 0
    for r, g, b in pixels[::50]:  # sample every 50th pixel
        if (r > 80 and g > 40 and b > 20 and
            max(r, g, b) - min(r, g, b) > 10 and
            r > g and r > b):
            skin_pixels += 1

    # If less than 10% skin-like pixels → not valid skin image
    return (skin_pixels / (len(pixels) / 50)) > 0.1


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    if request.method == "POST":
        if "image" not in request.files:
            error = "No file uploaded."
            return render_template("index.html", error=error)

        file = request.files["image"]

        if file.filename == "":
            error = "Please select an image file."
            return render_template("index.html", error=error)

        filepath = os.path.join("uploads", file.filename)
        os.makedirs("uploads", exist_ok=True)
        file.save(filepath)

        try:
            image = Image.open(filepath)

            # ✅ Check if valid skin image
            if not is_skin_image(image):
                # Instead of prediction, send message directly
                prediction = "❌ This is not a valid skin disease image. Please upload a proper skin image."
                return render_template("index.html", prediction=prediction)

            # Predict
            img_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(img_tensor)
                predicted_class = output.argmax(dim=1).item()
                prediction = f"🧠 Predicted Disease: {classes[predicted_class]}"

        except Exception as e:
            error = f"Error processing image: {e}"

    return render_template("index.html", prediction=prediction, error=error)


if __name__ == "__main__":
    app.run(debug=True)
 