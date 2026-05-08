import torch
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from flask import Flask, request, render_template
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
weights = ResNet18_Weights.DEFAULT
num_classes = 9
model = resnet18(weights=weights)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("best_skin_disease_model.pth", map_location=device))
model = model.to(device)
model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Original 9 classes
classes = [
    "Actinic keratosis",
    "Atopic Dermatitis",
    "Benign keratosis",
    "Dermatofibroma",
    "Melanocytic nevus",
    "Melanoma",
    "Squamous cell carcinoma",
    "Tinea Ringworm Candidiasis",
    "Vascular lesion"
]

# Allergy mapping
allergy_classes = {
    "Atopic Dermatitis": "Eczema (Skin Allergy)",
    "Tinea Ringworm Candidiasis": "Fungal Allergy / Infection"
}

# ----------------------------
# Skin validation function
# ----------------------------
def is_skin_image(image):
    image = image.convert("RGB")
    pixels = list(image.getdata())

    skin_pixels = 0
    for r, g, b in pixels[::50]:
        if (r > 80 and g > 40 and b > 20 and
            max(r, g, b) - min(r, g, b) > 10 and
            r > g and r > b):
            skin_pixels += 1

    return (skin_pixels / (len(pixels) / 50)) > 0.1


# ----------------------------
# Melasma / PIH Detection Rule
# ----------------------------
def detect_melasma_pih(image):
    image = image.resize((224, 224))
    img_array = np.array(image)

    R = img_array[:, :, 0].astype(float)
    G = img_array[:, :, 1].astype(float)
    B = img_array[:, :, 2].astype(float)

    brown_pixels = (
        (R > 70) &
        (G > 40) &
        (B < 150) &
        (R > G) &
        (G > B)
    )

    brown_ratio = np.sum(brown_pixels) / (224 * 224)

    return brown_ratio > 0.30


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    if request.method == "POST":
        if "image" not in request.files:
            error = "No image uploaded."
            return render_template("index.html", error=error)

        file = request.files["image"]

        if file.filename == "":
            error = "No file selected."
            return render_template("index.html", error=error)

        os.makedirs("uploads", exist_ok=True)
        filepath = os.path.join("uploads", file.filename)
        file.save(filepath)

        try:
            image = Image.open(filepath).convert("RGB")

            # Step 1: Validate skin image
            if not is_skin_image(image):
                prediction = "❌ Please upload a valid skin image."
                return render_template("index.html", prediction=prediction)

            # Step 2: Check Pigmentation Disorder FIRST
            if detect_melasma_pih(image):
                prediction = (
                    "🌑 Pigmentation Disorder Detected\n"
                    "Type: Melasma / Post-Inflammatory Hyperpigmentation\n\n"
                    "⚠️ This is an AI-assisted estimation only."
                )
                return render_template("index.html", prediction=prediction)

            # Step 3: Otherwise run CNN model
            img_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(img_tensor)
                predicted_class = outputs.argmax(dim=1).item()

            predicted_disease = classes[predicted_class]

            if predicted_disease in allergy_classes:
                prediction = (
                    "🤧 Skin Allergy Detected\n"
                    f"Type: {allergy_classes[predicted_disease]}\n\n"
                    "⚠️ This is not a medical diagnosis."
                )
            else:
                prediction = (
                    "🧠 Skin Disease Detected\n"
                    f"Type: {predicted_disease}\n\n"
                    "⚠️ This is not a medical diagnosis."
                )

        except Exception as e:
            error = f"Error processing image: {e}"

    return render_template("index.html", prediction=prediction, error=error)


if __name__ == "__main__":
    app.run(debug=True)