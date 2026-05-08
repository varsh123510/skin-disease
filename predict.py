import torch
import numpy as np
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

# ---------------- DEVICE ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- MODEL ----------------
weights = ResNet18_Weights.DEFAULT
num_classes = 9
model = resnet18(weights=weights)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

model.load_state_dict(torch.load("best_skin_disease_model.pth", map_location=device))
model.eval()

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------------- SKIN CHECK ----------------
def is_skin_image(image_pil):
    img = image_pil.resize((224, 224))
    img_array = np.array(img)

    R = img_array[:, :, 0]
    G = img_array[:, :, 1]
    B = img_array[:, :, 2]

    # Basic skin tone detection rule
    skin_pixels = (
        (R > 60) &
        (G > 40) &
        (B > 20) &
        (R > B)
    )

    skin_ratio = np.sum(skin_pixels) / (224 * 224)

    return skin_ratio > 0.20


# ---------------- MELASMA / PIH CHECK ----------------
def detect_melasma_pih(image_pil):
    img = image_pil.resize((224, 224))
    img_array = np.array(img)

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


# ---------------- LOAD IMAGE ----------------
image_path = r"C:\Users\TOSHIBA\OneDrive\Desktop\skin disease\uploads\test.jpg"  
image = Image.open(image_path).convert("RGB")

# ---------------- FINAL DECISION ----------------
if not is_skin_image(image):
    predicted_label = "Not a detectable skin condition"

elif detect_melasma_pih(image):
    predicted_label = "Melasma / Post-Inflammatory Hyperpigmentation"

else:
    image_tensor = transform(image).unsqueeze(0).to(device)
    output = model(image_tensor)
    predicted_class = output.argmax(dim=1).item()

    classes = [
        'Actinic keratosis',
        'Atopic Dermatitis',
        'Benign keratosis',
        'Dermatofibroma',
        'Melanocytic nevus',
        'Melanoma',
        'Squamous cell carcinoma',
        'Tinea Ringworm Candidiasis',
        'Vascular lesion'
    ]

    predicted_label = classes[predicted_class]

print("Predicted class:", predicted_label)