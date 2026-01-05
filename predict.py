import torch
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Recreate the model
weights = ResNet18_Weights.DEFAULT
num_classes = 9  # replace with your number of classes
model = resnet18(weights=weights)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Load the saved weights
model.load_state_dict(torch.load("best_skin_disease_model.pth", map_location=device))
model.eval()  # set model to evaluation mode

# Define transforms (same as validation transforms)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load a new image
image_path = r"c:\Users\TOSHIBA\Downloads\ben.JPG"  # put your image path here
image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(device)  # add batch dimension

# Make prediction
output = model(image)
predicted_class = output.argmax(dim=1).item()

# Map predicted index to class name
classes = ['Actinic keratosis', 'Atopic Dermatitis', 'Benign keratosis',
           'Dermatofibroma', 'Melanocytic nevus', 'Melanoma',
           'Squamous cell carcinoma', 'Tinea Ringworm Candidiasis', 'Vascular lesion']

print("Predicted class:", classes[predicted_class])