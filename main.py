from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
from transformers import pipeline
import io

app = FastAPI()

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(pretrained=False)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 7)
model.load_state_dict(torch.load('best_hair_classification_model22.pth', map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class_names = [
    "Coily hair", "Curly hair", "Long Hair", "Medium Hair",
    "Short hair", "Straight Hair", "Wavy Hair"
]

segmentation_pipeline = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return len(faces) > 0

def classify_image(image):
    image = Image.fromarray(image).convert("RGB")
    
    segmented_image = segmentation_pipeline(image)[0]['image']
    
    image = np.array(segmented_image)
    image = Image.fromarray(image).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    predicted_class = class_names[predicted.item()]
    return predicted_class

@app.post("/classify_hair")
async def classify_hair(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    if not detect_face(img):
        return JSONResponse(content={"error": "No face detected in the image"}, status_code=400)
    
    predicted_class = classify_image(img)
    
    return {"hair_type": predicted_class}

@app.get("/")
async def root():
    return {"message": "Welcome to the BRB Hair Classifier API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)