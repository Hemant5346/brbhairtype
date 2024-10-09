import streamlit as st
import cv2
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
from transformers import pipeline
import base64
from io import BytesIO

# Set page config
st.set_page_config(page_title="BRB Hair Type Classifier", layout="wide", initial_sidebar_state="collapsed")

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

# Functions for face detection and image classification
def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return len(faces) > 0

def classify_image(image):
    image = Image.fromarray(image).convert("RGB")
    
    pillow_mask = segmentation_pipeline(image, return_mask=True)  
    segmented_image = segmentation_pipeline(image) 
    
    image = np.array(segmented_image)
    image = Image.fromarray(image).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    predicted_class = class_names[predicted.item()]
    return segmented_image, predicted_class

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
    
    body {
        font-family: 'Poppins', sans-serif;
        background-color: #f0f4f8;
        color: #333;
    }
    .main {
        padding: 1rem;
    }
    h1, h2, h3 {
        color: #1a5f7a;
    }
    .stButton > button {
        background-color: #1a5f7a;
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 30px;
        padding: 0.6rem 1.5rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #2e8bc0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .webcam-container {
        border: 2px solid #1a5f7a;
        border-radius: 10px;
        overflow: hidden;
        max-width: 400px;
        margin: 0 auto;
    }
    .result-container {
        background-color: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-top: 1rem;
    }
    .result-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1a5f7a;
        margin-bottom: 0.5rem;
    }
    .result-text {
        font-size: 1rem;
        color: #34495e;
    }
    .hair-type {
        font-size: 1.2rem;
        font-weight: 600;
        color: #2e8bc0;
        background-color: #e6f3f8;
        padding: 0.5rem;
        border-radius: 5px;
        display: inline-block;
        margin-top: 0.5rem;
    }
    .app-description {
        background-color: #e6f3f8;
        border-radius: 10px;
        padding: 0.5rem;
        margin-bottom: 1rem;
        color: #1a5f7a;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Header and Description
st.markdown("<h1 style='text-align: center; color: #1a5f7a; font-size: 2rem;'>BRB Hair Classifier</h1>", unsafe_allow_html=True)
st.markdown("""
<div class="app-description">
    <p>Discover your unique hair type with our AI-powered classifier. Capture a selfie for personalized styling tips!</p>
</div>
""", unsafe_allow_html=True)

# Create a column layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("<h3 style='text-align: center; font-size: 1.2rem;'>Capture Your Style</h3>", unsafe_allow_html=True)
    st.markdown('<div class="webcam-container">', unsafe_allow_html=True)
    image_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)
    capture_button = st.button("📸 Capture & Analyze")

with col2:
    st.markdown("<h3 style='text-align: center; font-size: 1.2rem;'>Your Hair Analysis</h3>", unsafe_allow_html=True)
    result_placeholder = st.empty()

# Initialize the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("Unable to access the webcam. Please check your connection.")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image from webcam.")
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small_frame = cv2.resize(frame_rgb, (0, 0), fx=0.5, fy=0.5)
        image_placeholder.image(small_frame, channels="RGB", use_column_width=True)
        
        if capture_button:
            if detect_face(frame_rgb):
                background_removed_img, predicted_class = classify_image(frame_rgb)
                
                result_content = f"""
                <div class="result-container">
                    <img src="data:image/png;base64,{image_to_base64(background_removed_img)}" style="width:100%; border-radius:10px; margin-bottom:10px;">
                    <p class="result-header">Hair Analysis Results</p>
                    <p class="result-text">Your Hair Type:</p>
                    <p class="hair-type">{predicted_class}</p>
                    <p class="result-text">Our stylists recommend personalized treatments for your hair type. Book a consultation for expert advice!</p>
                </div>
                """
                result_placeholder.markdown(result_content, unsafe_allow_html=True)
            else:
                result_placeholder.warning("No face detected. Please ensure your face is clearly visible and try again.")
            break
        
        cv2.waitKey(1)

    cap.release()

# Call-to-action section
st.markdown("""
<div style="text-align: center; margin-top: 1rem; padding: 1rem; background-color: #1a5f7a; border-radius: 10px; color: white;">
    <h3 style="font-size: 1.2rem;">Ready for a Style Revolution?</h3>
    <p style="font-size: 0.9rem;">Book an appointment with our expert stylists!</p>
    <a href="#" style="background-color: white; color: #1a5f7a; padding: 8px 16px; text-decoration: none; border-radius: 30px; display: inline-block; margin-top: 10px; font-weight: 600; font-size: 0.9rem;">Book Now</a>
</div>
""", unsafe_allow_html=True)