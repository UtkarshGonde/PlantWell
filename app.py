import os
import json
import base64
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

# Add these lines at the top of your Streamlit app
st.set_page_config(page_title="PlantWell", page_icon="🌿")

# Create a secret input for the API key if not in environment
if 'GOOGLE_API_KEY' not in st.session_state:
    st.session_state['GOOGLE_API_KEY'] = None

# Add API key input in the sidebar
with st.sidebar:
    st.title("Configuration")
    api_key = st.text_input("Enter your Google API Key", type="password")
    if api_key:
        st.session_state['GOOGLE_API_KEY'] = api_key
        # Configure Gemini API with the entered key
        genai.configure(api_key=api_key)

# Function to convert an image file to base64
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Set background image path
background_image_path = r"C:\Users\Utkarsh\Videos\movies\leaves.webp"

# Set working directory and model path
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"

# Load the pre-trained model
plant_model = tf.keras.models.load_model(model_path)

# Loading the class indices
class_indices = json.load(open(f"{working_dir}/class_indices.json"))

def get_disease_cure(disease_name):
    """Get cure recommendations from Gemini"""
    if not st.session_state['GOOGLE_API_KEY']:
        return "Please enter your Google API key in the sidebar to get treatment recommendations."
    
    prompt = f"""
    As a plant pathologist, provide detailed cure and treatment recommendations for the plant disease: {disease_name}
    Include:
    1. Common treatments
    2. Preventive measures
    3. Organic solutions if available
    Please be specific and concise.
    """
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error getting cure information: {str(e)}"

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Main App
st.title('PlantWell - Plant Disease Detection & Treatment')

# Encode the background image as base64
bg_image_base64 = get_base64_image(background_image_path)

# Add custom CSS for the background image and improved styling
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/jpeg;base64,{bg_image_base64});
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        height: 100vh;
        padding: 20px;
        color: white;
    }}
    .prediction-box {{
        background-color: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 10px;
        color: black;
        margin: 10px 0;
    }}
    .cure-box {{
        background-color: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 10px;
        color: black;
        margin: 10px 0;
        max-height: 300px;
        overflow-y: auto;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

if not st.session_state['GOOGLE_API_KEY']:
    st.warning("Please enter your Google API key in the sidebar to get treatment recommendations.")

uploaded_image = st.file_uploader("Upload an image of the diseased plant...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img, caption="Uploaded Image")

    with col2:
        if st.button('Analyze Disease'):
            with st.spinner('Analyzing the image...'):
                # Predict disease
                prediction = predict_image_class(plant_model, uploaded_image, class_indices)
                st.markdown(f'<div class="prediction-box">Detected Disease: {str(prediction)}</div>', unsafe_allow_html=True)
                
                # Get cure information
                with st.spinner('Getting treatment information...'):
                    cure_info = get_disease_cure(prediction)
                    st.markdown(f'<div class="cure-box"><h4>Treatment Recommendations:</h4>{cure_info}</div>', 
                              unsafe_allow_html=True)

# Add footer
st.markdown("""
<div style='position: fixed; bottom: 0; width: 100%; background-color: rgba(0,0,0,0.7); padding: 10px; text-align: center;'>
    <p style='color: white; margin: 0;'>Powered by TensorFlow and Google Gemini</p>
</div>
""", unsafe_allow_html=True)