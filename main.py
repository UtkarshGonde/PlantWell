import os
import json
import base64
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

# Page configuration
st.set_page_config(page_title="PlantWell", page_icon="🌿")

# Initialize session state variables
if 'GOOGLE_API_KEY' not in st.session_state:
    st.session_state['GOOGLE_API_KEY'] = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_disease' not in st.session_state:
    st.session_state.current_disease = None

# Sidebar configuration
with st.sidebar:
    st.title("Configuration")
    api_key = st.text_input("Enter your Google API Key", type="password")
    if api_key:
        st.session_state['GOOGLE_API_KEY'] = api_key
        genai.configure(api_key=api_key)

# Helper Functions
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

def initialize_chat_model():
    if st.session_state['GOOGLE_API_KEY']:
        return genai.GenerativeModel('gemini-pro')
    return None

def get_chatbot_response(user_input, disease_context=None):
    if not st.session_state['GOOGLE_API_KEY']:
        return "Please enter your Google API key in the sidebar to use the chat feature."
    
    try:
        model = initialize_chat_model()
        context = ""
        if disease_context:
            context = f"Context: The plant has been diagnosed with {disease_context}. "
        
        prompt = f"""{context}As a plant pathologist, please respond to the following question 
        about plant disease treatment and care: {user_input}
        Provide specific, practical advice while being concise."""
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error getting response: {str(e)}"

def get_disease_cure(disease_name):
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


   # ... (previous imports and configurations remain the same)

def display_chat_interface():
    # Custom CSS for the chat interface
    st.markdown("""
        <style>
        # .chat-box {
        #     background-color: rgba(255, 255, 255, 0.95);
        #     border-radius: 15px;
        #     padding: 20px;
        #     margin: 20px 0;
        #     box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        #     height: 400px;
        #     display: flex;
        #     flex-direction: column;
        # }
        .chat-messages {
            flex-grow: 1;
            overflow-y: auto;
            padding-right: 10px;
            margin-bottom: 20px;
        }
        .user-message {
            background-color: #57EE12;
            color: #000000;
            padding: 10px 15px;
            border-radius: 15px;
            margin: 5px 0;
            margin-left: 20%;
            display: inline-block;
            max-width: 80%;
            float: right;
            clear: both;
        }
        .bot-message {
            background-color: #2C2C2C;
            color: #FFFFFF;
            padding: 10px 15px;
            border-radius: 15px;
            margin: 5px 0;
            margin-right: 20%;
            display: inline-block;
            max-width: 80%;
            float: left;
            clear: both;
        }
        .chat-input {
            background-color: white;
            border-radius: 10px;
            padding: 10px;
            margin-top: 10px;
        }
        /* Custom scrollbar */
        .chat-messages::-webkit-scrollbar {
            width: 6px;
        }
        .chat-messages::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 10px;
        }
        .chat-messages::-webkit-scrollbar-thumb {
            background: #888;
            border-radius: 10px;
        }
        .chat-messages::-webkit-scrollbar-thumb:hover {
            background: #555;
        }
        </style>
    """, unsafe_allow_html=True)

    # Chat container
    st.markdown('<div class="chat-box">', unsafe_allow_html=True)
    
    # Messages container
    st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-message">{message["content"]}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Input area
    st.markdown('<div class="chat-input">', unsafe_allow_html=True)
    col1, col2 = st.columns([4, 1])
    
    # Initialize the chat input key in session state if it doesn't exist
    if 'chat_input_key' not in st.session_state:
        st.session_state.chat_input_key = 0
    
    with col1:
        # Use a unique key for the text input
        user_input = st.text_input(
            "Ask about plant care and treatment...",
            key=f"chat_input_{st.session_state.chat_input_key}"
        )
    
    with col2:
        send_button = st.button("Send", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Handle the chat interaction
    if send_button and user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Get bot response
        with st.spinner("Getting response..."):
            bot_response = get_chatbot_response(user_input, st.session_state.current_disease)
        
        # Add bot response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
        
        # Increment the key to force a new input field
        st.session_state.chat_input_key += 1
        
        # Rerun the app to update the chat display
        st.rerun()

# ... (rest of the code remains the same)
# Main App
def main():
    # Set paths
    working_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"
    background_image_path = r"C:\Users\Utkarsh\Videos\movies\leaves.webp"

    # Load model and class indices
    plant_model = tf.keras.models.load_model(model_path)
    class_indices = json.load(open(f"{working_dir}/class_indices.json"))

    # Set background
    bg_image_base64 = get_base64_image(background_image_path)
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

    st.title('PlantWell - Plant Disease Detection & Treatment')

    if not st.session_state['GOOGLE_API_KEY']:
        st.warning("Please enter your Google API key in the sidebar to get treatment recommendations.")

    # Image upload and analysis
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
                    prediction = predict_image_class(plant_model, uploaded_image, class_indices)
                    st.session_state.current_disease = prediction
                    st.markdown(f'<div class="prediction-box">Detected Disease: {str(prediction)}</div>', unsafe_allow_html=True)
                    
                    with st.spinner('Getting treatment information...'):
                        cure_info = get_disease_cure(prediction)
                        st.markdown(f'<div class="cure-box"><h4>Treatment Recommendations:</h4>{cure_info}</div>', 
                                  unsafe_allow_html=True)

    # Chat interface
    st.markdown("### Chat with our Plant Care Assistant")
    st.markdown("Ask questions about plant diseases, treatments, and care recommendations.")
    display_chat_interface()

    # Footer
    st.markdown("""
    <div style='position: fixed; bottom: 0; width: 100%; background-color: rgba(0,0,0,0.7); padding: 10px; text-align: center;'>
        <p style='color: white; margin: 0;'>Powered by TensorFlow and Google Gemini</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()