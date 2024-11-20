import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import cv2

# Load the trained model
model = load_model('deepfake_detection_model.h5')

@st.cache_resource
def get_feature_extractor():
    return ResNet50(
        weights='imagenet',  # Or specify the local path to the weights
        include_top=False,
        pooling='avg'
    )

# Get feature extractor
feature_extractor = get_feature_extractor()

# Function to preprocess video and extract features
def preprocess_video_for_model(video_path, num_frames=20):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        frame = preprocess_input(frame)
        frames.append(frame)
    cap.release()

    frames = np.array(frames)
    features = feature_extractor.predict(frames)

    if len(features) > num_frames:
        indices = np.linspace(0, len(features) - 1, num_frames).astype(int)
        features = features[indices]
    elif len(features) < num_frames:
        padding = np.zeros((num_frames - len(features), 2048))
        features = np.vstack((features, padding))
    
    return features

# Streamlit app
st.title("Deepfake Video Detection")
st.write("Upload a video to check if it is a deepfake.")

# File upload
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
if uploaded_file is not None:
    st.video(uploaded_file)
    st.write("Processing video...")

    # Save uploaded video locally
    with open("uploaded_video.mp4", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Preprocess video and predict
    try:
        frames = preprocess_video_for_model("uploaded_video.mp4")  # Shape: (20, 2048)
        frames = np.expand_dims(frames, axis=0)  # Add batch dimension: (1, 20, 2048)

        # Dummy temporal input (if required by your model)
        temporal_input = np.ones((1, 20))  # Replace with actual temporal input if applicable

        # Predict
        predictions = model.predict([frames, temporal_input])
        avg_prediction = np.mean(predictions)

        # Output result
        if avg_prediction > 0.5:
            st.error("The video is a DEEPFAKE.")
        else:
            st.success("The video is REAL.")
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")



