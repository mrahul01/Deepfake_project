import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image
import tempfile
import os

from config import CONFIG
from utils.model_utils import load_model, predict_frame

# --- PAGE CONFIG ---
st.set_page_config(page_title="Deepfake Detector Pro", layout="wide", page_icon="🧠")

# Custom CSS for the Gallery
st.markdown("""
    <style>
    .stImage { border-radius: 10px; border: 2px solid #31333F; }
    </style>
    """, unsafe_allow_html=True)

st.title("🧠 Deepfake Detection System")
st.caption("AI-Powered Siamese Network for Real vs Fake Analysis")

# --- 1. MODEL LOADING ---
@st.cache_resource
def get_model():
    # Ensure the model file exists in your GitHub 'model/' folder
    return load_model("model/siamese_deepfake.pth", CONFIG)

model = get_model()

# --- 2. REFERENCE EMBEDDINGS ---
# In a Siamese network, we compare the input to a known Real and Fake vector.
# Replace these randn calls with your actual pre-saved tensors for accuracy.
ref_real = torch.randn(1, CONFIG["emb"])
ref_fake = torch.randn(1, CONFIG["emb"])

# --- 3. SIDEBAR ---
st.sidebar.header("Settings")
mode = st.sidebar.radio("Input Source", ["Image", "Video"])
process_rate = st.sidebar.slider("Analysis Frequency (Frames)", 5, 60, 15, 
                                help="Higher means faster playback, lower means more frequent AI checks.")

if st.sidebar.button("Clear Detection Gallery"):
    st.session_state.detection_gallery = []
    st.rerun()

# --- 4. GALLERY INITIALIZATION ---
if "detection_gallery" not in st.session_state:
    st.session_state.detection_gallery = []

# --- 5. IMAGE MODE ---
if mode == "Image":
    uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded:
        image = Image.open(uploaded)
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Process
        frame = np.array(image.convert('RGB'))
        label, dist = predict_frame(frame, model, ref_real, ref_fake, CONFIG)
        
        with col2:
            st.subheader("Analysis Result")
            if label == "REAL":
                st.success(f"Prediction: {label}")
            else:
                st.error(f"Prediction: {label}")
            st.metric("Similarity Distance", f"{dist:.4f}")

# --- 6. VIDEO MODE ---
elif mode == "Video":
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if uploaded_video:
        # Save uploaded video to a temporary file for OpenCV
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        
        cap = cv2.VideoCapture(tfile.name)
        st.subheader("Live Analysis Feed")
        frame_window = st.image([]) 
        
        # Initialize loop variables to prevent NameError
        count = 0 
        label = "Initializing..."
        dist = 0.0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            count += 1
            # Run AI prediction every X frames
            if count % process_rate == 0:
                label, dist = predict_frame(frame, model, ref_real, ref_fake, CONFIG)
                
                # Add to Gallery (convert BGR to RGB for Streamlit)
                thumb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                thumb = cv2.resize(thumb, (240, 180))
                st.session_state.detection_gallery.append((thumb, label, dist))

                # Keep gallery size manageable
                if len(st.session_state.detection_gallery) > 12:
                    st.session_state.detection_gallery.pop(0)

            # UI Visuals
            color = (0, 255, 0) if label == "REAL" else (0, 0, 255)
            if label == "Initializing...": color = (255, 255, 255)
            
            # Draw label on the frame for the live view
            cv2.putText(frame, f"STATUS: {label}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Display the processed frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_window.image(frame_rgb)

        cap.release()
        os.unlink(tfile.name) # Clean up the temp file

        # --- 7. THE GALLERY DISPLAY ---
        if st.session_state.detection_gallery:
            st.divider()
            st.subheader("📸 Detection History")
            st.info("The latest frames analyzed by the AI are shown below.")
            
            cols = st.columns(4)
            # Show gallery in reverse (newest first)
            for idx, (img, lbl, d) in enumerate(reversed(st.session_state.detection_gallery)):
                with cols[idx % 4]:
                    st.image(img, caption=f"{lbl} | Dist: {d:.3f}")
