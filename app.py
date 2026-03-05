import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image
import tempfile

from config import CONFIG
from utils.model_utils import load_model,predict_frame

st.set_page_config(page_title="Deepfake Detector",layout="wide")

st.title("🧠 Deepfake Detection System")

# 1. Load the model
model = load_model("model/siamese_deepfake.pth", CONFIG)

# 2. YOU ARE MISSING THIS PART: 
# We need to define what "Real" and "Fake" look like in vector space.
# For now, let's create dummy tensors so the code doesn't crash, 
# but ideally, these should be pre-calculated from your training set.
ref_real = torch.randn(1, CONFIG["emb"]) 
ref_fake = torch.randn(1, CONFIG["emb"]) 

# 3. Sidebar UI
mode = st.sidebar.radio("Choose Input",["Image","Video"])

def process_image(image):

    frame = np.array(image)

    label,dist = predict_frame(frame,model,ref_real,ref_fake,CONFIG)

    return label,dist


if mode=="Image":

    uploaded = st.file_uploader("Upload Image",type=["jpg","png","jpeg"])

    if uploaded:

        image = Image.open(uploaded)

        st.image(image,use_column_width=True)

        frame = np.array(image)

        label,dist = predict_frame(frame,model,ref_real,ref_fake,CONFIG)

        if label=="REAL":

            st.success(f"Prediction: {label}")

        else:

            st.error(f"Prediction: {label}")

        st.write("Distance:",dist)


elif mode == "Video":
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi"])

    # --- GALLERY INITIALIZATION ---
    if "detection_gallery" not in st.session_state:
        st.session_state.detection_gallery = [] # Stores (image, label, distance)

    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)

        st.subheader("Live Analysis")
        frame_window = st.image([]) 
        
        # We only want to "capture" a gallery item every few frames 
        # so we don't spam the page with 1000 images.
        count = 0 

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            count += 1
            # Predict every 10th frame to save speed
            if count % 10 == 0:
                label, dist = predict_frame(frame, model, ref_real, ref_fake, CONFIG)
                
                # Add to gallery state
                # We convert to RGB and resize for the thumbnail
                thumb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                thumb = cv2.resize(thumb, (200, 200))
                st.session_state.detection_gallery.append((thumb, label, dist))

                # Keep only the last 12 detections to prevent memory crash
                if len(st.session_state.detection_gallery) > 12:
                    st.session_state.detection_gallery.pop(0)

            # Standard UI Feedback
            color = (0, 255, 0) if label == "REAL" else (0, 0, 255)
            cv2.putText(frame, f"{label}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            frame_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_window.image(frame_display)

        cap.release()

        # --- THE GALLERY DISPLAY ---
        st.divider()
        st.subheader("📸 Detection Gallery (Recent Frames)")
        
        # Create a grid of 4 columns
        cols = st.columns(4)
        for idx, (img, lbl, d) in enumerate(reversed(st.session_state.detection_gallery)):
            with cols[idx % 4]:
                st.image(img, caption=f"{lbl} (Dist: {d:.2f})")

        cap.release()
