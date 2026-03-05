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


elif mode=="Video":

    uploaded_video = st.file_uploader("Upload Video",type=["mp4","avi"])

    if uploaded_video:

        tfile = tempfile.NamedTemporaryFile(delete=False)

        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)

        frame_window = st.image([])

        while cap.isOpened():

            ret,frame = cap.read()

            if not ret:
                break

            label,dist = predict_frame(frame,model,ref_real,ref_fake,CONFIG)

            color = (0,255,0) if label=="REAL" else (0,0,255)

            cv2.putText(frame,f"{label}",(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,color,2)

            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            frame_window.image(frame)

        cap.release()
