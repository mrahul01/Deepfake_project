import streamlit as st
import cv2
import torch
import numpy as np
import tempfile
from PIL import Image

from utils.model_utils import load_model
from utils.feature_utils import extract_features_from_frame


# --------------------------------
# Page Config
# --------------------------------

st.set_page_config(page_title="Deepfake Detector", layout="wide")

st.title("🧠 Deepfake Detection System")
st.markdown("Detect whether a face is **REAL or FAKE** using Siamese Deepfake Detection Model")


# --------------------------------
# Config
# --------------------------------

cfg = {
    "use_fcm": False,
    "fcm_k": 3,
    "hid": 128,
    "emb": 64
}


# --------------------------------
# Load Model
# --------------------------------

@st.cache_resource
def load_deepfake_model():
    return load_model("model/siamese_deepfake.pth", cfg)

model = load_deepfake_model()


# --------------------------------
# Prediction
# --------------------------------

def predict_frame(frame):

    frame_small = cv2.resize(frame,(320,240))

    feat = extract_features_from_frame(frame_small,cfg)

    feat = torch.from_numpy(feat).float().unsqueeze(0)

    with torch.no_grad():

        z_img,_ = model(feat,feat)

        distance = torch.norm(z_img).item()

    if distance < 1.0:
        return "REAL",distance
    else:
        return "FAKE",distance


# --------------------------------
# Sidebar
# --------------------------------

mode = st.sidebar.radio(
    "Select Mode",
    ["Image Detection","Video Detection"]
)

st.sidebar.markdown("---")
st.sidebar.write("Deepfake Detection Project")


# --------------------------------
# IMAGE MODE
# --------------------------------

if mode == "Image Detection":

    st.header("📷 Image Deepfake Detection")

    uploaded = st.file_uploader("Upload Image",type=["jpg","png","jpeg"])

    if uploaded:

        image = Image.open(uploaded)

        st.image(image,use_column_width=True)

        frame = np.array(image)

        label,dist = predict_frame(frame)

        if label == "REAL":
            st.success(f"Prediction: {label}")
        else:
            st.error(f"Prediction: {label}")

        st.write("Distance Score:",round(dist,3))


# --------------------------------
# VIDEO MODE
# --------------------------------

elif mode == "Video Detection":

    st.header("🎬 Video Deepfake Detection")

    uploaded_video = st.file_uploader("Upload Video",type=["mp4","avi","mov"])

    if uploaded_video:

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)

        frame_window = st.image([])

        while cap.isOpened():

            ret,frame = cap.read()

            if not ret:
                break

            label,dist = predict_frame(frame)

            color = (0,255,0) if label=="REAL" else (0,0,255)

            cv2.putText(frame,
                        f"{label} {dist:.2f}",
                        (30,50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        color,
                        2)

            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            frame_window.image(frame)

        cap.release()
