from utils.model_utils import load_model, predict_frame
from utils.feature_utils import extract_features_from_frame
st.set_page_config(page_title="Deepfake Detector", layout="wide")
st.markdown("Detect whether a face is **REAL or FAKE** using a Siamese Deepfake Model")

# -------------------------
# Load Model
# -------------------------

@st.cache_resource
def load_model(model_path):

    device = torch.device("cpu")
    checkpoint = torch.load(model_path, map_location=device)

    feat_dim = checkpoint["feat_dim"]

    model = Siamese(feat_dim)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model


model = load_model("model/siamese_deepfake.pth")

# -------------------------
# Sidebar Controls
# -------------------------

mode = st.sidebar.radio(
    "Choose Input Mode",
    ["Image", "Video", "Webcam"]
)

st.sidebar.markdown("---")
st.sidebar.write("Created for Deepfake Detection Project")

# -------------------------
# Prediction Function
# -------------------------

def predict_frame(frame):

    frame_small = cv2.resize(frame, (320,240))

    feat = extract_features_from_frame(frame_small, cfg)
    feat = torch.from_numpy(feat).float().unsqueeze(0)

    with torch.no_grad():
        z_img, _ = model(feat, feat)

        dist_real = torch.nn.functional.pairwise_distance(z_img, ref_real)
        dist_fake = torch.nn.functional.pairwise_distance(z_img, ref_fake)

    if dist_real.item() < dist_fake.item():
        return "REAL", dist_real.item()
    else:
        return "FAKE", dist_fake.item()


# ==========================
# IMAGE MODE
# ==========================

if mode == "Image":

    st.header("📷 Image Detection")

    uploaded = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if uploaded:

        image = Image.open(uploaded)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        frame = np.array(image)

        label, dist = predict_frame(frame)

        if label == "REAL":
            st.success(f"Prediction: {label}")
        else:
            st.error(f"Prediction: {label}")

        st.write("Distance Score:", dist)

# ==========================
# VIDEO MODE
# ==========================

elif mode == "Video":

    st.header("🎬 Video Detection")

    uploaded_video = st.file_uploader("Upload Video", type=["mp4","avi","mov"])

    if uploaded_video:

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)

        frame_window = st.image([])

        while cap.isOpened():

            ret, frame = cap.read()
            if not ret:
                break

            label, dist = predict_frame(frame)

            color = (0,255,0) if label=="REAL" else (0,0,255)

            cv2.putText(frame,
                        f"{label} {dist:.2f}",
                        (30,50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        color,
                        2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame_window.image(frame)

        cap.release()

# ==========================
# WEBCAM MODE
# ==========================

elif mode == "Webcam":

    st.header("📡 Webcam Detection")

    run = st.checkbox("Start Webcam")

    frame_window = st.image([])

    cap = cv2.VideoCapture(0)

    while run:

        ret, frame = cap.read()

        if not ret:
            break

        label, dist = predict_frame(frame)

        color = (0,255,0) if label=="REAL" else (0,0,255)

        cv2.putText(frame,
                    f"{label} {dist:.2f}",
                    (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_window.image(frame)

    cap.release()
