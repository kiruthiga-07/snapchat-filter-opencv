import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import os

st.set_page_config(page_title="Snapchat Filter Pro", layout="centered")
st.title("Snapchat Multi Filter OpenCV 😎")

# ---------- Face detector ----------
# Added a check to ensure the file exists to prevent crashing
if not os.path.exists("haarcascade_frontalface_default.xml"):
    st.error("Model file 'haarcascade_frontalface_default.xml' not found! Please upload it to your repo.")
    st.stop()

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# ---------- Load filters ----------
@st.cache_resource
def load_filters():
    # Using -1 to load the Alpha Channel (transparency)
    return {
        "None": None,
        "Glasses": cv2.imread("filters/glasses.png", -1),
        "Mask": cv2.imread("filters/mask.png", -1),
        "Moustache": cv2.imread("filters/moustache.png", -1),
        "Cap": cv2.imread("filters/cap.png", -1),
        "DogEars": cv2.imread("filters/dogears.png", -1),
    }

filters = load_filters()

filter_option = st.selectbox("Select Filter", list(filters.keys()))
mode = st.radio("Mode", ["Image", "Camera"], horizontal=True)

# ---------- Improved Overlay Function ----------
def overlay_image(bg, overlay, x, y, w, h):
    if overlay is None:
        return bg

    # Calculate boundaries to prevent indexing errors if filter goes off-screen
    h_bg, w_bg = bg.shape[:2]
    
    # Resize filter to target size
    overlay_res = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_AREA)

    # Calculate clipping
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w, w_bg), min(y + h, h_bg)

    # Calculate corresponding region in the overlay
    ol_x1, ol_y1 = x1 - x, y1 - y
    ol_x2, ol_y2 = ol_x1 + (x2 - x1), ol_y1 + (y2 - y1)

    if x1 >= x2 or y1 >= y2:
        return bg

    # Slice the regions
    overlay_patch = overlay_res[ol_y1:ol_y2, ol_x1:ol_x2]
    bg_patch = bg[y1:y2, x1:x2]

    # Alpha blending
    if overlay_patch.shape[2] == 4:
        alpha = overlay_patch[:, :, 3] / 255.0
        for c in range(3):
            bg_patch[:, :, c] = (alpha * overlay_patch[:, :, c] + 
                                (1.0 - alpha) * bg_patch[:, :, c])
        bg[y1:y2, x1:x2] = bg_patch

    return bg

# ---------- Apply filter with Adjusted Offsets ----------
def apply_filter(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    overlay = filters[filter_option]

    if overlay is None:
        return img

    for (x, y, w, h) in faces:
        # These multipliers are tuned for standard front-facing PNGs
        if filter_option == "Glasses":
            gw, gh = int(w * 0.9), int(h * 0.25)
            gx, gy = int(x + w * 0.05), int(y + h * 0.35)
            img = overlay_image(img, overlay, gx, gy, gw, gh)

        elif filter_option == "Mask":
            # Mask needs to be slightly wider than the face
            gw, gh = int(w * 1.0), int(h * 0.45)
            gx, gy = x, int(y + h * 0.25)
            img = overlay_image(img, overlay, gx, gy, gw, gh)

        elif filter_option == "Moustache":
            # Placed between nose and mouth
            gw, gh = int(w * 0.4), int(h * 0.15)
            gx, gy = int(x + w * 0.3), int(y + h * 0.68)
            img = overlay_image(img, overlay, gx, gy, gw, gh)

        elif filter_option == "Cap":
            # Sits on top of the head (y is negative relative to detected box)
            gw, gh = int(w * 1.2), int(h * 0.7)
            gx, gy = int(x - w * 0.1), int(y - h * 0.55)
            img = overlay_image(img, overlay, gx, gy, gw, gh)

        elif filter_option == "DogEars":
            gw, gh = int(w * 1.2), int(h * 0.6)
            gx, gy = int(x - w * 0.1), int(y - h * 0.45)
            img = overlay_image(img, overlay, gx, gy, gw, gh)

    return img

# ================= RUNTIME LOGIC =================

if mode == "Image":
    uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if uploaded:
        image = Image.open(uploaded)
        img = np.array(image)
        # Convert RGB to BGR for OpenCV processing
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = apply_filter(img)
        # Convert back to RGB for Streamlit display
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True)

else:
    class VideoProcessor(VideoProcessorBase):
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            # Apply filter directly
            img = apply_filter(img)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="filter-app",
        video_processor_factory=VideoProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
