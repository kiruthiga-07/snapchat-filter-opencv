import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import os

# 1. SETUP & PATH HANDLING
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_XML = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
NOSE_XML = os.path.join(BASE_DIR, "haarcascade_mcs_nose.xml")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.set_page_config(page_title="Snapchat Filter Pro", layout="wide")
st.title("Snapchat Multi-Filter OpenCV 😎")

# 2. LOAD MODELS & FILTERS
@st.cache_resource
def load_resources():
    if not os.path.exists(FACE_XML) or not os.path.exists(NOSE_XML):
        return None, None, None

    face = cv2.CascadeClassifier(FACE_XML)
    nose = cv2.CascadeClassifier(NOSE_XML)
    
    filter_dir = os.path.join(BASE_DIR, "filters")
    if not os.path.exists(filter_dir):
        return face, nose, {}

    f_imgs = {
        "None": None,
        "Glasses": cv2.imread(os.path.join(filter_dir, "glasses.png"), -1),
        "Mask": cv2.imread(os.path.join(filter_dir, "mask.png"), -1),
        "Moustache": cv2.imread(os.path.join(filter_dir, "moustache.png"), -1),
        "Cap": cv2.imread(os.path.join(filter_dir, "cap.png"), -1),
        "DogEars": cv2.imread(os.path.join(filter_dir, "dogears.png"), -1),
    }
    return face, nose, f_imgs

face_cascade, nose_cascade, filters = load_resources()

# Safety Check UI
if face_cascade is None or face_cascade.empty():
    st.error("🚨 XML Files not found! Make sure they are in your GitHub repo.")
    st.stop()

# 3. OVERLAY & FILTER LOGIC
def overlay_image(bg, overlay, x, y, w, h):
    if overlay is None or w <= 0 or h <= 0: return bg
    overlay = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_AREA)
    h_bg, w_bg = bg.shape[:2]
    x1, y1, x2, y2 = max(x, 0), max(y, 0), min(x + w, w_bg), min(y + h, h_bg)
    ol_x1, ol_y1 = x1 - x, y1 - y
    ol_x2, ol_y2 = ol_x1 + (x2 - x1), ol_y1 + (y2 - y1)
    if x1 >= x2 or y1 >= y2: return bg
    overlay_patch = overlay[ol_y1:ol_y2, ol_x1:ol_x2]
    bg_patch = bg[y1:y2, x1:x2]
    if overlay_patch.shape[2] == 4:
        alpha = overlay_patch[:, :, 3] / 255.0
        for c in range(3):
            bg_patch[:, :, c] = (alpha * overlay_patch[:, :, c] + (1 - alpha) * bg_patch[:, :, c])
        bg[y1:y2, x1:x2] = bg_patch
    return bg

def apply_filter(img, filter_name):
    if filter_name == "None": return img
    overlay = filters.get(filter_name)
    if overlay is None: return img
    
    output = img.copy()
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        noses = nose_cascade.detectMultiScale(roi_gray, 1.3, 5)

        if filter_name == "Cap":
            mw = int(w * 0.9); mh = int(mw * overlay.shape[0] / overlay.shape[1])
            output = overlay_image(output, overlay, x + (w//2) - (mw//2), int(y - mh*0.8), mw, mh)
        elif filter_name == "DogEars":
            mw = int(w * 1.0); mh = int(mw * overlay.shape[0] / overlay.shape[1])
            output = overlay_image(output, overlay, x + (w//2) - (mw//2), int(y - mh*0.7), mw, mh)

        for (nx, ny, nw, nh) in noses:
            n_top, n_bot, n_center_x = y + ny, y + ny + nh, x + nx + (nw // 2)
            if filter_name == "Moustache":
                mw = int(nw * 1.8)
                mh = int(mw * overlay.shape[0] / overlay.shape[1])
                # Changed from 0.7 to 0.5 to move it down slightly
                output = overlay_image(output, overlay, n_center_x - (mw//2), n_bot - int(mh*0.6), mw, mh)
            elif filter_name == "Mask":
                mw = int(w * 1.0); mh = int(mw * overlay.shape[0] / overlay.shape[1])
                output = overlay_image(output, overlay, (x+w//2) - (mw//2), n_top - int(mh*0.65), mw, mh)
            elif filter_name == "Glasses":
                mw = int(w * 0.8); mh = int(mw * overlay.shape[0] / overlay.shape[1])
                output = overlay_image(output, overlay, (x+w//2) - (mw//2), n_top - int(mh*0.55), mw, mh)
            break 
    return output

# 4. USER INTERFACE
filter_option = st.sidebar.selectbox("Choose Your Filter", list(filters.keys()))
tab1, tab2 = st.tabs(["🎥 Live Camera", "📤 Upload Image"])

with tab1:
    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        processed_img = apply_filter(img, filter_option)
        return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

    webrtc_streamer(
        key="camera-snap",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with tab2:
    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        result = apply_filter(image, filter_option)
        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_container_width=True)
