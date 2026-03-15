import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import threading

# ---------- Configuration & State ----------
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Use a lock to safely share the latest frame for snapshots
lock = threading.Lock()
container = {"img": None}

st.set_page_config(page_title="Snapchat Filter Pro", layout="centered")
st.title("Snapchat Multi Filter OpenCV 😎")

# ---------- Load Detectors ----------
@st.cache_resource
def load_models():
    face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    nose = cv2.CascadeClassifier("haarcascade_mcs_nose.xml")
    return face, nose

face_cascade, nose_cascade = load_models()

# ---------- Load filters ----------
@st.cache_resource
def load_filters():
    return {
        "None": None,
        "Glasses": cv2.imread("filters/glasses.png", -1),
        "Mask": cv2.imread("filters/mask.png", -1),
        "Moustache": cv2.imread("filters/moustache.png", -1),
        "Cap": cv2.imread("filters/cap.png", -1),
        "DogEars": cv2.imread("filters/dogears.png", -1),
    }

filters = load_filters()
filter_option = st.sidebar.selectbox("Select Filter", list(filters.keys()))

# ---------- Overlay Logic ----------
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

# ---------- Apply Filter ----------
def apply_filter(img, filter_name):
    overlay = filters[filter_name]
    if overlay is None: return img
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        noses = nose_cascade.detectMultiScale(roi_gray, 1.3, 5)

        if filter_name == "Cap":
            mw = int(w * 0.9)
            mh = int(mw * overlay.shape[0] / overlay.shape[1])
            img = overlay_image(img, overlay, x + (w//2) - (mw//2), int(y - mh*0.8), mw, mh)
        elif filter_name == "DogEars":
            mw = int(w * 1.0)
            mh = int(mw * overlay.shape[0] / overlay.shape[1])
            img = overlay_image(img, overlay, x + (w//2) - (mw//2), int(y - mh*0.7), mw, mh)

        for (nx, ny, nw, nh) in noses:
            n_top, n_bot, n_center_x = y + ny, y + ny + nh, x + nx + (nw // 2)
            if filter_name == "Moustache":
                mw = int(nw * 1.8)
                mh = int(mw * overlay.shape[0] / overlay.shape[1])
                img = overlay_image(img, overlay, n_center_x - (mw//2), n_bot - int(mh*0.7), mw, mh)
            elif filter_name == "Mask":
                mw = int(w * 1.0)
                mh = int(mw * overlay.shape[0] / overlay.shape[1])
                img = overlay_image(img, overlay, (x+w//2) - (mw//2), n_top - int(mh*0.65), mw, mh)
            elif filter_name == "Glasses":
                mw = int(w * 0.8)
                mh = int(mw * overlay.shape[0] / overlay.shape[1])
                img = overlay_image(img, overlay, (x+w//2) - (mw//2), n_top - int(mh*0.55), mw, mh)
            break 
    return img

# ---------- Camera Logic ----------
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    # Apply the filter selected in the sidebar
    img = apply_filter(img, filter_option)
    
    # Store the frame for the snapshot feature
    with lock:
        container["img"] = img
        
    return av.VideoFrame.from_ndarray(img, format="bgr24")

ctx = webrtc_streamer(
    key="filter-app",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# ---------- Snapshot Feature ----------
if st.button("📸 Take Photo"):
    with lock:
        snap = container["img"]
    if snap is not None:
        # Convert BGR to RGB for Streamlit
        snap_rgb = cv2.cvtColor(snap, cv2.COLOR_BGR2RGB)
        st.image(snap_rgb, caption="Your Snapshot (Right-click to save)")
    else:
        st.warning("Please start the camera first!")
