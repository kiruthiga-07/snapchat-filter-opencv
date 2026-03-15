import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import os

st.set_page_config(page_title="Snapchat Filter Pro", layout="centered")
st.title("Snapchat Multi Filter OpenCV 😎")

# ---------- Load Detectors ----------
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
nose_cascade = cv2.CascadeClassifier("haarcascade_mcs_nose.xml")

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
filter_option = st.selectbox("Select Filter", list(filters.keys()))
mode = st.radio("Mode", ["Image", "Camera"], horizontal=True)

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

# ---------- The Adjusted Scaling Logic ----------
def apply_filter(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    overlay = filters[filter_option]
    if overlay is None: return img

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        noses = nose_cascade.detectMultiScale(roi_gray, 1.3, 5)

        # 1. HEAD FILTERS (Cap/Ears) - Scaled Down
        if filter_option == "Cap":
            # Width reduced from 1.1 to 0.9 for a tighter fit
            mw = int(w * 0.9)
            mh = int(mw * overlay.shape[0] / overlay.shape[1])
            img = overlay_image(img, overlay, x + (w//2) - (mw//2), int(y - mh*0.8), mw, mh)
            
        elif filter_option == "DogEars":
            # Width reduced from 1.2 to 1.0
            mw = int(w * 1.0)
            mh = int(mw * overlay.shape[0] / overlay.shape[1])
            img = overlay_image(img, overlay, x + (w//2) - (mw//2), int(y - mh*0.7), mw, mh)

        # 2. NOSE/EYE/MOUSTACHE FILTERS
        for (nx, ny, nw, nh) in noses:
            n_top = y + ny
            n_bot = y + ny + nh
            n_center_x = x + nx + (nw // 2)

            if filter_option == "Moustache":
                # Reduced width from 2.5 to 1.8 of nose width
                mw = int(nw * 1.8) 
                mh = int(mw * overlay.shape[0] / overlay.shape[1])
                img = overlay_image(img, overlay, n_center_x - (mw//2), n_bot - int(mh*0.7), mw, mh)
            
           elif filter_option == "Mask":
                # Width stays at 1.0 for the face width
                mw = int(w * 1.0) 
                mh = int(mw * overlay.shape[0] / overlay.shape[1])
                # Positioning: 0.65 pulls it up slightly more than 0.60
                img = overlay_image(img, overlay, (x+w//2) - (mw//2), n_top - int(mh*0.65), mw, mh)

            elif filter_option == "Glasses":
                # Reduced width from 0.9 to 0.8
                mw = int(w * 0.8)
                mh = int(mw * overlay.shape[0] / overlay.shape[1])
                img = overlay_image(img, overlay, (x+w//2) - (mw//2), n_top - int(mh*0.55), mw, mh)
            break 
            
    return img

# ================= RUNTIME =================
if mode == "Image":
    uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if uploaded:
        image = Image.open(uploaded)
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = apply_filter(img)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
else:
    class VideoProcessor(VideoProcessorBase):
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            return av.VideoFrame.from_ndarray(apply_filter(img), format="bgr24")
    webrtc_streamer(key="snap", video_processor_factory=VideoProcessor)
