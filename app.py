import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av


st.title("Snapchat Multi Filter OpenCV 😎")


# ---------- Face detector ----------
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)


# ---------- Load filters ----------
filters = {
    "None": None,
    "Glasses": cv2.imread("filters/glasses.png", -1),
    "Mask": cv2.imread("filters/mask.png", -1),
    "Moustache": cv2.imread("filters/moustache.png", -1),
    "Cap": cv2.imread("filters/cap.png", -1),
    "DogEars": cv2.imread("filters/dogears.png", -1),
}


filter_option = st.selectbox(
    "Select Filter",
    list(filters.keys())
)


mode = st.radio(
    "Mode",
    ["Image", "Camera"]
)


# ---------- Overlay function ----------
def overlay_image(bg, overlay, x, y, w, h):

    if overlay is None:
        return bg

    overlay = cv2.resize(
        overlay,
        (w, h),
        interpolation=cv2.INTER_AREA
    )

    h_bg, w_bg, _ = bg.shape

    if x < 0:
        x = 0
    if y < 0:
        y = 0

    if x + w > w_bg:
        w = w_bg - x
    if y + h > h_bg:
        h = h_bg - y

    overlay = overlay[0:h, 0:w]

    if overlay.shape[2] == 4:

        alpha = overlay[:, :, 3] / 255.0

        for c in range(3):
            bg[y:y+h, x:x+w, c] = (
                alpha * overlay[:, :, c]
                + (1 - alpha) * bg[y:y+h, x:x+w, c]
            )

    return bg


# ---------- Apply filter ----------
def apply_filter(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        1.3,
        5
    )

    overlay = filters[filter_option]

    for (x, y, w, h) in faces:

        if overlay is None:
            continue

        # -------- Glasses --------
        if filter_option == "Glasses":

            gw = int(w * 1.05)
            gh = int(h * 0.35)

            gx = int(x - w * 0.02)
            gy = int(y + h * 0.30)

            img = overlay_image(
                img,
                overlay,
                gx,
                gy,
                gw,
                gh
            )

        # -------- Mask --------
        elif filter_option == "Mask":

            img = overlay_image(
                img,
                overlay,
                x,
                int(y + h * 0.55),
                w,
                int(h * 0.45)
            )

        # -------- Moustache --------
        elif filter_option == "Moustache":

            img = overlay_image(
                img,
                overlay,
                int(x + w * 0.2),
                int(y + h * 0.65),
                int(w * 0.6),
                int(h * 0.25)
            )

        # -------- Cap --------
        elif filter_option == "Cap":

            img = overlay_image(
                img,
                overlay,
                x,
                int(y - h * 0.5),
                w,
                int(h * 0.6)
            )

        # -------- Dog ears --------
        elif filter_option == "DogEars":

            img = overlay_image(
                img,
                overlay,
                x,
                int(y - h * 0.5),
                w,
                int(h * 0.6)
            )

    return img


# ================= IMAGE MODE =================

if mode == "Image":

    uploaded = st.file_uploader(
        "Upload Image",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded is not None:

        image = Image.open(uploaded)
        img = np.array(image)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        img = apply_filter(img)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        st.image(img)


# ================= CAMERA MODE =================

class VideoProcessor(VideoProcessorBase):

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        # -------- sharpen --------
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])

        img = cv2.filter2D(img, -1, kernel)

        img = apply_filter(img)

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )


if mode == "Camera":

    webrtc_streamer(
        key="snap",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 1280},
                "height": {"ideal": 720},
                "frameRate": {"ideal": 30},
            },
            "audio": False,
        },
    )
