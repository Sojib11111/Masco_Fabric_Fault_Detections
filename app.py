from __future__ import annotations

import base64
import mimetypes
import os
import threading
from pathlib import Path
from typing import Any

import av
import cv2
import streamlit as st
import torch
from streamlit_webrtc import (
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)
from ultralytics import YOLO


# ============================================================
# PROJECT PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
LOGO_PATH = BASE_DIR / "assets" / "masco_logo.jpg"
MODEL_PATH = BASE_DIR / "model" / "best.pt"


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Fabric Fault Detector",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# SESSION STATE
# ============================================================
if "camera_started" not in st.session_state:
    st.session_state.camera_started = False


# ============================================================
# CONVERT LOGO TO DATA URI
# ============================================================
def get_logo_data_uri(logo_path: Path) -> str:
    """Convert the logo to a browser-safe Base64 data URI."""

    if not logo_path.is_file():
        return ""

    try:
        mime_type, _ = mimetypes.guess_type(str(logo_path))

        if not mime_type:
            mime_type = "image/jpeg"

        encoded_logo = base64.b64encode(
            logo_path.read_bytes()
        ).decode("utf-8")

        return f"data:{mime_type};base64,{encoded_logo}"

    except Exception:
        return ""


LOGO_DATA_URI = get_logo_data_uri(LOGO_PATH)


# ============================================================
# GLOBAL CSS
# ============================================================
st.html(
    """
    <style>
        #MainMenu,
        footer,
        header,
        [data-testid="stToolbar"],
        [data-testid="stSidebar"],
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"],
        [data-testid="stHeader"] {
            display: none !important;
            visibility: hidden !important;
        }

        html,
        body,
        .stApp {
            margin: 0 !important;
            padding: 0 !important;
            width: 100% !important;
            min-height: 100vh !important;
            overflow-x: hidden !important;

            background:
                radial-gradient(
                    circle at 15% 15%,
                    rgba(37, 99, 235, 0.30),
                    transparent 32%
                ),
                radial-gradient(
                    circle at 85% 85%,
                    rgba(20, 184, 166, 0.23),
                    transparent 30%
                ),
                linear-gradient(
                    145deg,
                    #020617 0%,
                    #071426 50%,
                    #020617 100%
                ) !important;
        }

        .block-container {
            width: 100% !important;
            max-width: 100% !important;
            margin: 0 !important;
            padding: 0 !important;
        }

        /* Starting screen */
        .start-screen {
            width: 100%;
            min-height: 77vh;

            display: flex;
            align-items: center;
            justify-content: center;

            box-sizing: border-box;
            padding: 25px;
        }

        .start-card {
            position: relative;

            width: min(90vw, 640px);
            padding: 55px 35px 115px;

            overflow: hidden;
            text-align: center;

            border-radius: 32px;
            border: 1px solid rgba(255, 255, 255, 0.13);

            background:
                linear-gradient(
                    145deg,
                    rgba(15, 23, 42, 0.94),
                    rgba(8, 20, 38, 0.90)
                );

            box-shadow:
                0 35px 100px rgba(0, 0, 0, 0.50),
                inset 0 1px 0 rgba(255, 255, 255, 0.08);

            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
        }

        .start-card::before {
            content: "";

            position: absolute;
            width: 260px;
            height: 260px;

            top: -155px;
            right: -100px;

            border-radius: 50%;

            background: rgba(37, 99, 235, 0.42);
            filter: blur(3px);
        }

        .start-card::after {
            content: "";

            position: absolute;
            width: 220px;
            height: 220px;

            left: -120px;
            bottom: -145px;

            border-radius: 50%;

            background: rgba(20, 184, 166, 0.32);
            filter: blur(3px);
        }

        /* MASCO logo container */
        .detector-icon {
            position: relative;
            z-index: 2;

            width: 180px;
            height: 150px;
            margin: 0 auto 24px;

            display: flex;
            align-items: center;
            justify-content: center;

            padding: 8px;
            box-sizing: border-box;

            background: #ffffff;
            border: 1px solid rgba(255, 255, 255, 0.20);
            border-radius: 20px;

            box-shadow:
                0 20px 50px rgba(37, 99, 235, 0.30),
                inset 0 1px 0 rgba(255, 255, 255, 0.80);
        }

        .detector-icon img {
            display: block;
            width: 100%;
            height: 100%;

            object-fit: contain;
            object-position: center;

            border-radius: 10px;
        }

        .logo-error {
            display: flex;
            align-items: center;
            justify-content: center;

            width: 100%;
            height: 100%;

            color: #1e3a8a;

            font-family:
                Inter,
                Arial,
                sans-serif;

            font-size: 18px;
            font-weight: 900;
        }

        .detector-title {
            position: relative;
            z-index: 2;

            margin: 0;

            color: #ffffff;

            font-family:
                Inter,
                Arial,
                sans-serif;

            font-size: clamp(32px, 5vw, 54px);
            font-weight: 850;
            line-height: 1.12;
            letter-spacing: -1.5px;
        }

        /* Streamlit button container */
        div[data-testid="stButton"] {
            position: relative;
            z-index: 20;

            width: min(80vw, 320px);
            margin: -125px auto 0 auto;
        }

        div[data-testid="stButton"] > button {
            width: 100% !important;
            min-height: 62px !important;

            border: 1px solid rgba(255, 255, 255, 0.20) !important;
            border-radius: 18px !important;

            color: #ffffff !important;

            background:
                linear-gradient(
                    135deg,
                    #2563eb 0%,
                    #0891b2 52%,
                    #14b8a6 100%
                ) !important;

            font-family:
                Inter,
                Arial,
                sans-serif !important;

            font-size: 17px !important;
            font-weight: 800 !important;
            letter-spacing: 0.5px !important;

            box-shadow:
                0 18px 45px rgba(37, 99, 235, 0.35),
                inset 0 1px 0 rgba(255, 255, 255, 0.25) !important;

            transition:
                transform 0.20s ease,
                box-shadow 0.20s ease,
                filter 0.20s ease !important;
        }

        div[data-testid="stButton"] > button:hover {
            transform: translateY(-3px) scale(1.015) !important;

            box-shadow:
                0 24px 60px rgba(37, 99, 235, 0.46),
                inset 0 1px 0 rgba(255, 255, 255, 0.28) !important;

            filter: brightness(1.08) !important;
        }

        div[data-testid="stButton"] > button:active {
            transform: translateY(0) scale(0.985) !important;
        }

        div[data-testid="stButton"] > button:focus {
            outline: none !important;

            box-shadow:
                0 0 0 4px rgba(59, 130, 246, 0.20),
                0 20px 50px rgba(37, 99, 235, 0.42) !important;
        }

        /* WebRTC area */
        [data-testid="stCustomComponentV1"] {
            width: min(100vw, 1380px) !important;
            margin: 0 auto !important;
        }

        iframe {
            display: block !important;

            width: 100% !important;
            min-height: 96vh !important;

            margin: 0 !important;

            border: none !important;
            border-radius: 0 !important;

            background: #000000 !important;
            box-shadow: none !important;
        }

        /* Mobile */
        @media (max-width: 768px) {
            .start-screen {
                min-height: 74vh;
                padding: 15px;
            }

            .start-card {
                width: 100%;

                padding:
                    45px
                    20px
                    105px;

                border-radius: 26px;
            }

            .detector-icon {
                width: 150px;
                height: 125px;

                margin-bottom: 22px;
                padding: 7px;

                border-radius: 18px;
            }

            .detector-icon img {
                border-radius: 9px;
            }

            .detector-title {
                font-size: clamp(31px, 9vw, 43px);
                letter-spacing: -1px;
            }

            div[data-testid="stButton"] {
                width: min(82vw, 300px);
                margin-top: -115px;
            }

            div[data-testid="stButton"] > button {
                min-height: 58px !important;
                font-size: 16px !important;
                border-radius: 16px !important;
            }

            [data-testid="stCustomComponentV1"] {
                width: 100% !important;
                margin: 0 !important;
            }

            iframe {
                width: 100% !important;
                min-height: 96vh !important;
            }
        }

        /* Mobile landscape */
        @media screen
        and (max-height: 550px)
        and (orientation: landscape) {
            .start-screen {
                min-height: 65vh;
                padding: 8px;
            }

            .start-card {
                padding:
                    20px
                    20px
                    92px;
            }

            .detector-icon {
                width: 105px;
                height: 88px;

                margin-bottom: 10px;
                padding: 5px;

                border-radius: 14px;
            }

            .detector-title {
                font-size: 30px;
            }

            div[data-testid="stButton"] {
                margin-top: -100px;
            }

            iframe {
                min-height: 96vh !important;
            }
        }
    </style>
    """
)


# ============================================================
# DETECTION CONFIGURATION
# ============================================================
CONFIDENCE = float(
    os.getenv(
        "YOLO_CONFIDENCE",
        "0.25",
    )
)

IOU_THRESHOLD = float(
    os.getenv(
        "YOLO_IOU",
        "0.45",
    )
)

IMAGE_SIZE = int(
    os.getenv(
        "YOLO_IMAGE_SIZE",
        "640",
    )
)

MAX_DETECTIONS = int(
    os.getenv(
        "YOLO_MAX_DETECTIONS",
        "100",
    )
)

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 20


# ============================================================
# GPU / CPU
# ============================================================
if torch.cuda.is_available():
    DEVICE: Any = 0
    USE_HALF = True

else:
    DEVICE = "cpu"
    USE_HALF = False


# ============================================================
# LOAD MODEL ONCE
# ============================================================
@st.cache_resource(show_spinner=False)
def load_model(model_path: Path) -> YOLO:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model পাওয়া যায়নি: {model_path}. "
            "model folder-এর মধ্যে best.pt রাখুন।"
        )

    loaded_model = YOLO(
        str(model_path)
    )

    if torch.cuda.is_available():
        loaded_model.to("cuda")

        try:
            dummy_image = torch.zeros(
                1,
                3,
                IMAGE_SIZE,
                IMAGE_SIZE,
                device="cuda",
            )

            with torch.inference_mode():
                loaded_model.predict(
                    source=dummy_image,
                    imgsz=IMAGE_SIZE,
                    device=DEVICE,
                    half=True,
                    verbose=False,
                )

        except Exception:
            pass

    return loaded_model


try:
    MODEL = load_model(
        MODEL_PATH
    )

except Exception as error:
    st.error(
        f"Model load error: {error}"
    )
    st.stop()


MODEL_LOCK = threading.Lock()


# ============================================================
# LIVE YOLO DETECTOR
# ============================================================
class FabricFaultDetector(VideoProcessorBase):
    def recv(
        self,
        frame: av.VideoFrame,
    ) -> av.VideoFrame:

        image = frame.to_ndarray(
            format="bgr24"
        )

        if image is None or image.size == 0:
            return frame

        try:
            with MODEL_LOCK:
                with torch.inference_mode():
                    results = MODEL.predict(
                        source=image,
                        conf=CONFIDENCE,
                        iou=IOU_THRESHOLD,
                        imgsz=IMAGE_SIZE,
                        device=DEVICE,
                        half=USE_HALF,
                        max_det=MAX_DETECTIONS,
                        agnostic_nms=False,
                        verbose=False,
                    )

            if not results:
                return av.VideoFrame.from_ndarray(
                    image,
                    format="bgr24",
                )

            detected_frame = results[0].plot(
                boxes=True,
                labels=True,
                conf=True,
                line_width=3,
            )

            return av.VideoFrame.from_ndarray(
                detected_frame,
                format="bgr24",
            )

        except RuntimeError as error:
            output = image.copy()

            cv2.putText(
                output,
                f"Detection error: {str(error)[:70]}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return av.VideoFrame.from_ndarray(
                output,
                format="bgr24",
            )

        except Exception as error:
            output = image.copy()

            cv2.putText(
                output,
                f"Error: {str(error)[:80]}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            return av.VideoFrame.from_ndarray(
                output,
                format="bgr24",
            )


# ============================================================
# WEBRTC CONFIGURATION
# ============================================================
RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {
                "urls": [
                    "stun:stun.l.google.com:19302",
                    "stun:stun1.l.google.com:19302",
                ]
            }
        ]
    }
)


# ============================================================
# START PAGE
# ============================================================
if not st.session_state.camera_started:

    if LOGO_DATA_URI:
        logo_content = f"""
            <img
                src="{LOGO_DATA_URI}"
                alt="MASCO Logo"
            >
        """

    else:
        logo_content = """
            <div class="logo-error">
                Logo Not Found
            </div>
        """

    start_page_html = f"""
        <div class="start-screen">
            <div class="start-card">

                <div class="detector-icon">
                    {logo_content}
                </div>

                <h1 class="detector-title">
                    Fabric Fault Detector
                </h1>

            </div>
        </div>
    """

    st.html(
        start_page_html
    )

    start_clicked = st.button(
        "START CAMERA",
        type="primary",
        use_container_width=True,
    )

    if start_clicked:
        st.session_state.camera_started = True
        st.rerun()


# ============================================================
# CAMERA AND LIVE DETECTION
# ============================================================
else:
    webrtc_ctx = webrtc_streamer(
        key="fabric-fault-live-camera",

        mode=WebRtcMode.SENDRECV,

        rtc_configuration=RTC_CONFIGURATION,

        video_processor_factory=FabricFaultDetector,

        # Custom START button চাপার পর camera automatically start করবে
        desired_playing_state=True,

        media_stream_constraints={
            "video": {
                "width": {
                    "ideal": CAMERA_WIDTH,
                },

                "height": {
                    "ideal": CAMERA_HEIGHT,
                },

                "frameRate": {
                    "ideal": CAMERA_FPS,
                    "max": 30,
                },

                # Phone-এ rear camera prefer করবে
                # Laptop-এ available webcam ব্যবহার করবে
                "facingMode": {
                    "ideal": "environment",
                },
            },

            # Audio সম্পূর্ণ বন্ধ
            "audio": False,
        },

        async_processing=True,
    )