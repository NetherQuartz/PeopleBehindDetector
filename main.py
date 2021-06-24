"""The app"""

import logging
import torch
import streamlit as st

from pickle import UnpicklingError
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.transforms import ToTensor
from PIL import Image
from streamlit_webrtc import ClientSettings, WebRtcMode, webrtc_streamer

WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={
        "iceServers": [
            {
                "urls": ["stun:stun.l.google.com:19302"]
            }
        ]
    },
    media_stream_constraints={
        "video": True,
        "audio": False
    },
)

WEIGHTS_PATH = "weights.pth"

TO_TENSOR = ToTensor()


def detection_page(model):
    """Code of the page with detection on webcam image"""

    st.title("Say cheese! :camera:")

    webrtc_streamer(
        key="loopback",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        video_processor_factory=None
    )


def try_page(model):
    """Code of the page with tryout"""

    st.title("Try out the model :sparkler:")

    files = st.file_uploader(label="Upload images",
                             type=["png", "jpg", "jpeg"],
                             accept_multiple_files=True)

    images = []

    if len(files) > 0:
        for file in files:
            st.image(file.getvalue())
            image = TO_TENSOR(Image.open(file))[:3, :, :]
            images.append(image)
            logging.info(images[-1].shape)

        with st.spinner("Processing images…"):
            with torch.no_grad():
                ans = model(images)

        logging.info(len(ans))


@st.cache
def init_model():
    """Initializes and returns the model"""

    try:
        model = ssdlite320_mobilenet_v3_large(pretrained=False)
        weights = torch.load(WEIGHTS_PATH)
        model.load_state_dict(weights)

    except (FileNotFoundError, UnpicklingError) as _:
        model = ssdlite320_mobilenet_v3_large(pretrained=True)
        torch.save(model.state_dict(), WEIGHTS_PATH)

    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")
    model = model.to(device)

    return model


def main():
    """The program entry point"""

    st.set_page_config(page_title="People Behind Detector", page_icon=":mag:")
    st.sidebar.title("Let's detect 'em all!")

    mode = st.sidebar.selectbox(label="Choose a mode",
                                options=["Webcam detection", "Model tryout"])

    with st.spinner("Loading model, please wait…"):
        model = init_model()

    if mode == "Webcam detection":
        detection_page(model)
    elif mode == "Model tryout":
        try_page(model)


if __name__ == "__main__":
    main()
