"""The app"""

import logging
import queue

from pickle import UnpicklingError

import numpy as np
import torch
import streamlit as st

from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.transforms import ToTensor
from PIL import Image, ImageDraw
from streamlit_webrtc import ClientSettings, WebRtcMode, webrtc_streamer, VideoProcessorBase
from av.video.frame import VideoFrame

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

IMG_WIDTH = 600
DEFAULT_THRESHOLD = 0.5

TO_TENSOR = ToTensor()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def draw_box(draw, box, score):
    """Draws red box with score on an ImageDraw"""

    x_1, y_1, x_2, y_2 = box

    # the box
    draw.rectangle(xy=(x_1, y_1, x_2, y_2),
                   outline="#FF0000",
                   width=2)

    # text background
    draw.rectangle(xy=(x_1, y_2, x_2, y_2 - 13),
                   fill="#FF000000")

    draw.text((x_1 + 5, y_2 - 12), f"{score * 100:.3f}%", "#FFFFFF")  # text


def detection_page(model):
    """Code of the page with detection on webcam image"""

    st.title("Say cheese! :camera:")

    class VideoProcessor(VideoProcessorBase):

        threshold = DEFAULT_THRESHOLD
        result_queue = queue.Queue()

        def transform(self, frame: VideoFrame) -> np.ndarray:
            pass

        def recv(self, frame: VideoFrame) -> VideoFrame:
            image = frame.to_image()
            tensor = TO_TENSOR(image).to(DEVICE)

            with torch.no_grad():
                prediction = model([tensor])

            criterion = prediction[0]["labels"] == 1  # 1 — person

            boxes = prediction[0]["boxes"][criterion]
            scores = prediction[0]["scores"][criterion]

            draw = ImageDraw.Draw(image)
            people = 0
            for j, box in enumerate(boxes):
                if scores[j] < self.threshold:
                    continue

                people += 1
                draw_box(draw, box, scores[j])

            self.result_queue.put(people)
            return VideoFrame.from_image(image)

    ctx = webrtc_streamer(key="detection-filter",
                          mode=WebRtcMode.SENDRECV,
                          client_settings=WEBRTC_CLIENT_SETTINGS,
                          video_processor_factory=VideoProcessor,
                          async_processing=True)

    if ctx.video_processor is not None:
        threshold = st.sidebar.slider(label="Threshold",
                                      min_value=0.,
                                      max_value=1.,
                                      value=DEFAULT_THRESHOLD,
                                      step=0.01)
        ctx.video_processor.threshold = threshold

    if ctx.state.playing:
        count_placeholder = st.sidebar.empty()
        while True:
            if ctx.video_processor is not None:
                try:
                    result = ctx.video_processor.result_queue.get(timeout=.5)
                except queue.Empty:
                    result = None
                count_placeholder.text(f"Found {result} people")
            else:
                break


def try_page(model):
    """Code of the page with tryout"""

    st.title("Try out the model :sparkler:")
    threshold = st.sidebar.slider(label="Threshold",
                                  min_value=0.,
                                  max_value=1.,
                                  value=DEFAULT_THRESHOLD,
                                  step=0.01)

    files = st.file_uploader(label="Upload images",
                             type=["png", "jpg", "jpeg", "webp"],
                             accept_multiple_files=True)

    image_tensors = []
    images = []

    if len(files) > 0:
        for file in files:
            image = Image.open(file).convert("RGB").copy()
            image = image.resize((IMG_WIDTH, int(image.size[1] / image.size[0] * IMG_WIDTH)))
            images.append(image)

            image = TO_TENSOR(image).to(DEVICE)
            image_tensors.append(image)
            logging.info("Got tensor %s", image.shape)

        with st.spinner("Processing images…"):
            with torch.no_grad():
                prediction = model(image_tensors)

            for i, image in enumerate(images):

                criterion = prediction[i]["labels"] == 1  # 1 — person

                boxes = prediction[i]["boxes"][criterion]
                scores = prediction[i]["scores"][criterion]

                draw = ImageDraw.Draw(image)
                for j, box in enumerate(boxes):
                    if scores[j] < threshold:
                        continue

                    draw_box(draw, box, scores[j])

                st.image(image)

        logging.info("Processed %d images", len(prediction))


@st.cache
def init_model():
    """Initializes and returns the model"""

    try:
        logging.info("Trying to load weights from %s", WEIGHTS_PATH)
        model = ssdlite320_mobilenet_v3_large(pretrained=False)
        weights = torch.load(WEIGHTS_PATH)
        model.load_state_dict(weights)
        logging.info("Weights loaded from file %s", WEIGHTS_PATH)

    except (FileNotFoundError, UnpicklingError) as _:
        logging.info("Failed to load weights, downloading from the internet")
        model = ssdlite320_mobilenet_v3_large(pretrained=True)
        torch.save(model.state_dict(), WEIGHTS_PATH)
        logging.info("Weights downloaded and saved to %s", WEIGHTS_PATH)

    model.eval()

    logging.info("Device: %s", DEVICE)
    model = model.to(DEVICE)

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
