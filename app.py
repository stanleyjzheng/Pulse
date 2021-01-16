
from pathlib import Path

import os
import av
import cv2
import numpy as np
import PIL
import streamlit as st
from aiortc.contrib.media import MediaPlayer
import time


from streamlit_webrtc import (
    ClientSettings,
    VideoTransformerBase,
    WebRtcMode,
    webrtc_streamer,
)

from retinaface.pulse_retina import PulseMonitor

HERE = Path(__file__).parent


def main():
    st.header("WebRTC demo")
    pulsemonitor_page = "Real time pulse monitoring"
    loopback_page = "Simple video loopback"
    app_mode = st.sidebar.selectbox(
        "Choose the app mode",
        [
            pulsemonitor_page,
            loopback_page
        ],
    )
    st.subheader(app_mode)
    if app_mode == loopback_page:
        app_loopback()
    elif app_mode == pulsemonitor_page:
        get_pulsemonitor_frames()


def app_loopback():
    """ Simple video loopback """
    webrtc_streamer(
        key="loopback",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        video_transformer_factory=None,  # NoOp
    )

def get_pulsemonitor_frames():

    class NNVideoTransformer(VideoTransformerBase):

        def __init__(self) -> None:
            self.processor = PulseMonitor()

        def transform(self, frame: av.VideoFrame) -> np.ndarray:
            image = frame.to_ndarray(format="bgr24")
            self.processor.frame_in = image
            self.processor.run()
            annotated_image = self.processor.frame_out

            return annotated_image

    webrtc_ctx = webrtc_streamer(key="loopback", mode=WebRtcMode.SENDRECV, client_settings=WEBRTC_CLIENT_SETTINGS, video_transformer_factory=NNVideoTransformer, async_transform=True,)

WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)

if __name__ == "__main__":
    main()
