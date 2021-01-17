import os
import av
import traceback
import numpy as np
import streamlit as st
from retinaface.pulse_retina import PulseMonitor

from streamlit_webrtc import ClientSettings, VideoTransformerBase, WebRtcMode, webrtc_streamer


def main():
    st.header("Pulse detector demo")
    get_pulsemonitor_frames()

def get_pulsemonitor_frames():
    class NNVideoTransformer(VideoTransformerBase):
        def __init__(self) -> None:
            self.processor = PulseMonitor()

        def transform(self, frame: av.VideoFrame) -> np.ndarray:
            try:
                image = frame.to_ndarray(format="bgr24")
                annotated_image, current_bpm = self.processor.process_frame(image)
                return annotated_image
            except Exception as e:
                print("Caught Exception")
                traceback.print_exc()
                return image
    
    webrtc_ctx = webrtc_streamer(key="loopback", mode=WebRtcMode.SENDRECV, client_settings=WEBRTC_CLIENT_SETTINGS, video_transformer_factory=NNVideoTransformer, async_transform=True,)

WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)

if __name__ == "__main__":
    main()
