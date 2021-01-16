import os
import av
import cv2
import PIL
import time
import traceback
import numpy as np
import streamlit as st
from pathlib import Path
from aiortc.contrib.media import MediaPlayer
from retinaface.pulse_retina import PulseMonitor
from threading import current_thread
import altair as alt
import pandas as pd

from streamlit_webrtc import ClientSettings, VideoTransformerBase, WebRtcMode, webrtc_streamer

HERE = Path(__file__).parent

KEY_CONTEXT = st.report_thread.REPORT_CONTEXT_ATTR_NAME
main_context = getattr(current_thread(), KEY_CONTEXT)

def main():
    st.header("Pulse detector demo")
    source = pd.DataFrame({
        'x': np.arange(100),
        'y': np.zeros(100)
    })
    bars = alt.Chart(source).mark_line().encode(
    x=alt.X('x',axis=alt.Axis(title='Time (s)')),
    y=alt.Y('y',axis=alt.Axis(title='BPM'))
    ).properties(width=650, height=400)

    bar_plot = st.altair_chart(bars)

    get_pulsemonitor_frames()

def plot_bpm(bpm):
    x = np.arange(min([100, len(bpm)]))
    y = bpm[-100:] if len(bpm)>100 else bpm
    source = pd.DataFrame({
        'x': x,
        'y': y
    })
    return alt.Chart(source).mark_line().encode(
        x=alt.X('x',axis=alt.Axis(title='Time (s)'), 
        y=alt.Y('y',axis=alt.Axis(title='BPM')))).properties(width=650, height=400)

def get_pulsemonitor_frames():
    class NNVideoTransformer(VideoTransformerBase):
        def __init__(self) -> None:
            self.processor = PulseMonitor()
            self.bpm = []

        def transform(self, frame: av.VideoFrame) -> np.ndarray:
            try:
                image = frame.to_ndarray(format="bgr24")
                annotated_image, current_bpm = self.processor.process_frame(image)
                self.bpm.append(current_bpm)
                ''' temp disable
                thread = current_thread()
                if getattr(thread, KEY_CONTEXT, None) is None:
                    setattr(thread, KEY_CONTEXT, main_context)
                if len(self.bpm) % 10 == 0:
                    bars = plot_bpm(self.bpm)
                    time.sleep(0.01)
                    bar_plot.altair_chart(bars)
                '''
                return annotated_image
            except Exception as e:
                traceback.print_exc()
                return image

    webrtc_ctx = webrtc_streamer(key="loopback", mode=WebRtcMode.SENDRECV, client_settings=WEBRTC_CLIENT_SETTINGS, video_transformer_factory=NNVideoTransformer, async_transform=True,)

WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)

if __name__ == "__main__":
    main()
