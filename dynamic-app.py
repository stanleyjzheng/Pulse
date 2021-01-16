import os
import av
import cv2
import PIL
import time
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

source = pd.DataFrame({
    'x': np.arange(100),
    'y': np.zeros(100)
})
bars = alt.Chart(source).mark_line().encode(
x=alt.X('x',axis=alt.Axis(title='Time (s)')),
y=alt.Y('y',axis=alt.Axis(title='BPM'))
).properties(width=650, height=400)

bar_plot = st.altair_chart(bars)

def main():
    st.header("Pulse detector demo")
    get_pulsemonitor_frames()

def plot_bpm(bpm):
    x = np.arange(min([100, len(bpm)]))
    y = bpm[-100:] if len(bpm)>100 else bpm
    source = pd.DataFrame({
        'x': x,
        'y': y
    })
    return alt.Chart(source).mark_line().encode(x=alt.X('x',axis=alt.Axis(title='Time (s)')), y=alt.Y('y',axis=alt.Axis(title='BPM'))).properties(width=650, height=400)

bpm = []
processor = PulseMonitor()

def get_pulsemonitor_frames():
    webrtc_ctx = webrtc_streamer(
        key="loopback",
        mode=WebRtcMode.SENDONLY,
        client_settings=WEBRTC_CLIENT_SETTINGS,
    )

    if webrtc_ctx.video_receiver:
        image_loc = st.empty()
        while True:
            try:
                frame = webrtc_ctx.video_receiver.frames_queue.get(timeout=1)
            except queue.Empty:
                print("Queue is empty. Stop the loop.")
                webrtc_ctx.video_receiver.stop()
                break

            image = frame.to_ndarray(format="bgr24")
            annotated_image, current_bpm = processor.process_frame(image)
            bpm.append(current_bpm)
            if len(bpm) % 5 == 0:
                bars = plot_bpm(bpm)
                bar_plot.altair_chart(bars)
                image_loc.image(annotated_image, channels="BGR")

WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)

if __name__ == "__main__":
    main()
