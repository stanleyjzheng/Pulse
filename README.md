# Pulse
Video Heartrate Monitor

- TODO: write description & explanations


# Generate Executable
```python
/> pyinstaller --add-data "retinaface/;retinaface" PulseIT.py --clean

/> cd dist/PulseIT

/> PulseIT.exe
```

# Local Deployment

### Setup
```python
/> git clone https://github.com/Greg-Tarr/Pulse

/> cd Pulse

/> pip install -r requirements.txt

/> # install torch && torchvision from https://pytorch.org/get-started/locally/
```

### Python
```python
/> python3 PulseIT.py
```

### Webapp
```python
/> streamlit run app.py
```

Heart rate is a crucial metric; a high heart rate can indicate anemia or acrdiomyopathy, and a low heart rate can mean hypothyroidism or heart disease. 
However, for most, it's a great way to understand how extraneous exercise is.
Traditional heart rate monitoring is done through an optical sensor on the back of a smart watch, or through a chest strap. 
However, the accuracy and reliability of optical heart rate measurement varies from person to person and may not work at all with certain types of activities or sports. 
Currently, the best wrist heart rate measurements stay 80% of the time within 10% of the chest-measured heart rate. 
Chest straps are much more accurate, but their cumbersome nature is not desirable. 
In addition, the cost of a fitness watch or chest is unjustifiable for many.
Recent developments in machine learning have allowed for a new way to measure heart rate more accurately than these existing methods through video alone. 

Photoplethysmography is the extraction of heart rate through video. Loosely interpreted, the word refers to the recording of swellings as shown in the light. 
In this context, the swellings come from blood being pumped from the heart to every part of the body through a system of blood vessels called the circulatory system.

Every time a person's heart beats, the amount of blood that reaches the capillaries in the fingers and face swells and then recedes. Because blood absorbs light, apps on a phone can measure heart rate by detecting ebb and flow just by using the flash of a camera phone to illuminate the skin and create a reflection.

As a caveat, it shouldn't come as a surprise that DIY heart rate tracking apps don't perform consistently as well as clinical grade equipment (e.g. electrocardiogram) and methods (e.g. fingertip pulse oximetry) do. Studies found that heart rate readings generated by apps were off by 20+ beats per minute in over 20% of measurements. 

So much goes on underneath the skin that is hard to see with the naked eye. While it's much easier for computers to do the same, they have their own challenges to overcome, too.
If you're just looking for a high-level overview of your heart rate, it's hard to beat the convenience that this low-cost, non-invasive and safe video technique provides. 

This technology brings other benefits in the form of group monitoring. No longer does a fitness instructor have to guess if a large cohort is being excercised; thorough group pulse monitoring, a more tailor-made program is possible. 

Greg: 

This can be done using a face detector to crop a region of the forehead out from a live webcam feed. This patch of skin is then analysed for repetitions that are indicative of a heartbeat. 

Our system uses the RetinaFace mobilenet face detector as it is extremely fast while being incredibly robust. This fast face detector also runs real-time on an 8 core CPU. This pipeline was deployed as a streamlit webpage which uses WebRTC to transmit frames to the server which processes the stream.

Here is an example:


Do a demo at the end
