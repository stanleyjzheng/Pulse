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

# Why Did We Work On This Project?

Knowing your heart rate is important because the heart's function is so crucial. Namely, the heart is what's responsible for circulating oxygen and nutrient-rich blood throughout your body. When it's not working as expected, basically everything in the body is affected. One way to gauge heart health — really, your health — is by your heart rate. 

Measuring your resting heart rate (RHR), which is the number of heart beats per minute while you’re at rest, is a real-time snapshot of how your heart muscle is doing. A normal heart rate for adults generally falls between 60 to 100 beats per minute. A RHR that's higher than 100 can indicate anemia or acrdiomyopathy, and a RHR that's lower than 60 may be a sign of hypothyroidism, heart disease, high levels of potassium in the blood, or certain infections. At the same time, someone may have a slower heart rate due to being physically fit or pregnant, using a medication, or experiencing sleep patterns. What's important to recognize here is that a healthy heart rate will vary depending on the situation, because there are no hard numbers to go by.

Besides measuring heart health, there are other use cases for monitoring your heart rate, including but not limited to:
* Training and Fitness Optimization
  * Many top athletes and individuals who regularly exercise are interested in optimizing their training and fitness by understanding how their body responds to training and recovers
* Improving Meditation
  * A lower heart rate is one of the positive physiological effects of meditation and can indicate how effect a practice is
* Anxiety Treatment
  * Anxiety disorders are the most common psychiatric disorders today that have been shown to increase the risk of heart disease, so finding a slower heart rate could help with identifying and getting checked out sooner 

# Photoplethysmography: The Method Our Project Focuses On

Traditional heart rate monitoring is done through an optical sensor on the back of a smart watch, or through a chest strap. However, the accuracy and reliability of optical heart rate measurement varies from person to person and may not work at all with certain types of activities or sports. Currently, the best wrist heart rate measurements stay accurate 80% of the time within 10% of the chest-measured heart rate. Chest straps are much more accurate, but their cumbersome nature is not desirable. In addition, the cost of a fitness watch or chest is unjustifiable for many. Recent developments in machine learning have allowed for a new way to measure heart rate more accurately than these existing methods through video alone. As such, we developed a web app to provide a way measure heart rate or help supplement the tools out there, given the meteoric rise in telehealth due to the pandemic and its potential use cases.

The technique of photoplethysmography refers to the extraction of heart rate optically. Loosely interpreted from its greek roots, photoplethysmography refers to the recording of swellings as shown in the light. In this context, the swellings come from blood being pumped from the heart to every part of the body through a system of blood vessels called the circulatory system.

Every time a person's heart beats, the amount of blood that reaches the capillaries in the fingers and face swells and then recedes. Because blood absorbs light, we can measure heart rate by detecting ebb and flow just by using the flash of a camera phone to illuminate the skin and create a reflection. Since a video's frames per second are consistent, this can be used to extremely accurately measure heart rate.

It's hard to beat the convenience that this low-cost, non-invasive and safe video technique provides. 

To learn more about our project along with visualizing some key concepts and terminology, check out our website here: <http://pulse-it.glitch.me/>

# How We Went About Measuring Heart Rate

Sensing pulse rate remotely can be done using a face detector to crop a region of the forehead out from a live webcam feed. This patch of skin is then analysed for repetitions that are indicative of a heartbeat. 

Our system uses the RetinaFace mobilenet face detector as it is extremely fast while being incredibly robust. This fast face detector also runs real-time on an 8 core CPU. This pipeline was deployed as a streamlit webpage which uses WebRTC to transmit frames to the server which processes the stream.

# Team
* Gregory Tarr
* Anita Yip
* Stanley Zheng
