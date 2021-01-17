# Pulse
Video Heartrate Monitor

- TODO: write description & explanations


# Generate Executable
```
/> pyinstaller --add-data "retinaface/;retinaface" PulseIT.py --clean

/> cd dist/PulseIT

/> PulseIT.exe
```

# Local Deployment

### Setup
```
/> git clone https://github.com/Greg-Tarr/Pulse

/> cd Pulse

/> pip install -r requirements.txt

/> # install torch && torchvision from https://pytorch.org/get-started/locally/
```

### Python
```
/> python3 PulseIT.py
```

### Webapp
```
/> streamlit run app.py
```
