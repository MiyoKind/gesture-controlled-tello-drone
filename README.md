
# Gesture-Controlled Tello Drone

## Index
1. [Features](#Features)
2. [Requirements](#Requirements)
3. [Installation](#Installation)
4. [Usage](#Usage)
5. [Gestures](#Gestures)
6. [Contributing](#Contributing)
7. [License](#License)
8. [Acknowledgments](#Acknowledgements)

## Features
* Gesture recognition for controlling drone movements (up, down, left, right, forward, backward)
* Landing and takeoff gestures
* Flip gesture for performing flips
* Real-time display of drone camera feed and gesture recognition
* Battery level, height, and speed monitoring
* Weak Wi-Fi signal warning

## Requirements
* Python 3.7 or higher
* Tello drone
* Webcam
* TensorFlow 2.15.0
* MediaPipe 0.10.14
* comtypes

## Installation
1. Clone the repository:
```bash
git clone https://github.com/your-username/gesture-controlled-tello-drone.git
```
2. Navigate to the project directory:
```bash
cd gesture-controlled-tello-drone
```
3. Install the required packages:
```bash
pip install -r requirements.txt
```
## Usage
1. Connect your Tello drone to the computer's Wi-Fi network.
2. Run the main script:
```bash
python main.py
```
3. The program will open two windows: one for the `drone camera feed` and another for the `webcam feed` with gesture recognition.
4. Perform the gestures in front of your webcam to control the drone movements.


## Gestures
<img alt="Gesture List" src="https://ibb.co/kcfWpTV">
* **Up** 
* **Down**
* **Left**
* **Right**
* **Forward**
* **Backward** 
* **Land**
* **Takeoff**
* **Flip**
* **Turn**



## Contributing
Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License
This project is licensed under the [MIT License](https://github.com/git/git-scm.com/blob/main/MIT-LICENSE.txt).

## Acknowledgments
* [DJITelloPy](https://github.com/damiafuentes/DJITelloPy) - Python API for controlling the Tello drone.
* [MediaPipe](https://github.com/google-ai-edge/mediapipe) - Cross-platform, customizable ML solutions for live and streaming media.
* [OpenCV](https://opencv.org/) - Open-source computer vision library.
