# Lane Line Detection

## 🚗 Project Overview

This project implements a **lane line detection system in Python** using **computer vision techniques** (OpenCV). The system processes road images or video frames and highlights the road lane boundaries. Lane line detection is a fundamental task in developing **driver assistance systems (ADAS)** and autonomous driving features. :contentReference[oaicite:1]{index=1}

This repository contains:
- A **GUI interface** (in `gui.py`) for testing lane detection
- A **core detection logic** in `main.py`
- Sample **videos and debug images**

---

## 🎯 Use Case / Why It Matters

Lane detection is used in many real-world applications, such as:  
✔ Advanced Driver Assistance Systems (ADAS)  
✔ Lane departure warnings  
✔ Road navigation systems  
✔ Autonomous and semi-autonomous vehicles

Accurately detecting lane lines helps a vehicle understand the road geometry and maintain proper lane positioning. :contentReference[oaicite:2]{index=2}

---

## 🛠️ Technologies & Libraries

This project uses:
- **Python 3**
- **OpenCV** — for image and video processing
- **NumPy** — for array manipulation
- Other standard Python libraries

---

## 📂 Repository Contents

├── main.py # Core lane detection logic
├── gui.py # Simple graphical interface
├── challenge1.mp4 # Sample test video
├── challenge2.mp4 # Sample test video
├── challenge3.mp4 # Sample test video
├── debug_out.png # Example output image
├── README.md # This file
└── ... # (other files like .gitignore, venv excluded)


---

## ▶️ How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/akshtyadv/Lane-Line-Detection.git
cd Lane-Line-Detection
```


2️⃣ Install Dependencies

Make sure you have Python 3 installed.

- Then install required libraries:
```bash

pip install opencv-python numpy
```


- You can also create a virtual environment first (recommended):
```bash
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows
pip install opencv-python numpy
```

▶️ Run via Python (Terminal)

To run the core detection script:
```bash

python main.py
```

You may need to update the video file path inside main.py to point at one of the challenge*.mp4 videos if required.

🖱️ Run with GUI

To start the GUI interface:
```bash
python gui.py
```


This will launch a simple window where you can:

- Load a video

- See real-time lane detection visualization
