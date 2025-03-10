**README.md**

# American Sign Language (ASL) Recognition System

## Introduction
This project implements a real-time American Sign Language (ASL) recognition system using deep learning. The system detects hand gestures via webcam, processes them through a pre-trained convolutional neural network (CNN), and displays the predicted ASL alphabet character or command (e.g., "space", "del").

---

## Features
- **Real-time Detection**: Processes live webcam feed for instant gesture recognition.
- **Hand Tracking**: Uses MediaPipe to detect and track hand landmarks.
- **Prediction Stabilization**: Employs a majority voting buffer to reduce prediction flickering.
- **Confidence Display**: Shows prediction confidence scores and highlights bounding boxes based on confidence thresholds.
- **Support for 29 Classes**: Recognizes ASL letters (A-Z) and commands like "del", "nothing", and "space".

---

## Requirements
- Python 3.8+
- TensorFlow 2.x
- OpenCV (`opencv-python`)
- MediaPipe
- NumPy
- Matplotlib (optional for visualization)

Install dependencies:
```bash
pip install tensorflow opencv-python mediapipe numpy matplotlib
```

---

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo/asl-recognition.git
   cd asl-recognition
   ```

2. **Download the Pre-trained Model**:
   Place `my_model_v1.keras` in the project directory.  
   *(The model is trained on the ASL Alphabet Dataset and achieves 98.57% test accuracy.)*

---

## Usage
Run the real-time recognition script:
```bash
python asl_recognition.py
```

**Controls**:
- Press `q` to exit the webcam window.
- Adjust `PADDING` and `BUFFER_SIZE` in the script for sensitivity.

---

## Model Architecture
- **Base Model**: VGG16 (pre-trained on ImageNet) with frozen convolutional layers.
- **Custom Layers**:
  - Flatten layer.
  - Two dense layers (512 units, ReLU activation) with 50% dropout.
  - Output layer with 29 units (softmax activation).
- **Training**:
  - Trained on 128x128 RGB images.
  - Multi-GPU support for faster training.
  - Achieves **98.57% test accuracy** after 20 epochs.

---

## Performance
- **Test Accuracy**: 98.57% (on ASL Alphabet Dataset).
- **Validation Accuracy**: Peaks at 99.51% during training.
- **Inference Speed**: ~30 FPS on a GPU-enabled machine.

---

## Limitations
- **Lighting Sensitivity**: Works best in well-lit environments.
- **Hand Occlusion**: May struggle if hands are partially visible or overlapped.
- **Distance**: Optimal performance when hands are 1–2 feet from the camera.

---

## License
This project is licensed under the MIT License. See `LICENSE` for details.

---

**Demo**  
![ASL Recognition Demo](demo.gif)  
*(Example output showing real-time predictions with bounding boxes and confidence scores.)*

For questions or contributions, contact [bahlil2001@gmail.com].