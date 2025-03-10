import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque

# Load the trained model
model_path = "C:/Users/lenovo/Downloads/my_model_v1.keras"
model = load_model(model_path)

IMG_SIZE = 128
class_labels = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
    'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'
]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

# Padding value (adjust as needed)
PADDING = 30  # Add 20 pixels padding around the hand

# Prediction buffer for majority voting
BUFFER_SIZE = 5  # Number of predictions to consider
prediction_buffer = deque(maxlen=BUFFER_SIZE)

def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    normalized_frame = resized_frame / 255.0
    input_frame = np.expand_dims(normalized_frame, axis=0)
    return input_frame

def get_majority_vote(buffer):
    """Returns the most frequent class label in the buffer."""
    if not buffer:
        return None
    counts = np.bincount(buffer)  # Count occurrences of each class index
    return np.argmax(counts)  # Return the index of the most frequent class

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = frame.shape
            x_min, y_min, x_max, y_max = w, h, 0, 0

            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                x_min, y_min = min(x_min, x), min(y_min, y)
                x_max, y_max = max(x_max, x), max(y_max, y)

            # Add padding to the bounding box
            x_min_pad = max(0, x_min - PADDING)
            y_min_pad = max(0, y_min - PADDING)
            x_max_pad = min(w, x_max + PADDING)
            y_max_pad = min(h, y_max + PADDING)

            # Ensure valid bounding box dimensions
            if x_max_pad <= x_min_pad or y_max_pad <= y_min_pad:
                continue  # Skip invalid regions

            # Crop padded hand region
            hand_region = frame[y_min_pad:y_max_pad, x_min_pad:x_max_pad]

            # Convert BGR to RGB
            hand_rgb = cv2.cvtColor(hand_region, cv2.COLOR_BGR2RGB)
            
            # Preprocess and predict
            input_data = preprocess_frame(hand_rgb)
            predictions = model.predict(input_data, verbose=0)
            predicted_class_index = np.argmax(predictions)
            predicted_class_label = class_labels[predicted_class_index]

            # Add the current prediction to the buffer
            prediction_buffer.append(predicted_class_index)

            # Get the majority vote from the buffer
            majority_class_index = get_majority_vote(prediction_buffer)
            if majority_class_index is not None:
                stable_prediction_label = class_labels[majority_class_index]
            else:
                stable_prediction_label = "Uncertain"

            # Display the stable prediction
            cv2.putText(frame, f"Prediction: {stable_prediction_label}", 
                        (x_min_pad, y_min_pad - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow('Hand Gesture Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()