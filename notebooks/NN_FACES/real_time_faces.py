
import mediapipe as mp
import cv2
import time
import pandas as pd
import tensorflow as tf
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import numpy as np
from modules.loaders import ModelLoaderFace




##########* LOAD THE MODEL ################################
loader = ModelLoaderFace(model_name='face_model.h5', scaler_name='scaler_faces.pkl')
model = loader.load_face_model()
scaler = loader.load_face_scaler()
dict_labels = {0: 'ENOJO', 1: 'FELIZ', 2: 'NEUTRAL', 3: 'SORPRESA', 4: 'TRISTE'}
top_features = pd.read_csv('data\\features\\selected_index_faces.csv')
top = top_features['Selected_Features'].to_list() 

# Camera configuration
cap = cv2.VideoCapture(1) #### 0 for the default camera, 1 for the external camera
width, height = 1280, 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Time and prediction variables
start_time = time.time()
delay_time = 0.5
predicted = False
prediction = ""

# FPS variables
pTime = 0

# Mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh_images = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Main loop
while True:
    current_time = time.time()
    # Capture frame
    ret, frame = cap.read()
    
    if not ret:
        print('Error capturing frame')
        break
    
    # Convert frame to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_mesh_results = face_mesh_images.process(rgb_frame)
    
    if face_mesh_results.multi_face_landmarks:
        lm = face_mesh_results.multi_face_landmarks[0]  # Only use the first detected face
        landmarks = lm.landmark
        # Extract landmark positions
        positions_x = np.array([landmark.x for landmark in landmarks])
        positions_y = np.array([landmark.y for landmark in landmarks])

        # Calculate rectangle around the face
        min_x, min_y = np.min(positions_x), np.min(positions_y)
        max_x, max_y = np.max(positions_x), np.max(positions_y)

        # Draw the rectangle around the face
        cv2.rectangle(frame, (int(min_x * width), int(min_y * height)),
                      (int(max_x * width), int(max_y * height)), (255, 0, 0), 2)
        
        flag_face = 1
    else:
        flag_face = 0

    # Calculate FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    # Make a prediction at regular intervals
    if current_time - start_time >= delay_time:  
        if flag_face == 1:
            data = np.concatenate([
                np.reshape(positions_x, (468, 1)),
                np.reshape(positions_y, (468, 1))
            ], axis=1)
            data = data.reshape(1, 936)
            data = data[:, top]
            data_normalized = scaler.transform(data)
            predictions = model.predict(data_normalized, verbose=0)
            predicted_class = np.argmax(predictions, axis=1)
            print(f'Prediction: {dict_labels[predicted_class[0]]}')
            prediction = dict_labels[predicted_class[0]]
            predicted = True  # A prediction has been made
        start_time = current_time

    # Display the prediction if made
    if predicted:
        cv2.putText(frame, prediction, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (50, 50, 200), 3)

    # Show the resulting frame
    cv2.imshow('Face Detection', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
