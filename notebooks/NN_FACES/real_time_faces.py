from modules import keet_database as kdb
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
from modules.faces.face_positions import FaceMeshDetector
from modules.config_camera import CameraHandler




##########* LOAD THE MODEL ################################
loader = ModelLoaderFace(model_name='face_model_GERARDO.h5', scaler_name='scaler_faces_GERARDO.pkl')
model = loader.load_face_model()
scaler = loader.load_face_scaler()
dict_labels = {0: 'FELIZ', 1: 'NEUTRAL', 2: 'SORPRESA', 3: 'TRISTE'}
# top_features = pd.read_csv('data\\features\\selected_index_faces.csv')
# top = top_features['Selected_Features'].to_list() 

#####* Camera configuration ################################
camera = CameraHandler(camera_index=0, width_screen=1280, height_screen=720) ### 0 is the default camera, 1 is the external camera

camera.set_resolution(camera.width_screen, camera.height_screen) ### Set the resolution of the window of the frame
width, height = camera.get_resolution() ### Get the resolution of the camera
# Time and prediction variables
start_time = time.time()
delay_time = 1  # Delay between predictions in seconds
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
    ret, frame = camera.get_frames()
    
    detector = FaceMeshDetector(camera=camera,face_mesh=face_mesh_images)
    
    _, raw_positions_x, raw_positions_y , flag_face = detector.process_frame(frame)
    

    # Calculate FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    # Make a prediction at regular intervals
    if current_time - start_time >= delay_time:  
        if flag_face == 1:
            
            max_x = np.max(raw_positions_x)
            min_x = np.min(raw_positions_x)
            max_y = np.max(raw_positions_y)
            min_y = np.min(raw_positions_y)
            
            positions_x = (raw_positions_x*width).astype(int)
            positions_y = (raw_positions_y*height).astype(int)
            
            roi_positions_x = positions_x*((max_x-min_x)).astype(int)
            roi_positions_y = positions_y*((max_y-min_y)).astype(int)
            data =  np.hstack((positions_x, positions_y, roi_positions_x, roi_positions_y))
            data = data.reshape(1, data.shape[0])
            data = data.astype(int)
            # print(data)
            # print('---'*30)
            # print(data.shape)
            # data = data[:, top]
            data_normalized = scaler.transform(data)
            predictions = model.predict(data_normalized, verbose=0)
            predicted_class = np.argmax(predictions, axis=1)
            # print(f'Prediction: {dict_labels[predicted_class[0]]}')
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
camera.release_camera()
cv2.destroyAllWindows()
