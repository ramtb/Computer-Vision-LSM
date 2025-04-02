import mediapipe as mp
import cv2
import pandas as pd

import numpy as np
import time
from modules.config_camera import CameraHandler
from modules.positions.hand_positions import *
from modules.loaders import ModelLoaderSigns
#### Model #######

loader = ModelLoaderSigns(model_name='dynamic_model_all.h5', scaler_name=None)
model = loader.load_sign_model()

dict_labels = {0: 'Veintitres', 1:'Bueno', 2:'Hola', 3:'Mal', 4:'No', 5:'Tengo', 6: '¿Que tal?', 
               7:'Si', 8:'Veinticuatro', 9: 'Yo soy'}
predicted = False


#### Variables ####

data = pd.DataFrame(columns=['cx', 'cxROI', 'cy', 'cyROI'])
df = {}
n_frames = 30
waiting_tine = 1 ## waiting time to start the prediction
start = time.time()
start_clock = True
#### Config of camera ####
camera = CameraHandler(camera_index=0, width_screen=1280, height_screen=720)  ### 0 is the default camera, 1 is the external camera
camera.set_resolution(camera.width_screen, camera.height_screen)  ### Set the resolution of the window of the frame
width, height = camera.get_resolution()  ### Get the resolution of the camera
print('camera resolution', width, height)

# Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Initialize frame counter
contador_frames = 0

# Main loop
pTime = 0  # previous time for FPS calculation
while True:
    # Capture frame
    ret, frame = camera.get_frames()
    
    if not ret:
        print("Failed to capture frame")
        break
    
    detector = HandDetector(camera=camera, hands=hands)
    _, raw_x, raw_y, _, is_there_hand = detector.process_frame(frame)
    raw_x = np.array(raw_x)
    raw_y = np.array(raw_y)
    
    if is_there_hand:
        start = time.time()
        ###calculare max and min of x and y
        min_x = np.min(raw_x*width)
        min_y = np.min(raw_y*height)
        
        
        cv2.putText(frame, "Hand detected", (20, 100), cv2.FONT_HERSHEY_PLAIN, 2.5, (0, 255, 0), 2)
        
        positions_x = (raw_x*width).astype(int)
        positions_y = (raw_y*height).astype(int)
        roi_positions_x = (positions_x-min_x).astype(int)
        roi_positions_y = (positions_y-min_y).astype(int)
        
        
        
            
            
            
        # if start_clock == True:
        #     start = time.time()
        #     start_clock = False
        # end = time.time()
        # if end - start > waiting_tine:
        contador_frames += 1
        cv2.putText(frame, f'frames: {contador_frames}/{n_frames}', (int(camera.width_screen*0.1),int(camera.height_screen*0.9)), cv2.FONT_HERSHEY_PLAIN, 4, (0, 255, 0), 2)
                
        df['cx'] = positions_x
        df['cy'] = positions_y
        df['cxROI'] = roi_positions_x
        df['cyROI'] = roi_positions_y
        df = pd.DataFrame(df)
        data = pd.concat([data, df], axis=0)
                
                
        if contador_frames == n_frames:
            data = data.to_numpy().flatten()
            data = data.reshape(n_frames, 84).astype('int32')
            data = np.expand_dims(data, axis=0)
            # print(data.shape,data)
            predictions = model.predict(data, verbose=1)
            predicted_class = np.argmax(predictions, axis=1)
            # print(f'Predicción: {dict_labels[predicted_class[0]]}') 
            predicted = True      
            
            contador_frames = 0
            data = pd.DataFrame(columns=['cx', 'cxROI', 'cy', 'cyROI'])
            df = {}
            
        if predicted == True:
            cv2.putText(frame,dict_labels[predicted_class[0]], (int(min_x*0.8),int(min_y*0.8)), cv2.FONT_HERSHEY_SIMPLEX, 2, (50, 50, 200), 3)
            print(f'Predicción: {dict_labels[predicted_class[0]]}')
            
            
            
                
                
                    
                    
        
        
    
    # Calculate FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    
    
    
    
    # Show the resulting frame
    cv2.imshow('HAND Detection', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    if cv2.waitKey(1) & 0xFF == ord('1') and is_there_hand:
            cv2.putText(frame, "Capturing frames", (20, 150), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            capture = True

# Release the capture and close windows
camera.release_camera()
cv2.destroyAllWindows()


