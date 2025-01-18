import mediapipe as mp
import cv2
import pandas as pd
import joblib
import numpy as np
import time
from modules.config_camera import CameraHandler
from modules.positions.hand_positions import *
import os

sign = 'SI'


################################
# Define the base path for the dataset
base_path = os.path.join('data', 'dataset', 'DINAMICAS',sign )  
# Check if the path exists; if not, create the directories
if not os.path.exists(base_path):
    os.makedirs(base_path)

last_element = os.listdir(base_path)
if last_element:
    last_element = sorted(last_element)[-1]
    last_element = last_element.split('.csv')
    last_element = int(last_element[0][-1])
else:
    last_element = 0
count_last_element = 1


####
df = None
count_waiting = 0
time_to_wait = 3
data = {}
n_frames = 30
contador_frames = 0
capture = False
#### Config of camera ####
camera = CameraHandler(camera_index=0, width_screen=1280, height_screen=720)  ### 0 is the default camera, 1 is the external camera
camera.set_resolution(camera.width_screen, camera.height_screen)  ### Set the resolution of the window of the frame
width, height = camera.get_resolution()  ### Get the resolution of the camera
print('camera resolution', width, height)

# Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Initialize frame counter
frame_counter = 0

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
        ###calculare max and min of x and y
        min_x = np.min(raw_x*width)
        min_y = np.min(raw_y*height)
        
        
        cv2.putText(frame, "Hand detected, press 1 to capture dynamic", (20, 100), cv2.FONT_HERSHEY_PLAIN, 2.5, (0, 255, 0), 2)
        
        positions_x = (raw_x*width).astype(int)
        positions_y = (raw_y*height).astype(int)
        roi_positions_x = (positions_x-min_x).astype(int)
        roi_positions_y = (positions_y-min_y).astype(int)
        
        
        if capture == True:
            if count_waiting == 0:
                start = time.time()
                count_waiting = 1
                data = pd.DataFrame( columns=['cx', 'cxROI', 'cy', 'cyROI'])
                df = {}
            
            end = time.time()
            if end - start < time_to_wait:
                cv2.putText(frame, f'{time_to_wait - abs(end- start):.2f}', (int(camera.width_screen*0.4),int(camera.height_screen*0.5)), cv2.FONT_HERSHEY_PLAIN, 4, (0, 255, 0), 2)
                
            
            elif end - start > time_to_wait:
                
                contador_frames += 1
                cv2.putText(frame, f'frames: {contador_frames}/{n_frames}', (int(camera.width_screen*0.4),int(camera.height_screen*0.5)), cv2.FONT_HERSHEY_PLAIN, 4, (0, 255, 0), 2)
                
                df['cx'] = positions_x
                df['cy'] = positions_y
                df['cxROI'] = roi_positions_x
                df['cyROI'] = roi_positions_y
                df = pd.DataFrame(df)
                data = pd.concat([data, df], axis=0)
                
                
                if contador_frames == n_frames:
                    cv2.putText(frame, f'frames saved', (int(camera.width_screen*0.5),int(camera.height_screen*0.5)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                    capture = False
                    count_waiting = 0
                    contador_frames = 0
                    # data.drop(list(range(0,2
                    # 1)),inplace=True)
                    data.to_csv(base_path + '\\'+ f'{sign + str(last_element+ count_last_element)}.csv', index=False)
                    count_last_element += 1
                    
        
        
    
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


