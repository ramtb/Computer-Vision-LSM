from modules import keet_database as kdb
import mediapipe as mp
import cv2
import pandas as pd
import joblib
import numpy as np
import time
import h5py as h5
from modules.config_camera import CameraHandler
from modules.faces.face_positions import FaceMeshDetector, draw_regions
import os
##### Path to save the data ####
datas = []
number_data = 750


#### Config of camera ####
camera = CameraHandler(camera_index=1, width_screen=1280, height_screen=720) ### 0 is the default camera, 1 is the external camera

camera.set_resolution(camera.width_screen, camera.height_screen) ### Set the resolution of the window of the frame
width, height = camera.get_resolution() ### Get the resolution of the camera
print('camera resolution',width, height)

# variables
pTime = 0
time_to_wait = 3
start_time = 0
what_region = 0
emotions = [ 'FELIZ', 'NEUTRAL', 'SORPRESA', 'TRISTE']
what_emotion = 0
n_data = 0
contador = 0
df = {'cx':[], 'cy':[]}
df = pd.DataFrame(df)
positions = []
data={}
# print(data)
#####
# Mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh_images = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
current_time = 0
# Main loop
while True:
    
    # Capture frame
    ret, frame = camera.get_frames()
    
    detector = FaceMeshDetector(camera=camera,face_mesh=face_mesh_images)
    
    _, raw_positions_x, raw_positions_y , is_there_face = detector.process_frame(frame)
    
    #### draw the lines of four regions###
    # Draw the region lines
    draw_regions(frame, camera)
    font = cv2.FONT_HERSHEY_SIMPLEX ; thickness = 2
    if is_there_face:
        
        cv2.putText(frame, f'Colocate en la region {what_region+1}' , (int(camera.width_screen*0.2),int(camera.height_screen*0.6)), font, 1, (0,255,0), thickness)
        cv2.putText(frame, f'Haz la emocion: {emotions[what_emotion]}' , (int(camera.width_screen*0.2),int(camera.height_screen*0.7)), font, 1, (0,255,0), thickness)
            
            
        #### Make the regions ####
        max_x = np.max(raw_positions_x)
        min_x = np.min(raw_positions_x)
        max_y = np.max(raw_positions_y)
        min_y = np.min(raw_positions_y)
        ####################################### First region ###############################################################
        
        if max_x < 0.5 and max_y < 0.5:
            # print('First region')
            if what_region == 0 and (abs(start_time - current_time)) < time_to_wait:
                if contador == 0:
                    start_time = time.time()
                    contador += 1
                current_time = time.time()
                cv2.putText(frame, f'{time_to_wait - abs(start_time- current_time):.2f}', (int(camera.width_screen*0.25),int(camera.height_screen*0.25)), font, 4, (0,255,0), thickness)
                
            if (abs(start_time - current_time) > time_to_wait) and what_region == 0:  
                n_data += 1
                print(n_data)
                cv2.putText(frame, f'n_data: {n_data}', (int(camera.width_screen*0.25),int(camera.height_screen*0.25)), font, 4, (0,255,0), thickness)
                positions_x = (raw_positions_x*width).astype(int)
                positions_y = (raw_positions_y*height).astype(int)
                roi_positions_x = positions_x*((max_x-min_x)).astype(int)
                roi_positions_y = positions_y*((max_y-min_y)).astype(int)
                positions = np.hstack((positions_x, positions_y,roi_positions_x,roi_positions_y)) ##### cx, cy concatenated 
                positions = positions.reshape(positions.shape[0], 1)
                datas.append(positions)
                
                
                
                
                if n_data == number_data:
                    what_region += 1
                    n_data = 0
                    current_time = 0

                    datas = np.concatenate(datas, axis=1)
                    data[emotions[what_emotion]] = datas
                    print('Change region')
                    print(data[emotions[what_emotion]].shape)
                    datas = []
                    current_time, start_time,contador = 0, 0,0
                    
                 
            
            fist_region = True
            
            
        #### Second region ####
        elif max_x > 0.5 and max_y < 0.5:
            
            if what_region == 1 and (abs(start_time - current_time)) < time_to_wait:
                if contador == 0:
                    start_time = time.time()
                    contador += 1
                current_time = time.time()
                cv2.putText(frame, f'{time_to_wait - abs(start_time- current_time):.2f}', (int(camera.width_screen*0.25),int(camera.height_screen*0.25)), font, 4, (0,255,0), thickness)
                
            if (abs(start_time - current_time) > time_to_wait) and what_region == 1:  
                n_data += 1
                print(n_data)
                cv2.putText(frame, f'n_data: {n_data}', (int(camera.width_screen*0.25),int(camera.height_screen*0.25)), font, 4, (0,255,0), thickness)
                positions_x = (raw_positions_x*width).astype(int)
                positions_y = (raw_positions_y*height).astype(int)
                roi_positions_x = positions_x*((max_x-min_x)).astype(int)
                roi_positions_y = positions_y*((max_y-min_y)).astype(int)
                positions = np.hstack((positions_x, positions_y,roi_positions_x,roi_positions_y)) ##### cx, cy concatenated 
                positions = positions.reshape(positions.shape[0], 1)
                datas.append(positions)
                
                
                
                
                if n_data == number_data:
                    what_region += 1
                    n_data = 0
                    current_time = 0

                    datas = np.concatenate(datas, axis=1)
                    data[emotions[what_emotion]] = np.hstack((data[emotions[what_emotion]], datas))
                    print('Change region')
                    print(data[emotions[what_emotion]].shape)
                    
                    
                    datas = []
                    current_time, start_time,contador = 0, 0,0
            
            
            
            # print('Second region')
            second_region = True
            
            
        #### Third region ####
        elif max_x < 0.5 and max_y > 0.5:
            
            if what_region == 2 and (abs(start_time - current_time)) < time_to_wait:
                if contador == 0:
                    start_time = time.time()
                    contador += 1
                current_time = time.time()
                cv2.putText(frame, f'{time_to_wait - abs(start_time- current_time):.2f}', (int(camera.width_screen*0.25),int(camera.height_screen*0.25)), font, 4, (0,255,0), thickness)
                
            if (abs(start_time - current_time) > time_to_wait) and what_region == 2:  
                n_data += 1
                print(n_data)
                cv2.putText(frame, f'n_data: {n_data}', (int(camera.width_screen*0.25),int(camera.height_screen*0.25)), font, 4, (0,255,0), thickness)
                positions_x = (raw_positions_x*width).astype(int)
                positions_y = (raw_positions_y*height).astype(int)
                roi_positions_x = positions_x*((max_x-min_x)).astype(int)
                roi_positions_y = positions_y*((max_y-min_y)).astype(int)
                positions = np.hstack((positions_x, positions_y,roi_positions_x,roi_positions_y)) ##### cx, cy concatenated 
                positions = positions.reshape(positions.shape[0], 1)
                datas.append(positions)
                
                
                
                
                if n_data == number_data:
                    what_region += 1
                    n_data = 0
                    current_time = 0

                    datas = np.concatenate(datas, axis=1)
                    data[emotions[what_emotion]] = np.hstack((data[emotions[what_emotion]], datas))
                    
                    print('Change region')
                    
                    print(data[emotions[what_emotion]].shape)
                    datas = []
                    current_time, start_time,contador = 0, 0,0
            
            
            
            
            
            # print('Third region')
            third_region = True
        
        #### Fourth region ####
        elif max_x > 0.5 and max_y > 0.5:
            
            
            if what_region == 3 and (abs(start_time - current_time)) < time_to_wait:
                if contador == 0:
                    start_time = time.time()
                    contador += 1
                current_time = time.time()
                cv2.putText(frame, f'{time_to_wait - abs(start_time- current_time):.2f}', (int(camera.width_screen*0.25),int(camera.height_screen*0.25)), font, 4, (0,255,0), thickness)
                
            if (abs(start_time - current_time) > time_to_wait) and what_region == 3:  
                n_data += 1
                print(n_data)
                cv2.putText(frame, f'n_data: {n_data}', (int(camera.width_screen*0.25),int(camera.height_screen*0.25)), font, 4, (0,255,0), thickness)
                positions_x = (raw_positions_x*width).astype(int)
                positions_y = (raw_positions_y*height).astype(int)
                roi_positions_x = positions_x*((max_x-min_x)).astype(int)
                roi_positions_y = positions_y*((max_y-min_y)).astype(int)
                positions = np.hstack((positions_x, positions_y,roi_positions_x,roi_positions_y)) ##### cx, cy concatenated 
                positions = positions.reshape(positions.shape[0], 1)
                datas.append(positions)
                
                
                
                
                if n_data == number_data:
                    what_region = 0
                    n_data = 0
                    current_time = 0
                    datas = np.concatenate(datas, axis=1)
                    data[emotions[what_emotion]] = np.hstack((data[emotions[what_emotion]], datas))
                    
                    print('Change region')
                    datas = []
                    current_time, start_time,contador = 0, 0,0
                    
                    print(data[emotions[what_emotion]].shape)
                    what_emotion += 1
                    
                    
                    
            # print('Fourth region')
            fourth_region = True
        
    
    
    # Calculate FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)


  
    # Show the resulting frame
    cv2.imshow('Face Detection', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if what_emotion == len(emotions):
        break
# print(data)
# Release the capture and close windows
camera.release_camera()
cv2.destroyAllWindows()

for emotion in emotions:
    print(f'{emotion}: {data[emotion].shape}')

option = input('Do you want to save the data? [y/n]: ')
if option == 'y':
    save_path_model =f'data\\features\\positions_FACES_GERARDO.h5'
    save_path_model = save_path_model.replace('\\', os.path.sep) ###* Replace the backslash with the correct separator
    print('save_path:', save_path_model)
    with h5.File(save_path_model, 'w') as h5file:
        for emotion, positions in data.items():
            grp = h5file.create_group(emotion)
            grp.create_dataset('positions', data=positions)
else:
    print('Data not saved.')