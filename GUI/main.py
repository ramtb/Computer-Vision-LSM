import sys
import io
import os

import cv2
import pandas as pd
import tensorflow as tf
import joblib
import numpy as np
import time 
import mediapipe as mp

import modules.mod_main.basic_voice_system as bvs
import modules.mod_main.start as st, modules.mod_main.tracking as tr, modules.mod_main.reset as res, modules.mod_main.show as sh
from modules.loaders import ModelLoaderFace, ModelLoaderSigns, RelativeDirToRoot 
from modules.faces.face_positions import FaceMeshDetector
from modules.config_camera import CameraHandler
from modules.positions.hand_positions import HandDetector

from PySide6.QtWidgets import QApplication
from gui_español import *

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

##### EMOJIS for the GUI #####

emotion_emojis = {
    'FELIZ': '😊',
    'NEUTRAL': '😐',
    'SORPRESA': '😲',
    'TRISTE': '😢'
}


##########* LOAD THE MODEL ################################
loader = ModelLoaderSigns(model_name='all_statics_model2.h5', scaler_name='scaler.pkl')
model = loader.load_sign_model()
scaler = loader.load_sign_scaler()

dict_labels = {0: 'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 9:'L', 10:'M', 11:'N', 12:'O', 13:'P', 14:'R', 15:'S', 16:'T', 17:'U', 18:'V', 19:'W', 20:'Y'}
delay_time = 1.25
predicted = False
start_time = time.time()
phrase = ''
n_letters = 0
max_min = [0, 0, 0, 0]
#######* LOAD THE MODEL of faces ################################

loader_faces = ModelLoaderFace(model_name='face_model_GERARDO.h5', scaler_name='scaler_faces_GERARDO.pkl')
model_faces = loader_faces.load_face_model()
scaler_faces = loader_faces.load_face_scaler()
dict_labels_faces = {0: 'FELIZ', 1: 'NEUTRAL', 2: 'SORPRESA', 3: 'TRISTE'}
predicted_face = False

#########* CONFIGURAR GUI ###########

app = QApplication(sys.argv)

relative_dir = RelativeDirToRoot(root_dir='Computer-vision-LSM')
style_path = relative_dir.generate_path(os.path.join("GUI", "assets", "style.qss"))
with open(style_path, "r") as qss_file:
    qss_style = qss_file.read()

app.setStyleSheet(qss_style)

main = MainWindow(app)
main.show()



#########* CAMERA SETTINGS ###########


camera = CameraHandler(camera_index=0, width_screen=1280, height_screen=720) ### 0 is the default camera, 1 is the external camera

camera.set_resolution(camera.width_screen, camera.height_screen) ### Set the resolution of the window of the frame
width, height = camera.get_resolution() ### Get the resolution of the camera
print('camera resolution',width, height)  


##########* Begin parameters ################# 
### Statics Signs ###

time_frames, t= st.time_set()

num_hand = 1

mpHands = mp.solutions.hands

hands = mpHands.Hands(static_image_mode=False, max_num_hands= num_hand, min_detection_confidence=0.5, min_tracking_confidence=0.5)
detector = HandDetector(camera=camera, hands=hands)


mpDraw = mp.solutions.drawing_utils

window_move = st.wind_move(roi1_x=0.1, roi1_y=0.4, roi2_x=0.1, roi2_y=0.6)

cTime, pTime, fps, Ts, time_frames = st.frame_settings()

#*FIRTS ARGUMENT FOR ROI 1 AND THE SECOND FOR ROI 2

#####* MEDIAPIPE FACE MESH ########

mp_face_mesh = mp.solutions.face_mesh
face_mesh_images = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

############## PARAMETERS ################
completed_translations = []
###################* While loop for tracking    #################  

while main.close_all_windows == False:
    app.processEvents()
    if main.show_gui == True:
        
        
        current_time = time.time()

        ret, frame = camera.get_frames()
    
        if not ret:
            print("Failed to capture frame")
            break
        
        #######* HAND EXTRACTION ########
        _, raw_x, raw_y, _, is_there_hand = detector.process_frame(frame)
        raw_x = np.array(raw_x)
        raw_y = np.array(raw_y)
        main.gui.update_led_status(is_there_hand)
        ################* FACE MESH ############	
        face_mesh_results = face_mesh_images.process(frame)
        
        if face_mesh_results.multi_face_landmarks:
            lm = face_mesh_results.multi_face_landmarks[0] 
            landmarks = lm.landmark
            # Extraer las posiciones de los landmarks
            positions_x = np.array([landmark.x for landmark in landmarks])
            positions_y = np.array([landmark.y for landmark in landmarks])

            # Calcular el rectángulo alrededor de la cara
            min_x, min_y = np.min(positions_x), np.min(positions_y)
            max_x, max_y = np.max(positions_x), np.max(positions_y)

            # Dibujar el rectángulo alrededor de la cara
            cv2.rectangle(frame, (int(min_x * width), int(min_y * height)),
                        (int(max_x * width), int(max_y * height)), (0, 0, 255), 3)
            
            flag_face = 1
        else:
            flag_face = 0 

        #############* REAL TIME MODEL FACE #########
        if current_time - start_time >= delay_time:  
            if flag_face == 1:
                positions_x = (positions_x*width).astype(int)
                positions_y = (positions_y*height).astype(int)
                roi_positions_x = positions_x*((max_x-min_x)).astype(int)
                roi_positions_y = positions_y*((max_y-min_y)).astype(int)
                data =  np.hstack((positions_x, positions_y, roi_positions_x, roi_positions_y))
                data = data.reshape(1, data.shape[0])
                data_face = data.astype(int)
                data_normalized_face = scaler_faces.transform(data_face)
                predictions_face = model_faces.predict(data_normalized_face, verbose=0)
                predicted_class_face = np.argmax(predictions_face, axis=1)
                predictions_face = dict_labels_faces[predicted_class_face[0]]
                predicted_face = True  # Se realizó una predicción
                emotion_emoji = emotion_emojis.get(predictions_face, '')
            # start_time = current_time
        # Mostrar la predicción si se ha realizado
        # if predicted_face:
        #     cv2.putText(frame, predictions_face, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (50, 50, 200), 2)
        
        #############* REAL TIME MODEL HAND #########
        if  current_time - start_time >= delay_time:
            # print(current_time - start_time)
            if is_there_hand == True:
                # start = time.time()
                ###calculare max and min of x and y
                
                
                # Paso 1: Calcular las posiciones escaladas
                positions_x = (raw_x * width).astype(int)
                positions_y = (raw_y * height).astype(int)

                # Paso 2: Calcular los límites mínimos y máximos de las posiciones
                min_x = np.min(positions_x)
                min_y = np.min(positions_y)
                max_x = np.max(positions_x)
                max_y = np.max(positions_y)
                size_roi = 0.087  # Tamaño de la ROI (20% del tamaño de la mano)

                roi_factor_x =  raw_x*(max_y + int(height * size_roi) - (min_y - int( height* size_roi)))
                
                roi_factor_y = raw_y*(max_x + int(width* size_roi) - (min_x - int(width* size_roi)))
               
                roi_positions_x = (raw_x*(roi_factor_x)).astype(int)
                roi_positions_y = (raw_y*(roi_factor_y)).astype(int)
                if n_letters == 0:
                    phrase = ''
                    n_letters += 1
                data = [np.reshape(positions_x, (21, 1)), 
                        np.reshape(roi_positions_x, (21, 1)), np.reshape(positions_y, (21, 1)), 
                        np.reshape(roi_positions_y, (21, 1))]
            
                data = np.concatenate(data,axis=1)
                data = data.reshape(1, 84)
                # print(data)
                data_normalized = scaler.transform(data)
                predictions = model.predict(data_normalized, verbose=1)
                predicted_class = np.argmax(predictions, axis=1)
                    # print(f'Predicción: {dict_labels[predicted_class[0]]}') 
                prediction = dict_labels[predicted_class[0]]
                predicted = True
                phrase = phrase + prediction
                

                    # Add emoji
                if predicted_face:
                    phrase_with_emoji = f"{emotion_emoji} {phrase}"
                else:
                    phrase_with_emoji = phrase

                main.gui.update_text(phrase_with_emoji) # Actualizar el texto
                start_time = current_time
            else:
                if n_letters > 0:
                    bvs.sintetizar_emocion('emocion=alegria', texto = phrase )
                    completed_translations.append(phrase)  # Añadir la traducción completada a la lista
                    main.gui.update_text('Esperando ...', completed_translations[-1])  # Ac
                    predicted = False
                    n_letters = 0
                                
        # if predicted == True:
        #     cv2.putText(frame, prediction, (int(x_max), int(min_y)), cv2.FONT_HERSHEY_SIMPLEX, 2, (50, 50, 200), 3)
        #     cv2.putText(frame, phrase , (int(max_x), int(min_y)), cv2.FONT_HERSHEY_SIMPLEX, 2, (50, 50, 200), 3)
                    
            
        
        #*######### ENDS IF ##############
            
        cTime, fps, Ts, pTime, time_frames = tr.ends_if(cTime, fps, Ts, pTime, time_frames)
        

        ################* DRAW RECTANGULOS and text ###############
        if is_there_hand:
            cv2.rectangle(frame, pt1=(int(min_x), int(min_y)), pt2=(int(max_x), int(max_y)), color=(100, 100, 255), thickness=3)
            # cv2.putText(frame, "Hand detected", (20, 100), cv2.FONT_HERSHEY_PLAIN, 2.5, (0, 255, 0), 2)

        cv2.imshow('HAND Detection', frame)
        cv2.waitKey(1)

        max_min = [0,0,0,0]
    if main.destroy_gui == True:
        cv2.destroyAllWindows()

camera.release_camera()
