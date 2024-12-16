import modules.start as st, modules.tracking as tr, modules.reset as res, modules.show as sh
import cv2
import pandas as pd
import tensorflow as tf
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import numpy as np
import time 
from modules.loaders import ModelLoaderSigns




############################################################################################################*

##########* LOAD THE MODEL ################################
loader = ModelLoaderSigns(model_name='all_statics_model.h5', scaler_name='all_statics_scaler.pkl')
model = loader.load_sign_model()
scaler = loader.load_sign_scaler()

dict_labels = {0: 'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 9:'L', 10:'M', 11:'N', 12:'O', 13:'P', 14:'R', 15:'S', 16:'T', 17:'U', 18:'V', 19:'W', 20:'Y'}
start_time = time.time()
delay_time = 0.5
predicted = False
#########* CAMERA SETTINGS ###########

cap, width, height = st.camera_settings(width_cam= 1280, height_cam= 720, camera=1) #* Width and height of the camera
                                                                                    #* 0 for the default camera, 1 for the external camera	 

#########* PARAMETERS ###########

imgformat, dataformat, tiempo_de_espera, ventana_de_tiempo = st.format(imgformat = 'jpg', dataformat= 'csv', waiting_time= 3, record_time = 2)

##########* Begin parameters ################# 

time_frames, t, t1, timeflag = st.time_set()

num_hand = 1

mpHands, hands, mpDraw = st.hand_medipip(num_hand)

window_move = st.wind_move(roi1_x=0.1, roi1_y=0.4, roi2_x=0.1, roi2_y=0.6)

cTime, pTime, fps, Ts, time_frames = st.frame_settings()

#*FIRTS ARGUMENT FOR ROI 1 AND THE SECOND FOR ROI 2

###################* While loop for tracking    #################  

while True:
    
    current_time = time.time()
    
    ret, frame, frame_copy, frame_gray, frame_equali, results = tr.read_frames(cap,hands,equali=True)
    #print(frame.shape)
    
    roi_save, save_len, point_save, lm_x_h1, lm_y_h1, lm_x_h2, lm_y_h2, lm_x_h1_roi, lm_y_h1_roi, lm_x_h2_roi, lm_y_h2_roi, flag = tr.process_hand_landmarks(frame_equali= frame_equali,
                                                    results= results, width= width, height= height, t= t,tiempo_de_espera= tiempo_de_espera
                                                    ,save_len = None,print_lm=False, size_roi = 0.087, point_save={})       
    if current_time - start_time >= delay_time:  
        if flag == 1:
            data =  [ np.reshape(np.array(list(lm_x_h1.values())),(21,1))
                     , np.reshape(np.array(list(lm_x_h1_roi.values())),(21,1)), 
                      np.reshape(np.array(list(lm_y_h1.values())),(21,1)), 
                      np.reshape(np.array(list(lm_y_h1_roi.values())),(21,1)),]      
            data = np.concatenate(data,axis=1)
            data = data.reshape(1, 84)
            data_normalized = scaler.transform(data)
            predictions = model.predict(data_normalized, verbose=1)
            predicted_class = np.argmax(predictions, axis=1)
            # print(f'Predicción: {dict_labels[predicted_class[0]]}') 
            predicted = True
        start_time = current_time      
    if predicted == True and point_save != {}:
        cv2.putText(frame,dict_labels[predicted_class[0]], (point_save['h1_x_max'], point_save['h1_y_min']-40), cv2.FONT_HERSHEY_SIMPLEX, 2, (50, 50, 200), 3)
         
    #*######### ENDS IF ##############
        
    cTime, fps, Ts, pTime, time_frames = tr.ends_if(cTime, fps, Ts, pTime, time_frames)
        

    ################* DRAW RECTANGULOS and text ###############

    tr.draw_text_and_rectangles(point_save, frame, width, height, fps, draw_rectangules=True,draw_text=True)

    ##############* SHOW THE FRAMES #############

    SAVED = sh.main_show(frame = frame, SAVED = None, width= width, height=height, roi_save= roi_save, window_move= window_move, df = None, RECORDING = None, t1 = None, save_len = None)
        
    #####* RESET THE LIST ########

    roi_save, point_save= res.reset_save(roi_save)

    ###* EXIT
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break   #press q for exit
cap.release()
cv2.destroyAllWindows()
