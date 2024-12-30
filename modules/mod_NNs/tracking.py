import cv2
import numpy as np
import mediapipe as mp
import time
import os
import tkinter as tk
from tkinter import simpledialog
from tkinter import filedialog
import sys
import pandas as pd
import re
import platform
from typing import Dict


lm_x_h1 = {'h1_x0': 0, 'h1_x1': 0, 'h1_x2': 0, 'h1_x3': 0, 'h1_x4': 0, 'h1_x5': 0, 'h1_x6': 0, 'h1_x7': 0, 'h1_x8': 0, 'h1_x9': 0, 'h1_x10': 0, 'h1_x11': 0, 'h1_x12': 0, 'h1_x13': 0, 'h1_x14': 0, 'h1_x15': 0, 'h1_x16': 0, 'h1_x17': 0, 'h1_x18': 0, 'h1_x19': 0, 'h1_x20': 0}
lm_x_h1_roi = {'h1_x0_roi': 0, 'h1_x1_roi': 0, 'h1_x2_roi': 0, 'h1_x3_roi': 0, 'h1_x4_roi': 0, 'h1_x5_roi': 0, 'h1_x6_roi': 0, 'h1_x7_roi': 0, 'h1_x8_roi': 0, 'h1_x9_roi': 0, 'h1_x10_roi': 0, 'h1_x11_roi': 0, 'h1_x12_roi': 0, 'h1_x13_roi': 0, 'h1_x14_roi': 0, 'h1_x15_roi': 0, 'h1_x16_roi': 0, 'h1_x17_roi': 0, 'h1_x18_roi': 0, 'h1_x19_roi': 0, 'h1_x20_roi': 0}
lm_y_h1 = {'h1_y0': 0, 'h1_y1': 0, 'h1_y2': 0, 'h1_y3': 0, 'h1_y4': 0, 'h1_y5': 0, 'h1_y6': 0, 'h1_y7': 0, 'h1_y8': 0, 'h1_y9': 0, 'h1_y10': 0, 'h1_y11': 0, 'h1_y12': 0, 'h1_y13': 0, 'h1_y14': 0, 'h1_y15': 0, 'h1_y16': 0, 'h1_y17': 0, 'h1_y18': 0, 'h1_y19': 0, 'h1_y20': 0}
lm_y_h1_roi = {'h1_y0_roi': 0, 'h1_y1_roi': 0, 'h1_y2_roi': 0, 'h1_y3_roi': 0, 'h1_y4_roi': 0, 'h1_y5_roi': 0, 'h1_y6_roi': 0, 'h1_y7_roi': 0, 'h1_y8_roi': 0, 'h1_y9_roi': 0, 'h1_y10_roi': 0, 'h1_y11_roi': 0, 'h1_y12_roi': 0, 'h1_y13_roi': 0, 'h1_y14_roi': 0, 'h1_y15_roi': 0, 'h1_y16_roi': 0, 'h1_y17_roi': 0, 'h1_y18_roi': 0, 'h1_y19_roi': 0, 'h1_y20_roi': 0}
            
lm_x_h2 = {}  #* DICTIONARY FOR X POSITIONS OF LANDMARKS
lm_y_h2 = {}  #* DICTIONARY FOR Y POSITIONS OF LANDMARKS
lm_x_h2_roi = {}  #* DICTIONARY FOR X POSITIONS OF LANDMARKS
lm_y_h2_roi = {}  #* DICTIONARY FOR Y POSITIONS OF LANDMARKS

#################* TRACKING ####################

def read_frames(cap, hands, equali=True) -> tuple:

    ret, frame = cap.read()  # Leer frame de la cámara
    if ret == False:
        raise 'CamError'
    frame_copy = frame.copy()  # Copiar el frame si es válido
    frame_gray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)  # Cambiar escala de color
    
    if equali:
        frame_equali = cv2.equalizeHist(frame_gray)  # Ecualización de histograma
    else:
        frame_equali = frame_gray

    results = hands.process(frame)  # Procesar el frame con Mediapipe
    
    return ret, frame, frame_copy, frame_gray, frame_equali, results

def process_hand_landmarks( point_save, frame_equali, results, width,height, t, tiempo_de_espera, save_len, print_lm = False, size_roi = 0.087)-> tuple:
    """
    This function processes the hand landmarks.
    Parameters:
    frame: The frame from the camera.
    frame_equali: The frame with the histogram equalized.
    results: The results from the hands object.
    width: The width of the camera.
    height: The height of the camera.
    corner_x: The size of the ROI in X.
    corner_y: The size of the ROI in Y.
    t (float): The time for the ROI.
    tiempo_de_espera (int): The waiting time to capture the ROI.
    cx_saved (array): The list to save the x positions of the landmarks. shape(21,1)
    cy_saved (array): The list to save the y positions of the landmarks. shape(21,1)
    cxROI_saved (array): The list to save the x positions of the ROI. shaep (21,1)
    cyROI_saved (array): The list to save the y positions of the ROI.  shaep (21,1)
    roi_save (list): The list to save the ROIs.
    point_save (list): The list to save the points of the ROIs.
    save_len (int): The length of the saved lists.
    Returns:
    cx_saved (list): The list to save the x positions of the landmarks.
    cy_saved (list): The list to save the y positions of the landmarks.
    cxROI_saved (list): The list to save the x positions of the ROI.
    cyROI_saved (list): The list to save the y positions of the ROI.
    roi_save (list): The list to save the ROIs.
    point_save (list): The list to save the points of the ROIs.
    save_len (int): The length of the saved lists.
    
    """
    global lm_x_h1, lm_y_h1, lm_x_h2, lm_y_h2, lm_x_h1_roi, lm_y_h1_roi, lm_x_h2_roi, lm_y_h2_roi
    roi_save = [] #* SAVE THE ROIS
   
    if results.multi_hand_landmarks:  ###* if THIS OBJECT IS NOT EMPTY DO THE CONDITIONAL
        # print(results.multi_hand_landmarks)
        # print('**'*30)
        for i,handlms in enumerate(results.multi_hand_landmarks):  ###* Detects how many hands there are in the frame
            if i == 0:
                lm_h1 =  handlms.landmark #results.multi_hand_landmarks[0].landmark
                lm_x_h1 = {'h1_x0': int(lm_h1[0].x * width), 'h1_x1': int(lm_h1[1].x * width), 'h1_x2': int(lm_h1[2].x * width), 'h1_x3': int(lm_h1[3].x * width), 'h1_x4': int(lm_h1[4].x * width), 'h1_x5': int(lm_h1[5].x * width), 'h1_x6': int(lm_h1[6].x * width), 'h1_x7': int(lm_h1[7].x * width), 'h1_x8': int(lm_h1[8].x * width), 'h1_x9': int(lm_h1[9].x * width), 'h1_x10': int(lm_h1[10].x * width), 'h1_x11': int(lm_h1[11].x * width), 'h1_x12': int(lm_h1[12].x * width), 'h1_x13': int(lm_h1[13].x * width), 'h1_x14': int(lm_h1[14].x * width), 'h1_x15': int(lm_h1[15].x * width), 'h1_x16': int(lm_h1[16].x * width), 'h1_x17': int(lm_h1[17].x * width), 'h1_x18': int(lm_h1[18].x * width), 'h1_x19': int(lm_h1[19].x * width), 'h1_x20': int(lm_h1[20].x * width)}
                lm_y_h1 = {'h1_y0': int(lm_h1[0].y * height), 'h1_y1': int(lm_h1[1].y * height), 'h1_y2': int(lm_h1[2].y * height), 'h1_y3': int(lm_h1[3].y * height), 'h1_y4': int(lm_h1[4].y * height), 'h1_y5': int(lm_h1[5].y * height), 'h1_y6': int(lm_h1[6].y * height), 'h1_y7': int(lm_h1[7].y * height), 'h1_y8': int(lm_h1[8].y * height), 'h1_y9': int(lm_h1[9].y * height), 'h1_y10': int(lm_h1[10].y * height), 'h1_y11': int(lm_h1[11].y * height), 'h1_y12': int(lm_h1[12].y * height), 'h1_y13': int(lm_h1[13].y * height), 'h1_y14': int(lm_h1[14].y * height), 'h1_y15': int(lm_h1[15].y * height), 'h1_y16': int(lm_h1[16].y * height), 'h1_y17': int(lm_h1[17].y * height), 'h1_y18': int(lm_h1[18].y * height), 'h1_y19': int(lm_h1[19].y * height), 'h1_y20': int(lm_h1[20].y * height)}
                lm_z_h1 = {'h1_z0': int(lm_h1[0].z), 'h1_z1': int(lm_h1[1].z), 'h1_z2': int(lm_h1[2].z), 'h1_z3': int(lm_h1[3].z), 'h1_z4': int(lm_h1[4].z), 'h1_z5': int(lm_h1[5].z), 'h1_z6': int(lm_h1[6].z), 'h1_z7': int(lm_h1[7].z), 'h1_z8': int(lm_h1[8].z), 'h1_z9': int(lm_h1[9].z), 'h1_z10': int(lm_h1[10].z), 'h1_z11': int(lm_h1[11].z), 'h1_z12': int(lm_h1[12].z), 'h1_z13': int(lm_h1[13].z), 'h1_z14': int(lm_h1[14].z), 'h1_z15': int(lm_h1[15].z), 'h1_z16': int(lm_h1[16].z), 'h1_z17': int(lm_h1[17].z), 'h1_z18': int(lm_h1[18].z), 'h1_z19': int(lm_h1[19].z), 'h1_z20': int(lm_h1[20].z)}
                flag = i+1
                if print_lm == True:
                    print(lm_x_h1)
                    print(lm_y_h1)
                    print(lm_z_h1)
                x_min, y_min = min(lm_x_h1.values()) - int(width*size_roi), min(lm_y_h1.values()) - int(height*size_roi)
                x_max, y_max = max(lm_x_h1.values()) + int(width*size_roi), max(lm_y_h1.values()) + int(height*size_roi)
                roi = frame_equali[y_min:y_max, x_min:x_max]  #! EQUALIZED ROIS SAVED
                roi_save.append(roi)
                roi_width, roi_height = roi.shape
                lm_x_h1_roi = {f'h1_xroi{i}': int(lm_h1[i].x*roi_width) for i in range(21)}
                lm_y_h1_roi = {f'h1_yroi{i}': int(lm_h1[i].y*roi_height) for i in range(21)}
                point_save['h1_x_min'], point_save['h1_y_min'], point_save['h1_x_max'], point_save['h1_y_max'] = x_min, y_min, x_max, y_max
            if  i == 1:
                lm_h2 =  handlms.landmark   #results.multi_hand_landmarks[1].landmark
                lm_x_h2 = {'h2_x0': int(lm_h2[0].x * width), 'h2_x1': int(lm_h2[1].x * width), 'h2_x2': int(lm_h2[2].x * width), 'h2_x3': int(lm_h2[3].x * width), 'h2_x4': int(lm_h2[4].x * width), 'h2_x5': int(lm_h2[5].x * width), 'h2_x6': int(lm_h2[6].x * width), 'h2_x7': int(lm_h2[7].x * width), 'h2_x8': int(lm_h2[8].x * width), 'h2_x9': int(lm_h2[9].x * width), 'h2_x10': int(lm_h2[10].x * width), 'h2_x11': int(lm_h2[11].x * width), 'h2_x12': int(lm_h2[12].x * width), 'h2_x13': int(lm_h2[13].x * width), 'h2_x14': int(lm_h2[14].x * width), 'h2_x15': int(lm_h2[15].x * width), 'h2_x16': int(lm_h2[16].x * width), 'h2_x17': int(lm_h2[17].x * width), 'h2_x18': int(lm_h2[18].x * width), 'h2_x19': int(lm_h2[19].x * width), 'h2_x20': int(lm_h2[20].x * width)}
                lm_y_h2 = {'h2_y0': int(lm_h2[0].y * height), 'h2_y1': int(lm_h2[1].y * height), 'h2_y2': int(lm_h2[2].y * height), 'h2_y3': int(lm_h2[3].y * height), 'h2_y4': int(lm_h2[4].y * height), 'h2_y5': int(lm_h2[5].y * height), 'h2_y6': int(lm_h2[6].y * height), 'h2_y7': int(lm_h2[7].y * height), 'h2_y8': int(lm_h2[8].y * height), 'h2_y9': int(lm_h2[9].y * height), 'h2_y10': int(lm_h2[10].y * height), 'h2_y11': int(lm_h2[11].y * height), 'h2_y12': int(lm_h2[12].y * height), 'h2_y13': int(lm_h2[13].y * height), 'h2_y14': int(lm_h2[14].y * height), 'h2_y15': int(lm_h2[15].y * height), 'h2_y16': int(lm_h2[16].y * height), 'h2_y17': int(lm_h2[17].y * height), 'h2_y18': int(lm_h2[18].y * height), 'h2_y19': int(lm_h2[19].y * height), 'h2_y20': int(lm_h2[20].y * height)}
                lm_z_h2 = {'h2_z0': int(lm_h2[0].z), 'h2_z1': int(lm_h2[1].z), 'h2_z2': int(lm_h2[2].z), 'h2_z3': int(lm_h2[3].z), 'h2_z4': int(lm_h2[4].z), 'h2_z5': int(lm_h2[5].z), 'h2_z6': int(lm_h2[6].z), 'h2_z7': int(lm_h2[7].z), 'h2_z8': int(lm_h2[8].z), 'h2_z9': int(lm_h2[9].z), 'h2_z10': int(lm_h2[10].z), 'h2_z11': int(lm_h2[11].z), 'h2_z12': int(lm_h2[12].z), 'h2_z13': int(lm_h2[13].z), 'h2_z14': int(lm_h2[14].z), 'h2_z15': int(lm_h2[15].z), 'h2_z16': int(lm_h2[16].z), 'h2_z17': int(lm_h2[17].z), 'h2_z18': int(lm_h2[18].z), 'h2_z19': int(lm_h2[19].z), 'h2_z20': int(lm_h2[20].z)}

                if print_lm == True:
                    print(lm_x_h2)
                    print(lm_y_h2)
                    print(lm_z_h2)
                x_min, y_min = min(lm_x_h2.values()) - int(width*size_roi), min(lm_y_h2.values())- int(height*size_roi)
                x_max, y_max = max(lm_x_h2.values()) + int(width*size_roi), max(lm_y_h2.values())+ int(height*size_roi)
                roi = frame_equali[y_min:y_max, x_min:x_max]  #! EQUALIZED ROIS SAVED
                roi_save.append(roi)
                roi_width, roi_height = roi.shape
                prop_x = roi_width / width
                prop_y = roi_height / height
                lm_x_h2_roi = {f'h2_xroi{i}': int(lm_h2[i].x*roi_width) for i in range(21)}
                lm_y_h2_roi = {f'h2_yroi{i}': int(lm_h2[i].y*roi_height) for i in range(21)}
                point_save['h2_x_min'], point_save['h2_y_min'], point_save['h2_x_max'], point_save['h2_y_max'] = x_min, y_min, x_max, y_max
            
            # for id, lm in enumerate(handlms.landmark):
            #     cx, cy = int(lm.x * width), int(lm.y * height)
            #     if cx < x_min:
            #         x_min = cx - corner_x
            #     if cx > x_max:
            #         x_max = cx + corner_x
            #     if cy < y_min:
            #         y_min = cy - corner_y
            #     if cy > y_max:
            #         y_max = cy + corner_y
            #     cv2.circle(frame, (cx, cy), 15, (139, 0, 0), cv2.FILLED)
            #     print(cx_saved)
            #     cx_saved[id] = cx  ### SAVE THE POSITIONS OF LANDMARKS
            #     cy_saved[id] = cy  ### SAVE THE POSITIONS OF LANDMARKS
            #* Definir la ROI (Region of Interest)
    else:
         flag = 0
            
            # save_len = len(cx_saved)
            # if t > tiempo_de_espera:
            #     cx_saved = np.zeros(21).reshape(21,1)
            #     cy_saved = np.zeros(21).reshape(21,1)
            #     save_len = 0
    return roi_save, save_len, point_save, lm_x_h1, lm_y_h1, lm_x_h2, lm_y_h2, lm_x_h1_roi, lm_y_h1_roi, lm_x_h2_roi, lm_y_h2_roi,flag

def draw_text_and_rectangles(point_save,frame, width, height, fps,draw_rectangules = True,draw_text = True)-> None:
    """
    This function draws the text and rectangles in the frame.
    Parameters:
    frame: The frame from the camera.
    width: The width of the camera.
    height: The height of the camera.
    fps: The frames per second.
    point_save: The list with the points of the ROIs.
    
    """
    cv2.putText(frame, 'Sign Language recognition', (int(width*0.05), int(height*0.05)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, 'FPS = ' + str(int(fps)), (int(width*0.05), int(height*0.1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, 'Press q to exit', (int(width*0.7), int(height*0.05)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if np.any(point_save):
        if draw_rectangules == True:
            cv2.rectangle(frame, pt1=(point_save['h1_x_min'], point_save['h1_y_min']), pt2=(point_save['h1_x_max'], point_save['h1_y_max']), color=(100, 100, 255), thickness=3)
        if draw_text == True:
            cv2.putText(frame, f'ROI{1}', (point_save['h1_x_min'], point_save['h1_y_min']-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 255), 3)
        
        if 'h2_x_min' in point_save.keys():
            if draw_rectangules == True:
                cv2.rectangle(frame, pt1=(point_save['h2_x_min'], point_save['h2_y_min']), pt2=(point_save['h2_x_max'], point_save['h2_y_max']), color=(255 ,100, 100), thickness=3)
            if draw_text == True:
                cv2.putText(frame, f'ROI{2}', (point_save['h2_x_min'], point_save['h2_y_min']-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 255), 3) 
    else:
        pass

def ends_if(cTime, fps, Ts, pTime, time_frames)-> tuple:
    """
    This function ends the program if the key 'q' is pressed.
    Parameters:
    cTime (float): The current time.
    fps (float): The frames per second.
    Ts (float): The time for the frames.
    pTime (float): The previous time.
    time_frames (float): The time for the frames.
    Returns:
    cTime (float): The current time.
    fps (float): The frames per second.
    Ts (float): The time for the frames.
    pTime (float): The previous time.
    time_frames (float): The time for the frames.
    
    """    
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime 
    Ts = time.time() - time_frames
    time_frames = time.time()
    return cTime, fps, Ts, pTime, time_frames
