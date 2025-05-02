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

mpDraw = None
point_save = None
mpHands = None
Letra = None
csv = None

def remake_parameters(df,  RECORDING, t1, save_len)-> tuple:
    """
    This function resets the lists and the flags for the recording.
    Parameters:
    df (DataFrame): The DataFrame to save the data.
    cx_saved (list): The list to save the x positions of the landmarks.
    cy_saved (list): The list to save the y positions of the landmarks.
    cxROI_saved (list): The list to save the x positions of the ROI.
    cyROI_saved (list): The list to save the y positions of the ROI.
    RECORDING (bool): A flag to record the data.
    t1 (float): The time for the recording.
    save_len (int): The length of the saved lists.
    Returns:
    df: The DataFrame to save the data.
    cx_saved: The list to save the x positions of the landmarks.
    cy_saved: The list to save the y positions of the landmarks.
    cxROI_saved: The list to save the x positions of the ROI.
    cyROI_saved: The list to save the y positions of the ROI.
    RECORDING: A flag to record the data.
    t1: The time for the recording.
    save_len: The length of the saved lists.
    
    """
    global lm_x_h1, lm_y_h1, lm_x_h2, lm_y_h2, lm_x_h1_roi, lm_y_h1_roi, lm_x_h2_roi, lm_y_h2_roi
    df = pd.DataFrame()
    """lm_x_h1 = {'h1_x0': 0, 'h1_x1': 0, 'h1_x2': 0, 'h1_x3': 0, 'h1_x4': 0, 'h1_x5': 0, 'h1_x6': 0, 'h1_x7': 0, 'h1_x8': 0, 'h1_x9': 0, 'h1_x10': 0, 'h1_x11': 0, 'h1_x12': 0, 'h1_x13': 0, 'h1_x14': 0, 'h1_x15': 0, 'h1_x16': 0, 'h1_x17': 0, 'h1_x18': 0, 'h1_x19': 0, 'h1_x20': 0}
    lm_x_h1_roi = {'h1_x0_roi': 0, 'h1_x1_roi': 0, 'h1_x2_roi': 0, 'h1_x3_roi': 0, 'h1_x4_roi': 0, 'h1_x5_roi': 0, 'h1_x6_roi': 0, 'h1_x7_roi': 0, 'h1_x8_roi': 0, 'h1_x9_roi': 0, 'h1_x10_roi': 0, 'h1_x11_roi': 0, 'h1_x12_roi': 0, 'h1_x13_roi': 0, 'h1_x14_roi': 0, 'h1_x15_roi': 0, 'h1_x16_roi': 0, 'h1_x17_roi': 0, 'h1_x18_roi': 0, 'h1_x19_roi': 0, 'h1_x20_roi': 0}
    lm_y_h1 = {'h1_y0': 0, 'h1_y1': 0, 'h1_y2': 0, 'h1_y3': 0, 'h1_y4': 0, 'h1_y5': 0, 'h1_y6': 0, 'h1_y7': 0, 'h1_y8': 0, 'h1_y9': 0, 'h1_y10': 0, 'h1_y11': 0, 'h1_y12': 0, 'h1_y13': 0, 'h1_y14': 0, 'h1_y15': 0, 'h1_y16': 0, 'h1_y17': 0, 'h1_y18': 0, 'h1_y19': 0, 'h1_y20': 0}
    lm_y_h1_roi = {'h1_y0_roi': 0, 'h1_y1_roi': 0, 'h1_y2_roi': 0, 'h1_y3_roi': 0, 'h1_y4_roi': 0, 'h1_y5_roi': 0, 'h1_y6_roi': 0, 'h1_y7_roi': 0, 'h1_y8_roi': 0, 'h1_y9_roi': 0, 'h1_y10_roi': 0, 'h1_y11_roi': 0, 'h1_y12_roi': 0, 'h1_y13_roi': 0, 'h1_y14_roi': 0, 'h1_y15_roi': 0, 'h1_y16_roi': 0, 'h1_y17_roi': 0, 'h1_y18_roi': 0, 'h1_y19_roi': 0, 'h1_y20_roi': 0}
    lm_x_h2 = {}
    lm_x_h2_roi = {}
    lm_y_h2 = {}
    lm_y_h2_roi = {}"""
    RECORDING = False
    t1 = 0
    save_len = 0
    return df, RECORDING, t1, save_len, lm_x_h1, lm_y_h1, lm_x_h2, lm_y_h2, lm_x_h1_roi, lm_y_h1_roi, lm_x_h2_roi, lm_y_h2_roi

def roi_in_roi(roi_save, window_move)-> list:
    """
    This function shows the ROIs in the windows.
    Parameters:
    roi_save (list): The list to save the ROIs.
    window_move (list): The list with the position of the windows.
    Returns:
    roi_save: The list to save the ROIs.

    """
    for i in range(len(roi_save)):  ## for how many rois there is do the loop
        r1,r2 = roi_save[i].shape  #### Shape of every roi 
        if r1 & r2 > 0:  ### If the shape is not empty show it
            #roi_save[i] = cv2.medianBlur(roi_save[i],3)
            #roi_save[i] = cv2.Sobel(roi_save[i],cv2.CV_64F,0,1,kernel)  ##APLY SOBEL METHOD to edge detection
            #ret, roi_save[i] = cv2.threshold(roi_save[i],100,255,cv2.THRESH_TRUNC)
            #roi_save[i] = cv2.morphologyEx(roi_save[i],cv2.MORPH_OPEN,kernel)
            cv2.namedWindow(f'ROI{i+1}', cv2.WINDOW_NORMAL)  # Usa WINDOW_NORMAL para permitir cambiar el tamaño de la ventana
            cv2.resizeWindow(f'ROI{i+1}', int(1920*0.1) , int(1080*0.1 ))
            cv2.imshow(f'ROI{i+1}', roi_save[i])
            cv2.moveWindow(f'ROI{i+1}', window_move[i][0], window_move[i][1])
    return roi_save  

def save_true(frame, width, height)-> bool:
    """
    This function shows the message to save the ROI.
    Parameters:
    frame: The frame from the camera.
    width: The width of the camera.
    height: The height of the camera.
    Returns:
    SAVED (bool): A flag to save the ROI.
    
    """
    cv2.putText(frame,'Press any key to continue', (int(width*0.2),int(height*0.5)),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),3)
    cv2.imshow('frame', frame)
    cv2.waitKey(0)  # Espera a que se presione cualquier tecla
    SAVED = False
    return SAVED

def imshow(frame)-> None:
    """
    This function shows the frame from the camera.
    Parameters:
    frame: The frame from the camera.
    
    """
    cv2.imshow('frame',frame)

def main_show(frame, SAVED, width, height, roi_save, window_move, df, RECORDING, t1, save_len, show_rois = False)-> bool:
    """
    This function shows the frame from the camera.
    Parameters:
    frame: The frame from the camera.
    SAVED (bool): A flag to save the ROI.
    width (int): The width of the camera.
    height (int): The height of the camera.
    roi_save (list): The list to save the ROIs.
    window_move (list): The list with the position of the windows.
    df (DataFrame): The DataFrame to save the data.
    cx_saved (list): The list to save the x positions of the landmarks.
    cy_saved (list): The list to save the y positions of the landmarks.
    cxROI_saved (list): The list to save the x positions of the ROI.
    cyROI_saved (list): The list to save the y positions of the ROI.
    RECORDING (bool): A flag to record the data.
    t1 (float): The time for the recording.
    save_len (int): The length of the saved lists.
    Returns:
    SAVED (bool): A flag to save the ROI.
    
    """
    global  lm_x_h1, lm_y_h1, lm_x_h2, lm_y_h2, lm_x_h1_roi, lm_y_h1_roi, lm_x_h2_roi, lm_y_h2_roi
    imshow(frame)
    
    if SAVED == True:

        SAVED = save_true(frame, width, height)
    if len(roi_save)>0 and len(roi_save)<3:  #### if there is some roi in roi_save do the for loop
        if show_rois:
            roi_in_roi(roi_save, window_move)  
    else:

        df, RECORDING, t1, save_len, lm_x_h1, lm_y_h1, lm_x_h2, lm_y_h2, lm_x_h1_roi, lm_y_h1_roi, lm_x_h2_roi, lm_y_h2_roi = remake_parameters(df, RECORDING, t1, save_len)
    return SAVED
