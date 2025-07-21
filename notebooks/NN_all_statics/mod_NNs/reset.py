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



def reset_save(roi_save)-> tuple:
    """
    This function resets the lists.
    Parameters:
    point_save (list): The list to save the points of the ROIs.
    roi_save (list): The list to save the ROIs.
    Returns:
    point_save: The list to save the points of the ROIs.
    roi_save: The list to save the ROIs.
    
    """
    global point_save
    point_save =  {'h1_x_min': 0, 'h1_y_min': 0, 'h1_x_max': 0, 'h1_y_max' : 0, 'h2_x_min': 0, 'h2_y_min': 0, 'h2_x_max': 0, 'h2_y_max' : 0 }
    roi_save = []
    return roi_save, point_save

def fals_dinam(DINAMIC, df, RECORDING, t1, save_len)-> tuple:
    """
    This function resets the lists and the flags for the dinamic signals.
    Parameters:
    DINAMIC (bool): A flag to change between static and dinamic signals.
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
    if DINAMIC == False:
        df = pd.DataFrame()
        """lm_x_h1 = {'h1_x0': 0, 'h1_x1': 0, 'h1_x2': 0, 'h1_x3': 0, 'h1_x4': 0, 'h1_x5': 0, 'h1_x6': 0, 'h1_x7': 0, 'h1_x8': 0, 'h1_x9': 0, 'h1_x10': 0, 'h1_x11': 0, 'h1_x12': 0, 'h1_x13': 0, 'h1_x14': 0, 'h1_x15': 0, 'h1_x16': 0, 'h1_x17': 0, 'h1_x18': 0, 'h1_x19': 0, 'h1_x20': 0}
        lm_x_h1_roi = {'h1_x0_roi': 0, 'h1_x1_roi': 0, 'h1_x2_roi': 0, 'h1_x3_roi': 0, 'h1_x4_roi': 0, 'h1_x5_roi': 0, 'h1_x6_roi': 0, 'h1_x7_roi': 0, 'h1_x8_roi': 0, 'h1_x9_roi': 0, 'h1_x10_roi': 0, 'h1_x11_roi': 0, 'h1_x12_roi': 0, 'h1_x13_roi': 0, 'h1_x14_roi': 0, 'h1_x15_roi': 0, 'h1_x16_roi': 0, 'h1_x17_roi': 0, 'h1_x18_roi': 0, 'h1_x19_roi': 0, 'h1_x20_roi': 0}
        lm_y_h1 = {'h1_y0': 0, 'h1_y1': 0, 'h1_y2': 0, 'h1_y3': 0, 'h1_y4': 0, 'h1_y5': 0, 'h1_y6': 0, 'h1_y7': 0, 'h1_y8': 0, 'h1_y9': 0, 'h1_y10': 0, 'h1_y11': 0, 'h1_y12': 0, 'h1_y13': 0, 'h1_y14': 0, 'h1_y15': 0, 'h1_y16': 0, 'h1_y17': 0, 'h1_y18': 0, 'h1_y19': 0, 'h1_y20': 0}
        lm_y_h1_roi = {'h1_y0_roi': 0, 'h1_y1_roi': 0, 'h1_y2_roi': 0, 'h1_y3_roi': 0, 'h1_y4_roi': 0, 'h1_y5_roi': 0, 'h1_y6_roi': 0, 'h1_y7_roi': 0, 'h1_y8_roi': 0, 'h1_y9_roi': 0, 'h1_y10_roi': 0, 'h1_y11_roi': 0, 'h1_y12_roi': 0, 'h1_y13_roi': 0, 'h1_y14_roi': 0, 'h1_y15_roi': 0, 'h1_y16_roi': 0, 'h1_y17_roi': 0, 'h1_y18_roi': 0, 'h1_y19_roi': 0, 'h1_y20_roi': 0}
           """ 
        lm_x_h2 = {}
        lm_x_h2_roi = {}
        lm_y_h2 = {}
        lm_y_h2_roi = {}
        
        ####### print('Manos fuera antes de tiempo')
        RECORDING = False
        t1 = 0
        save_len=0
    return df, RECORDING, t1, save_len, lm_x_h1, lm_y_h1, lm_x_h2, lm_y_h2, lm_x_h1_roi, lm_y_h1_roi, lm_x_h2_roi, lm_y_h2_roi

def tru_dinam(DINAMIC, df, RECORDING, t1, save_len, ventana_de_tiempo, full_path, Nimages, width, height, frame)-> tuple:
    """
    This function saves the data in a csv file.
    Parameters:
    DINAMIC (bool): A flag to change between static and dinamic signals.
    df (DataFrame): The DataFrame to save the data.
    cx_saved (list): The list to save the x positions of the landmarks.
    cy_saved (list): The list to save the y positions of the landmarks.
    cxROI_saved (list): The list to save the x positions of the ROI.
    cyROI_saved (list): The list to save the y positions of the ROI.
    RECORDING (bool): A flag to record the data.
    t1 (float): The time for the recording.
    save_len (int): The length of the saved lists.
    ventana_de_tiempo (int): The time window to save the data.
    full_path (str): The path of the folder.
    Nimages (int): The last element in the folder.
    width (int): The width of the camera.
    height (int): The height of the camera.
    frame: The frame from the camera.
    Returns:
    df: The DataFrame to save the data.
    cx_saved: The list to save the x positions of the landmarks.
    cy_saved: The list to save the y positions of the landmarks.
    cxROI_saved: The list to save the x positions of the ROI.
    cyROI_saved: The list to save the y positions of the ROI.
    RECORDING: A flag to record the data.
    t1: The time for the recording.
    save_len: The length of the saved lists.
    Nimages: The last element in the folder.
    
    """
    global lm_x_h1, lm_y_h1, lm_x_h2, lm_y_h2, lm_x_h1_roi, lm_y_h1_roi, lm_x_h2_roi, lm_y_h2_roi
    if DINAMIC == True:
        if t1 > ventana_de_tiempo:
            df['cx'] = lm_x_h1.values()
            df['cxROI'] = lm_x_h1_roi.values()
            df['cy'] = lm_y_h1.values()
            df['cyROI'] = lm_y_h1_roi.values()
            #print(df)
            Nimages += 1
            df.to_csv(full_path+'//' + csv(Nimages), index=True)
            """
            lm_x_h1 = {'h1_x0': 0, 'h1_x1': 0, 'h1_x2': 0, 'h1_x3': 0, 'h1_x4': 0, 'h1_x5': 0, 'h1_x6': 0, 'h1_x7': 0, 'h1_x8': 0, 'h1_x9': 0, 'h1_x10': 0, 'h1_x11': 0, 'h1_x12': 0, 'h1_x13': 0, 'h1_x14': 0, 'h1_x15': 0, 'h1_x16': 0, 'h1_x17': 0, 'h1_x18': 0, 'h1_x19': 0, 'h1_x20': 0}
            lm_x_h1_roi = {'h1_x0_roi': 0, 'h1_x1_roi': 0, 'h1_x2_roi': 0, 'h1_x3_roi': 0, 'h1_x4_roi': 0, 'h1_x5_roi': 0, 'h1_x6_roi': 0, 'h1_x7_roi': 0, 'h1_x8_roi': 0, 'h1_x9_roi': 0, 'h1_x10_roi': 0, 'h1_x11_roi': 0, 'h1_x12_roi': 0, 'h1_x13_roi': 0, 'h1_x14_roi': 0, 'h1_x15_roi': 0, 'h1_x16_roi': 0, 'h1_x17_roi': 0, 'h1_x18_roi': 0, 'h1_x19_roi': 0, 'h1_x20_roi': 0}
            lm_y_h1 = {'h1_y0': 0, 'h1_y1': 0, 'h1_y2': 0, 'h1_y3': 0, 'h1_y4': 0, 'h1_y5': 0, 'h1_y6': 0, 'h1_y7': 0, 'h1_y8': 0, 'h1_y9': 0, 'h1_y10': 0, 'h1_y11': 0, 'h1_y12': 0, 'h1_y13': 0, 'h1_y14': 0, 'h1_y15': 0, 'h1_y16': 0, 'h1_y17': 0, 'h1_y18': 0, 'h1_y19': 0, 'h1_y20': 0}
            lm_y_h1_roi = {'h1_y0_roi': 0, 'h1_y1_roi': 0, 'h1_y2_roi': 0, 'h1_y3_roi': 0, 'h1_y4_roi': 0, 'h1_y5_roi': 0, 'h1_y6_roi': 0, 'h1_y7_roi': 0, 'h1_y8_roi': 0, 'h1_y9_roi': 0, 'h1_y10_roi': 0, 'h1_y11_roi': 0, 'h1_y12_roi': 0, 'h1_y13_roi': 0, 'h1_y14_roi': 0, 'h1_y15_roi': 0, 'h1_y16_roi': 0, 'h1_y17_roi': 0, 'h1_y18_roi': 0, 'h1_y19_roi': 0, 'h1_y20_roi': 0}
            """
            save_len = 0
            t1 = 0
            df = pd.DataFrame()
            RECORDING = False
            cv2.putText(frame,'Press any key to continue', (int(width*0.2),int(height*0.5)),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2)
            cv2.imshow('frame', frame)
            cv2.waitKey(0)  # Espera a que se presione cualquier tecla
    return df, RECORDING, t1, save_len, Nimages, lm_x_h1, lm_y_h1, lm_x_h2, lm_y_h2, lm_x_h1_roi, lm_y_h1_roi, lm_x_h2_roi, lm_y_h2_roi   
