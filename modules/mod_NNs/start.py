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



######* CAMERA SETTINGS ###########
def camera_settings(width_cam = 1280, height_cam = 720, camera = 0)-> tuple:
    """
    This function sets the camera settings.
    Returns:
    cap: The camera object.
    width: The width of the camera.
    height: The height of the camera.
    kernel: The kernel for the morphological operations.
    
    """

    cap = cv2.VideoCapture(camera) #* CAMERA SETTINGS
    #print('hola',cap)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width_cam)  #* set the width of the camera
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height_cam)  #* set the height of the camera
    # Añadir esta línea para ajustar el tamaño del buffer
    #cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'Resolution: {width},{height}')
    print( 'using default camera' if camera == 0 else 'using external camera')
    return cap, width, height

########* PARAMETERS ###############

def time_set()-> tuple:
    """
    This function sets the time parameters for the program.
    Returns:
    time_frames (float): The time for the frames.
    t (float): The time for the ROI.
    t1 (float): The time for the recording.
    timeflag (bool): A flag to set the time for the ROI.
    
    """
    time_frames = time.time()
    t = 0.0
    t1 = 0
    timeflag = False
    return time_frames, t, t1, timeflag

def wind_move(roi1_x = 0.4,roi1_y =0.3, roi2_x = 0.6, roi2_y = 0.3 )-> list:
    """
    This function sets the window position for the ROI.
    Returns:
    window_move (list): The list with the position of the windows.
    
    """
    window_move = [[int(1920*roi1_x),int(1080*roi1_y)],[int(1920*roi2_x),int(1080*roi2_y)] ]  
    return window_move

def frame_settings()-> tuple:
    """
    This function sets the variables for the frames.
    Returns:
    cTime (float): The current time.
    pTime (float): The previous time.
    fps (float): The frames per second.
    Ts (float): The time for the frames.
    time_frames (float): The time for the frames.
    
    """
    cTime, pTime, fps, Ts, time_frames = 0, 0, 0, 0, 0  #* VARIABLES FOR FRAMES
    return cTime, pTime, fps, Ts, time_frames

