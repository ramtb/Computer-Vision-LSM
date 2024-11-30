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
mpDraw = None
point_save = None
mpHands = None
Letra = None
csv = None
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
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width_cam)  #* set the width of the camera
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height_cam)  #* set the height of the camera
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'Resolution: {width},{height}')
    print( 'using default camera' if camera == 0 else 'using external camera')
    return cap, width, height


########* PARAMETERS ###############

def format(imgformat = 'jpg', dataformat = 'csv', waiting_time = 3, record_time = 2)-> tuple:
    """
    This function sets the format of the images and the data.
    Returns:
    imgformat (str): The format of the images.
    dataformat (str): The format of the data.
    tiempo_de_espera (int): The waiting time to capture the ROI.
    
    """
    print(f'Format: {imgformat},{dataformat}')
    print(f'Waiting time: {waiting_time} seconds, Recording time: {record_time} seconds')
    
    return imgformat,dataformat,waiting_time,record_time


#########* FOLDER ############### 
def folder()-> tuple:
    """
    Runs the functions to create the folder, select the folder, and get the last element in the folder.
    Returns:
    folder_name (str): The name of the sign to be captured.
    full_path (str): The path of the folder.
    elements (list): The list of elements inside the folder.
    
    """
    
########* FOLDER ###############
    folder_name = sign()
    print(sign_name(folder_name))
    if sign_name(folder_name) == ("Sign don't selected"):
        sys.exit()
    folder_path = select_folder()
    folder_path = doubled_path(folder_path)
    if folder_path == ("Folder don't selected"):
        print(folder_path)
        sys.exit()
    folder_name = doubled_path(folder_name)
    full_path = (folder_path + '//' + folder_name) if platform.system() == 'Windows' else (folder_path + '/' + folder_name)
    print(full_path)
    print(folder_exist(folder_name,full_path,folder_path))
    elements = os.listdir(full_path)
    # last_element(elements)
    return folder_name, full_path, elements

def sign() -> str:
    """ 
    This function creates a dialog box to enter the name of the sign to be captured.
    Returns:
    name_selected: The name of the sign to be captured.

    """
    root = tk.Tk()
    root.withdraw()
    name_selected = simpledialog.askstring("Sign", "Enter the signal to capture:") 
    root.destroy()
    return name_selected

def n_hand():
    """
    This function creates a dialog box to enter the number of hands to be detected.
    Returns:
    num_hand: The number of hands to be detected.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    
    while True:
        num_hand = simpledialog.askinteger("Number of Hands", "Enter the number of hands to be detected:")
        if num_hand in (1, 2):
            root.destroy()
            return num_hand
        else:
            tk.messagebox.showerror("Invalid Input", "The number of hands must be 1 or 2.")


def sign_name (folder_name) -> str:
    """ 
    This function returns a message with the name of the sign to be captured.
    Parameters:
    folder_name (str): The name of the sign to be captured.
    Returns:
    message (str): A message with the name of the sign to be captured.
    
    """
    if folder_name:
        message = ("Selected Sign:") + (folder_name)
    else:
        message = ("Sign don't selected")
    return message

def select_folder() -> str:
    """
    This function creates a dialog box to select the folder where the images will be saved.
    Returns:
    folder_selected: The path of the selected folder.

    """
    root = tk.Tk()
    root.withdraw()  # Oculta la ventana principal de Tkinter
    folder_selected = filedialog.askdirectory()  # Abre el cuadro de diálogo para seleccionar la carpeta
    return folder_selected

def doubled_path(folder_path) -> str:
    """ 
    This function replaces the slashes in the path with double slashes.
    Parameters:
    folder_path (str): The path of the selected folder.
    Returns:
    doubled (str): The path with double slashes.
    """
    if folder_path and platform.system() == 'Windows':
        doubled= folder_path.replace('/', '//')
    elif folder_path and platform.system() == 'Linux':
        doubled= folder_path
    else:
        doubled= ("Folder don't selected")
        sys.exit()
    # print(f'Selected path: {doubled}')
    return doubled

def folder_exist(folder_name,full_path,folder_path) -> str:
    """
    This function checks if the folder already exists before creating it.
    Parameters:
    full_path (str): The path of the folder.
    Returns:
    foldex (str): A message indicating if the folder was created or if it already exists.
    
    """
    if not os.path.exists(full_path):
        os.mkdir(full_path)
        foldex = (f"Folder '{folder_name}' created successfully at '{folder_path}'.")
    else:
        foldex = (f"Folder '{folder_name}' already exists at '{folder_path}'.")
    return foldex

def last_element(elements) -> int:
    """
    This function returns the last element in the folder.
    Parameters:
    elements (list): The list of elements inside the folder.
    Returns:
    Nimages (int): The last element in the folder.
    
    """
    if elements:
        elements = [int(re.findall('\d+', x.split('.')[0])[0]) for x in elements]   
        max_element = max(elements, key=int)
        Nimages = max_element
        print (("Last element in the folder:") + str(Nimages))
    else:
        Nimages = 0
        print ("Folder is empty")
        
    return Nimages

########* PARAMETERS ###############



def parameters(elements)-> tuple:
    """
    This function sets the parameters for the program.
    Parameters:
    elements (list): The list of elements inside the folder.
    Returns:
    Nimages (int): The last element in the folder.
    df (DataFrame): The DataFrame to save the data.
    cx_saved (list): The list to save the x positions of the landmarks.
    cy_saved (list): The list to save the y positions of the landmarks.
    cxROI_saved (list): The list to save the x positions of the ROI.
    cyROI_saved (list): The list to save the y positions of the ROI.
    save_len (int): The length of the saved lists.
    cTime (float): The current time.
    pTime (float): The previous time.
    dinamic_count (int): The count for the dinamic signals.
    roi_save (list): The list to save the ROIs.
    point_save (list): The list to save the points of the ROIs.
    
    """
    global cx_saved  
    global cy_saved 
    global cxROI_saved  
    global cyROI_saved  
    global point_save
    Nimages = last_element(elements)
    df = pd.DataFrame() 
    cx_saved = np.zeros(21).reshape(21,1) ####* SAVE THE X POSITION OF THE LANDMARKS
    cy_saved = np.zeros(21).reshape(21,1) ####* SAVE THE Y POSITION OF THE LANDMARKS
    cxROI_saved = np.zeros(21).reshape(21,1) ####* SAVE THE X POSITION OF THE ROI
    cyROI_saved = np.zeros(21).reshape(21,1) ####* SAVE THE Y POSITION OF THE ROI
    save_len = 0 
    cTime = 0 #* VARIABLES FOR FRAMES
    pTime = 0 #* VARIABLES FOR FRAMES
    dinamic_count = 0
    point_save = {'h1_x_min': 0, 'h1_y_min': 0, 'h1_x_max': 0, 'h1_y_max' : 0, 'h2_x_min': 0, 'h2_y_min': 0, 'h2_x_max': 0, 'h2_y_max' : 0 } #* SAVE THE POINTS OF THE ROIS
    return Nimages, df, save_len, cTime, pTime, dinamic_count, point_save

def save_number(imgformat, dataformat, folder_name)-> tuple:
    """
    This function sets the name of the images and the data.
    Parameters:
    Nimages (int): The last element in the folder.
    Returns:
    Letra (lambda): The name of the images.
    csv (lambda): The name of the data.
    
    """
    global Letra, csv
    Letra = lambda Nimages: folder_name + f'{Nimages}.'+imgformat
    csv = lambda Nimages: folder_name + f'{Nimages}.'+dataformat
    return Letra, csv

def constants()-> tuple:
    """
    This function sets the constants for the program.
    Returns:
    SAVED (bool): A flag to save the ROI.
    DINAMIC (bool): A flag to change between static and dinamic signals.
    RECORDING (bool): A flag to record the data.
    
    """
    SAVED = False   
    DINAMIC = False
    RECORDING = False
    return SAVED, DINAMIC, RECORDING

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

def hand_medipip(num_hand, min_detection_c = 0.5, min_detection_t = 0.5)-> tuple:
    """
    This function sets the mediapipe parameters for the program.
    Returns:
    hands: The hands object.
    mpDraw: The drawing object.
    
    """
    global mpDraw, mpHands
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(static_image_mode=False, max_num_hands= num_hand, min_detection_confidence=min_detection_c, min_tracking_confidence=min_detection_t)
    
    mpDraw = mp.solutions.drawing_utils
    return mpHands, hands, mpDraw

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

#################* TRACKING ####################

def read_frames(cap,hands,equali = True)-> tuple:
    """
    This function reads the frames from the camera.
    Returns:
    frame: The frame from the camera.
    frame_copy: A copy of the frame.
    frame_gray: The frame in gray scale.
    frame_equali: The frame with the histogram equalized.
    results: The results from the hands object.
    
    """
    ret, frame = cap.read()           #* READ FRAMES FROM CAMERA
    frame_copy = frame.copy()  ### Copy of the FRAME
    frame_gray = cv2.cvtColor(frame_copy,cv2.COLOR_BGR2GRAY)  #*CHANGE THE SCALE COLOR
    if equali == True:
        frame_equali = cv2.equalizeHist(frame_gray)  #*Equalize the histogram
    else:
        frame_equali = frame_gray
    results = hands.process(frame) 
    return ret, frame, frame_copy, frame_gray, frame_equali, results


lm_x_h1 = {'h1_x0': 0, 'h1_x1': 0, 'h1_x2': 0, 'h1_x3': 0, 'h1_x4': 0, 'h1_x5': 0, 'h1_x6': 0, 'h1_x7': 0, 'h1_x8': 0, 'h1_x9': 0, 'h1_x10': 0, 'h1_x11': 0, 'h1_x12': 0, 'h1_x13': 0, 'h1_x14': 0, 'h1_x15': 0, 'h1_x16': 0, 'h1_x17': 0, 'h1_x18': 0, 'h1_x19': 0, 'h1_x20': 0}
lm_x_h1_roi = {'h1_x0_roi': 0, 'h1_x1_roi': 0, 'h1_x2_roi': 0, 'h1_x3_roi': 0, 'h1_x4_roi': 0, 'h1_x5_roi': 0, 'h1_x6_roi': 0, 'h1_x7_roi': 0, 'h1_x8_roi': 0, 'h1_x9_roi': 0, 'h1_x10_roi': 0, 'h1_x11_roi': 0, 'h1_x12_roi': 0, 'h1_x13_roi': 0, 'h1_x14_roi': 0, 'h1_x15_roi': 0, 'h1_x16_roi': 0, 'h1_x17_roi': 0, 'h1_x18_roi': 0, 'h1_x19_roi': 0, 'h1_x20_roi': 0}
lm_y_h1 = {'h1_y0': 0, 'h1_y1': 0, 'h1_y2': 0, 'h1_y3': 0, 'h1_y4': 0, 'h1_y5': 0, 'h1_y6': 0, 'h1_y7': 0, 'h1_y8': 0, 'h1_y9': 0, 'h1_y10': 0, 'h1_y11': 0, 'h1_y12': 0, 'h1_y13': 0, 'h1_y14': 0, 'h1_y15': 0, 'h1_y16': 0, 'h1_y17': 0, 'h1_y18': 0, 'h1_y19': 0, 'h1_y20': 0}
lm_y_h1_roi = {'h1_y0_roi': 0, 'h1_y1_roi': 0, 'h1_y2_roi': 0, 'h1_y3_roi': 0, 'h1_y4_roi': 0, 'h1_y5_roi': 0, 'h1_y6_roi': 0, 'h1_y7_roi': 0, 'h1_y8_roi': 0, 'h1_y9_roi': 0, 'h1_y10_roi': 0, 'h1_y11_roi': 0, 'h1_y12_roi': 0, 'h1_y13_roi': 0, 'h1_y14_roi': 0, 'h1_y15_roi': 0, 'h1_y16_roi': 0, 'h1_y17_roi': 0, 'h1_y18_roi': 0, 'h1_y19_roi': 0, 'h1_y20_roi': 0}
            
lm_x_h2 = {}  #* DICTIONARY FOR X POSITIONS OF LANDMARKS
lm_y_h2 = {}  #* DICTIONARY FOR Y POSITIONS OF LANDMARKS
lm_x_h2_roi = {}  #* DICTIONARY FOR X POSITIONS OF LANDMARKS
lm_y_h2_roi = {}  #* DICTIONARY FOR Y POSITIONS OF LANDMARKS

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

###################* INTERFACE ####################

def save_roi(index, Nimages, full_path, roi_save, frame, width, height, df)-> tuple:
    """
    This function saves the ROI.
    Parameters:
    index (int): The index of the ROI.
    Nimages (int): The last element in the folder.
    full_path (str): The path of the folder.
    roi_save (list): The list to save the ROIs.
    frame: The frame from the camera.
    width (int): The width of the camera.
    height (int): The height of the camera.
    df (DataFrame): The DataFrame to save the data.
    cx_saved (list): The list to save the x positions of the landmarks.
    cxROI_saved (list): The list to save the x positions of the ROI.
    cy_saved (list): The list to save the y positions of the landmarks.
    cyROI_saved (list): The list to save the y positions of the ROI.
    Returns:
    Nimages (int): The last element in the folder.
    SAVED (bool): A flag to save the ROI.
    
    """
    global lm_x_h1, lm_y_h1, lm_x_h2, lm_y_h2, lm_x_h1_roi, lm_y_h1_roi, lm_x_h2_roi, lm_y_h2_roi
    Nimages += 1
    cv2.imwrite(full_path + '\\' + Letra(Nimages), roi_save[index])
    cv2.putText(frame, f'ROI {index+1} SAVED', (int(width*0.36),int(height*0.4)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
    if index == 0:
        df['cx'], df['cxROI'], df['cy'], df['cyROI'] = lm_x_h1.values(), lm_x_h1_roi.values(), lm_y_h1.values(), lm_y_h1_roi.values()
        df.to_csv(full_path + '//' + csv(Nimages), index=False)
        print(df)
    if index == 1:
        df['cx'], df['cxROI'], df['cy'], df['cyROI'] = lm_x_h1.values(), lm_x_h1_roi.values(), lm_y_h1.values(), lm_y_h1_roi.values()
        df.to_csv(full_path + '//' + csv(Nimages) + 'R', index=False)
        df['cx'] = lm_x_h2.values()
        df['cxROI'] = lm_x_h2_roi.values()
        df['cy'] =  lm_y_h2.values()
        df['cyROI'] = lm_y_h2_roi.values()
        df.to_csv(full_path + '//' + csv(Nimages) + 'L', index=False)
    return Nimages, True

def show_message(frame, message, position, width, height)-> None:
    """
    This function shows a message in the frame.
    Parameters:
    frame: The frame from the camera.
    message (str): The message to show.
    position (tuple): The position of the message.
    width (int): The width of the camera.
    height (int): The height of the camera.
    
    """
    cv2.putText(frame, message, (int(width * position[0]), int(height * position[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)

def process_static_mode(frame, width, height, roi_save, full_path, Nimages, df,  SAVED)-> tuple:
    """
    This function manages the static mode of the program.
    Parameters:
    frame: The frame from the camera.
    width: The width of the camera.
    height: The height of the camera.
    roi_save (list): The list to save the ROIs.
    full_path (str): The path of the folder.
    Nimages (int): The last element in the folder.
    df (DataFrame): The DataFrame to save the data.
    cx_saved (list): The list to save the x positions of the landmarks.
    cxROI_saved (list): The list to save the x positions of the ROI.
    cy_saved (list): The list to save the y positions of the landmarks.
    cyROI_saved (list): The list to save the y positions of the ROI.
    SAVED (bool): A flag to save the ROI.
    Returns:
    Nimages (int): The last element in the folder.
    SAVED (bool): A flag to save the ROI.
    
    """
    global lm_x_h1, lm_y_h1, lm_x_h2, lm_y_h2, lm_x_h1_roi, lm_y_h1_roi, lm_x_h2_roi, lm_y_h2_roi
    show_message(frame, 'Press 1 or 2 to capture the ROI', (0.1, 0.9), width, height)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('1') and len(roi_save) > 0:
        Nimages, SAVED = save_roi(0, Nimages, full_path, roi_save, frame, width, height, df)
    elif key == ord('2') and len(roi_save) > 1:
        Nimages, SAVED = save_roi(1, Nimages, full_path, roi_save, frame, width, height, df)
    return Nimages, SAVED

def process_dynamic_mode(frame, width, height, roi_save, full_path, Nimages, timeflag)-> bool:
    """
    This function manages the dynamic mode of the program.
    Parameters:
    frame: The frame from the camera.
    width: The width of the camera.
    height: The height of the camera.
    roi_save (list): The list to save the ROIs.
    full_path (str): The path of the folder.
    Nimages (int): The last element in the folder.
    timeflag (bool): A flag to set the time for the ROI.
    Returns:
    timeflag (bool): A flag to set the time for the ROI.
    
    """
    show_message(frame, 'Press 1 or 2 to capture the ROI', (0.1, 0.9), width, height)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('1'):
        timeflag = True
    elif key == ord('2') and len(roi_save) > 1:
        cv2.imwrite(full_path + '\\' + Letra(Nimages), roi_save[1])
        cv2.putText(frame, 'ROI 2 SAVED', (int(width * 0.36), int(height * 0.4)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
    return timeflag

def update_time_display(frame, t, tiempo_de_espera)-> None:
    """
    This function updates the time display.
    Parameters:
    frame: The frame from the camera.
    point_save (list): The list to save the points of the ROIs.
    t (float): The time for the ROI.
    tiempo_de_espera (int): The waiting time to capture the ROI.
    
    """
    global point_save
    for i in range(len(point_save)):
        if i == 0:
            cv2.putText(frame, str(tiempo_de_espera - t), (point_save['h1_x_min'] + 50, point_save['h1_y_min'] - 40), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,100,100), 3)
        if i == 1:
            cv2.putText(frame, str(tiempo_de_espera - t), (point_save['h2_x_min'] + 50, point_save['h2_y_min'] - 40), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,100,100), 3)

def update_recording_display(frame, t1)-> None:
    """
    This function updates the recording display.
    Parameters:
    frame: The frame from the camera.
    point_save (list): The list to save the points of the ROIs.
    t1 (float): The time for the recording.
    
    """
    global point_save
    for i in range(len(point_save)):
        if i == 0:
            cv2.putText(frame, "RECORDING:" + str(t1), (point_save['h1_x_min'] + 50, point_save['h1_y_min'] - 40), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,100,100), 3)
        if i == 1:
            cv2.putText(frame, "RECORDING:" + str(t1), (point_save['h2_x_min'] + 50, point_save['h2_y_min'] - 40), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,100,100), 3)

def interface(frame, width, height, roi_save, full_path, Nimages, df, SAVED, timeflag, t, Ts, tiempo_de_espera, RECORDING, t1, DINAMIC)-> tuple:
    """
    This function manages the interface of the program.
    Parameters:
    frame: The frame from the camera.
    width: The width of the camera.
    height: The height of the camera.
    roi_save (list): The list to save the ROIs.
    full_path (str): The path of the folder.
    Nimages (int): The last element in the folder.
    df (DataFrame): The DataFrame to save the data.
    cx_saved (list): The list to save the x positions of the landmarks.
    cxROI_saved (list): The list to save the x positions of the ROI.
    cy_saved (list): The list to save the y positions of the landmarks.
    cyROI_saved (list): The list to save the y positions of the ROI.
    DINAMIC (bool): A flag to change between static and dinamic signals.
    SAVED (bool): A flag to save the ROI.
    timeflag (bool): A flag to set the time for the ROI.
    t (float): The time for the ROI.
    Ts (float): The time for the frames.
    point_save (list): The list to save the points of the ROIs.
    tiempo_de_espera (int): The waiting time to capture the ROI.
    RECORDING (bool): A flag to record the data.
    t1 (float): The time for the recording.
    Returns:
    Nimages (int): The last element in the folder.
    SAVED (bool): A flag to save the ROI.
    timeflag (bool): A flag to set the time for the ROI.
    t (float): The time for the ROI.
    RECORDING (bool): A flag to record the data.
    t1 (float): The time for the recording.
    
    """
    global lm_x_h1, lm_y_h1, lm_x_h2, lm_y_h2, lm_x_h1_roi, lm_y_h1_roi, lm_x_h2_roi, lm_y_h2_roi
    if DINAMIC == False:
        if 0 < len(roi_save) < 3:
            Nimages, SAVED = process_static_mode(frame, width, height, roi_save, full_path, Nimages, df,  SAVED)
        elif len(roi_save) == 0:
            show_message(frame, 'There are no ROIS', (0.3, 0.9), width, height)
    else:
        if 0 < len(roi_save) < 3:
            timeflag = process_dynamic_mode(frame, width, height, roi_save, full_path, Nimages, timeflag)
        elif len(roi_save) == 0:
            show_message(frame, 'There are no ROIS', (0.3, 0.9), width, height)

    if timeflag:
        t += Ts
        t = round(t, 2)
        update_time_display(frame, t, tiempo_de_espera)

    if t > tiempo_de_espera:
        t = 0
        timeflag = False
        RECORDING = True

    if RECORDING:
        t1 += Ts
        t1 = round(t1, 2)
        update_recording_display(frame, t1)

    return Nimages, SAVED, timeflag, t, RECORDING, t1

def update_mode(DINAMIC)-> bool:
    """
    This function changes the mode of the program.
    Parameters:
    DINAMIC (bool): A flag to change between static and dinamic signals.
    Returns:
    DINAMIC: A flag to change between static and dinamic signals.
    
    """
    if cv2.waitKey(1) & 0xFF == ord('a'):
        return True
    if cv2.waitKey(1) & 0xFF == ord('s'):
        return False
    return DINAMIC

def display_mode_text(frame, width, height, DINAMIC)-> None:
    """
    This function displays the mode of the program.
    Parameters:
    frame: The frame from the camera.
    width: The width of the camera.
    height: The height of the camera.
    DINAMIC (bool): A flag to change between static and dinamic signals.
    
    """
    if DINAMIC:
        cv2.putText(frame, 'Dinamic signals', (int(width * 0.7), int(height * 0.1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # cv2.putText(frame, str(t), (int(width * 0.8), int(height * 0.15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    else:
        cv2.putText(frame, 'Static signals', (int(width * 0.7), int(height * 0.1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

###################* SHOW ########################
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

def main_show(frame, SAVED, width, height, roi_save, window_move, df, RECORDING, t1, save_len)-> bool:
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

        roi_in_roi(roi_save, window_move)  
    else:

        df, RECORDING, t1, save_len, lm_x_h1, lm_y_h1, lm_x_h2, lm_y_h2, lm_x_h1_roi, lm_y_h1_roi, lm_x_h2_roi, lm_y_h2_roi = remake_parameters(df, RECORDING, t1, save_len)
    return SAVED

###################* RESETS THE LIST ####################

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

