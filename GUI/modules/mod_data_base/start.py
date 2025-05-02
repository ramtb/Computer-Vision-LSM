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

