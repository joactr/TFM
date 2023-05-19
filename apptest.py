# %%
import cv2
import numpy as np
import face_detection
import collections
import os, python_speech_features
import scipy.io.wavfile as wav
import random
import pandas as pd
import pickle
import subprocess
import torch
from talkNet import talkNet
detector = face_detection.build_detector(
  "DSFDDetector", confidence_threshold=.2, nms_iou_threshold=.2)

# %%
inputVideo =  "C:/Users/jmmol/Desktop/LIP-RTVE/MP4s/speaker042/speaker042_0063.mp4"

def convert_video_to_audio_ffmpeg(video_file, output_ext="wav"):
    """Extracts audio from video to .wav format, returns path of the resulting audio"""
    filename, ext = os.path.splitext(video_file)
    filename = filename.split('/')[-1]
    #Gets working directory and extracts audio
    subprocess.call(["ffmpeg", "-y", "-i", video_file, os.getcwd()+f"/{filename}.{output_ext}"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT)
    return os.getcwd()+f"/{filename}.{output_ext}" #Nombre del audio

def extractBiggestFace(img):
    """
    Detecta todas las caras de una imagen y devuelve la más grande recortada y reescalada a 112x112
    """
    detections = detector.detect(img)
    idx_max = -1
    area_max = -1
    for i,cntr in enumerate(detections):
        xmin,ymin,xmax,ymax = int(cntr[0]),int(cntr[1]),int(cntr[2]),int(cntr[3]) #Guardamos bounding box
        area = (xmax-xmin)*(ymax-ymin)
        if area > area_max: #Comprobamos si la cara es la más grande
            idx_max = i
            area_max = area
            #print(area,idx_max)
        #cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

    cntr = detections[idx_max]
    try:
        xmin,ymin,xmax,ymax = int(cntr[0]),int(cntr[1]),int(cntr[2]),int(cntr[3])
        resImage = cv2.resize(img[max(ymin,0):ymax, xmin:xmax], (112, 112)) #Cara detectada, reescalamos
        resImage = cv2.cvtColor(resImage, cv2.COLOR_BGR2GRAY)
        return resImage
    except:
        print(cntr)
        cv2.imshow('image',img)
        cv2.waitKey(0)

def saveFaceCrops(videoPath):
    vidcap = cv2.VideoCapture(videoPath)
    success,image = vidcap.read()
    count = 0
    faceArray = []
    while success:
        faceArray.append(extractBiggestFace(image)) #DE MOMENTO SIEMPRE HAY CARA
        success,image = vidcap.read()
        count += 1
    return faceArray #Devuelve número de frames



# %%
res = saveFaceCrops(inputVideo)

# %%
res[0].shape

# %%
audioPath = convert_video_to_audio_ffmpeg(inputVideo)
_,sig = wav.read(audioPath)


# %%
audio = python_speech_features.mfcc(sig, 16000, numcep = 13, winlen = 0.025, winstep = 0.010) #ASUME VIDEO A 25 Y AUDIO A 100, MODIFICAR

# %%
model = talkNet()
model.load_state_dict(torch.load("./exps/exp2/model/model_0006.model"))

# %%
model("hola")

# %%



