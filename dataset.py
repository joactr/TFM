import os
import random
import torchaudio
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import cv2
from tqdm import tqdm
def randomNoOverlap(videoCenter, audioLen, treshold, nSideFrames):
    """
    Threshold entre 0 y 1, porcentaje maximo de overlap permitido
    No funciona para muestras muy pequeñas, echar ojo
    """
    windowSize = nSideFrames*2+1
    while True:
        index = random.randint(0, audioLen)
        overlap = False
        if abs(videoCenter-index/4) < windowSize*treshold:
            overlap = True
        if not overlap:
            return index

class MyDataset(Dataset):

    def __init__(self, nframes, video_dir, audio_dir, csv_path):
        """
            nframes: descartamos videos que superen dicho nº de frames
            video_dir: directorio donde se encuentran almacenados los videos
            csv_path: fichero csv que define una partición
        """
        self.nframes = nframes
        self.video_dir = video_dir
        self.audio_dir = audio_dir

        samples_data = pd.read_csv(csv_path, delimiter=",")
        #if "train" in csv_path:
        #    samples_data = samples_data[samples_data["nFrames"]< nframes]
        
        self.videoIDs = samples_data["video"].tolist()
        self.audioIDs = samples_data["audio"].tolist()
        self.labels = samples_data["label"].tolist()
        self.centers = samples_data["center"].tolist()
        self.nSideFrames = int((self.nframes-1)/2)
        self.nSideFramesAudio = self.nSideFrames*4
        print(self.nSideFrames,self.nSideFramesAudio)
        #TESTING
        self.ini = 0
        self.fin = 0

    def __len__(self):
        return len(self.videoIDs)

    def __getitem__(self, index):
        videoID = self.videoIDs[index]
        audioID = self.audioIDs[index]
        label = self.labels[index]
        center = self.centers[index]

        video = self.__get_video__(videoID, audioID, label, center)
        audio = self.__get_audio__(audioID, videoID, label, center)
        if label == 1: #Positiva
            label = np.ones(self.nframes)
        else:
            label = np.zeros(self.nframes)
        label = torch.LongTensor(label)
        return audio, video, label

    def __get_video__(self, videoID, audioID, label, center):
        spkrID = videoID.split("_")[0]
        video_path = os.path.join(self.video_dir, spkrID, videoID + ".npz")
        video = np.load(video_path)["images"]
        videoFrames = video.shape[0]
        video = torch.FloatTensor(video)
        ini = center-self.nSideFrames
        fin = center+self.nSideFrames+1
        #print(video.shape)
        if center < self.nSideFrames: #Necesitamos hacer padding por la izquierda
            padAmount = self.nSideFrames - center
            video = F.pad(video,(0,0,0,0,padAmount,0), "constant", 0) #Padding al principio
            ini = 0
            fin = self.nframes
            #print("pad izquierda:",video.shape, padAmount)
        if center+self.nSideFrames >= videoFrames: #Necesitamos hacer padding al final
            padAmount = (self.nSideFrames+center) - videoFrames+1
            video = F.pad(video,(0,0,0,0,0,padAmount), "constant", 0) #Padding al final
            ini = len(video)-(self.nframes)
            #print("pad derecha:",video.shape, padAmount)
        video = video[ini:fin]

        # self.ini = ini
        # self.fin = fin
        # if(label ==1):
        #     print("inifinVideo",ini,fin)
        # print(len(video))
        # if len(video) != 51:
        #    print("fake",len(video))
        return video # (T,96,96)
    
    def __get_audio__(self, audioID, videoID, label, center):
        spkrID = audioID.split("_")[0]
        audio_path = os.path.join(self.audio_dir, spkrID, audioID + ".npz")
        audio = np.load(audio_path)["mfcc"]
        audio = torch.FloatTensor(audio)
        audioFrames = audio.shape[0]
        if label == 1: #Muestra positiva
            center = center*4
        if label == 0: #Muestra negativa
            if audioID == videoID: #Audio desfasado
                center = randomNoOverlap(center, audioFrames, 0.5, self.nSideFrames)
            else:
                center = random.randint(0,len(audio))
        ini = center-self.nSideFramesAudio
        fin = center+self.nSideFramesAudio+4
        if center < self.nSideFramesAudio: #Necesitamos hacer padding por la izquierda
            padAmount = self.nSideFramesAudio - center
            audio = F.pad(audio,(0,0,padAmount,0), "constant", 0) #Padding al principio
            ini = 0
            fin = self.nframes*4
            #print("pad izquierda audio:",audio.shape, padAmount)
        if center+self.nSideFramesAudio+4 >= audioFrames: #Necesitamos hacer padding al final
            padAmount = (self.nSideFramesAudio+center) - audioFrames+4
            audio = F.pad(audio,(0,0,0,padAmount), "constant", 0) #Padding al final
            ini = len(audio)-(self.nframes*4)
            #print("pad derecha audio:",audio.shape, padAmount)
            
        audio = audio[ini:fin]
        # if audio.shape[0] != 44:
        #     print("center",center,audioID==videoID)
        #     print("fakeaudio:",ini,fin,audio.shape[0])
        #if (ini!=self.ini*4 or fin!=self.fin*4) and label == 1:
        #    print("Error")

        return audio # (T,96,96)
# videoDir = "C:/Users/jmmol/Desktop/COSAS V7/TFM/npz"
# audioDir = "C:/Users/jmmol/Desktop/COSAS V7/TFM/mfccs"
# dataset = MyDataset(5,videoDir,audioDir,"trainSamples.csv")
# # for i in tqdm(range(100000)):
# #     item = dataset.__getitem__(i)
# # cv2.imshow("test",item[0][0])
# # cv2.waitKey(0)

# dataloader = DataLoader(dataset=dataset,batch_size=128,num_workers=8) #Cambiar num_workers
# dataloader_iterator = iter(dataloader)
# X = next(dataloader_iterator)

