import scipy.io.wavfile as wav
import pandas as pd
import torch.nn.functional as F
from talkNet import talkNet
from dataset import MyDataset
from torch.utils.data import DataLoader
import tqdm
import torch

if __name__ == '__main__':
    model = talkNet()
    #model.load_state_dict(torch.load("./exps/exp2/model/model_0002.model"))
    model.load_state_dict(torch.load("./exps/exp1/model/model21_0006.model"))
    windowSize = 21

    videoDir = "C:/Users/jmmol/Desktop/COSAS V7/TFM/npz"
    audioDir = "C:/Users/jmmol/Desktop/COSAS V7/TFM/mfccs"
    datasetTrain = MyDataset(windowSize,videoDir,audioDir,"testSamples.csv")
    item = datasetTrain.__getitem__(0)
    valLoader = DataLoader(dataset=datasetTrain,shuffle=False,batch_size=32,num_workers=14) #Cambiar num_workers


    correctFrames, totalFrames = 0, 0
    correctSamples, totalSamples = 0, 0
    totalPreds = []
    totalScores = []
    totalFramePreds = []
    totalFrameScores = []

    # for num, (audioFeature, visualFeature, labels) in enumerate(tqdm.tqdm(valLoader)):
    #     with torch.no_grad():    
    #         predScores,predLabels = model((audioFeature,visualFeature))
    #         labels = labels.cuda()
    #         batchPreds = torch.reshape(predLabels, labels.shape)
    #         #Precision a nivel de video
    #         videoPreds = torch.mode(batchPreds,dim=1)[0]
    #         totalPreds.extend(videoPreds.detach().cpu().numpy().astype(int))
    #         for ten in torch.split(predScores[:,1],windowSize):
    #             totalScores.append(torch.mean(ten).detach().cpu().numpy())
    #         labelMode = torch.mode(labels,dim=1)[0]
    #         correctSamples += (videoPreds == labelMode).sum().float()
    #         totalSamples += len(labels)
    #         # Precision a nivel de frame
    #         labels = labels.reshape((-1))
    #         correctFrames += (predLabels == labels).sum().float()
    #         totalFrames += len(labels) 

    for num, (audioFeature, visualFeature, labels) in enumerate(tqdm.tqdm(valLoader)):
        with torch.no_grad():    
            predScores,predLabels = model((audioFeature,visualFeature))
            labels = labels.cuda()
            batchPreds = torch.reshape(predLabels, labels.shape)
            totalPreds.extend(batchPreds.detach().cpu().numpy().astype(int))
            for ten in torch.split(predScores[:,1],windowSize):
                totalScores.append(ten.detach().cpu().numpy())
            correctSamples += (batchPreds == labels).sum().float()
            labels = labels.reshape((-1))
            totalSamples += len(labels)
            # Precision a nivel de frame
            correctFrames += (predLabels == labels).sum().float()
            totalFrames += len(labels) 
    #7865
    print("Frame acc:", correctFrames/totalFrames)
    #print("Sample acc:", correctSamples/totalSamples)




