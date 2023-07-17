import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os

import sys, time, numpy, os, subprocess, pandas, tqdm
from subprocess import PIPE

from lightLoss import lossAV, lossV
from model.Model import ASD_Model

class ASD(nn.Module):
    def __init__(self, lr = 0.001, lrDecay = 0.95, **kwargs):
        super(ASD, self).__init__()        
        self.model = ASD_Model().cuda()
        self.lossAV = lossAV().cuda()
        self.lossV = lossV().cuda()
        self.optim = torch.optim.Adam(self.parameters(), lr = lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size = 1, gamma=lrDecay)
        print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f"%(sum(param.numel() for param in self.model.parameters()) / 1000 / 1000))

    def train_network(self, loader, epoch, **kwargs):
        self.train()
        self.scheduler.step(epoch - 1)  # StepLR
        index, top1, lossV, lossAV, loss = 0, 0, 0, 0, 0
        lr = self.optim.param_groups[0]['lr']
        r = 1.3 - 0.02 * (epoch - 1)
        for num, (audioFeature, visualFeature, labels) in enumerate(loader, start=1):
            self.zero_grad()

            audioEmbed = self.model.forward_audio_frontend(audioFeature.cuda())
            visualEmbed = self.model.forward_visual_frontend(visualFeature.cuda())

            outsAV= self.model.forward_audio_visual_backend(audioEmbed, visualEmbed)  
            outsV = self.model.forward_visual_backend(visualEmbed)

            labels = labels.reshape((-1)).cuda() # Loss
            nlossAV, _, _, prec = self.lossAV.forward(outsAV, labels, r)
            nlossV = self.lossV.forward(outsV, labels, r)
            nloss = nlossAV + 0.5 * nlossV

            lossV += nlossV.detach().cpu().numpy()
            lossAV += nlossAV.detach().cpu().numpy()
            loss += nloss.detach().cpu().numpy()
            top1 += prec
            nloss.backward()
            self.optim.step()
            index += len(labels)
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
            " [%2d] r: %2f, Lr: %5f, Training: %.2f%%, "    %(epoch, r, lr, 100 * (num / loader.__len__())) + \
            " LossV: %.5f, LossAV: %.5f, Loss: %.5f, ACC: %2.2f%% \r"  %(lossV/(num), lossAV/(num), loss/(num), 100 * (top1/index)))
            sys.stderr.flush()  

        sys.stdout.write("\n")      

        return loss/num, lr

    def evaluate_network(self, epoch, loader,  **kwargs): #evalCsvSave, evalOrig,
        self.eval()
        windowSize = kwargs.get('windowSize',24)
        predScores, predLabels = [], []
        index, top1, lossV, lossAV, loss = 0, 0, 0, 0, 0
        r = 1.3 - 0.02 * (epoch - 1)
        for num, (audioFeature, visualFeature, labels) in enumerate(tqdm.tqdm(loader)):
            with torch.no_grad():                
                audioEmbed  = self.model.forward_audio_frontend(audioFeature.cuda())
                visualEmbed = self.model.forward_visual_frontend(visualFeature.cuda())
                outsAV= self.model.forward_audio_visual_backend(audioEmbed, visualEmbed)  
                outsV = self.model.forward_visual_backend(visualEmbed)
                labels = labels.reshape((-1)).cuda()             
                
                nlossAV, predScore, predLabel, prec = self.lossAV.forward(outsAV, labels)    
                nlossV = self.lossV.forward(outsV, labels, r)
                nloss = nlossAV + 0.5 * nlossV

                lossV += nlossV.detach().cpu().numpy()
                lossAV += nlossAV.detach().cpu().numpy()
                loss += nloss.detach().cpu().numpy()
                predScore = predScore[:,1].detach().cpu().numpy()
                predScores.extend(predScore)
                predLabels.extend(predLabel.detach().cpu().numpy().astype(int))
                #Evaluacion
                top1 += prec
                index += len(labels)
                # break
        print(len(predScores),len(predLabels))
        acc = (100 * (top1/index)).detach().cpu().numpy()
        sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
            "ACC: %2.2f%% \r"  %(acc))
        
        df = pd.read_csv("../testSamples.csv")
        df = df.loc[df.index.repeat(windowSize)].reset_index(drop=True)
        df["pred"] = predLabels
        df["posScore"] = predScores
        df.index.name = 'uid'
        df.to_csv("testPreds.csv")

        cmd = "python -O ../get_map.py -p testPreds.csv"
        mAP = float(str(subprocess.check_output(cmd)).split(' ')[2][:5])
        print(mAP)
        return loss/num, acc, mAP

    def saveParameters(self, path):
        torch.save(self.state_dict(), path)

    def loadParameters(self, path):
        selfState = self.state_dict()
        loadedState = torch.load(path)
        for name, param in loadedState.items():
            origName = name;
            if name not in selfState:
                name = name.replace("module.", "")
                if name not in selfState:
                    print("%s is not in the model."%origName)
                    continue
            if selfState[name].size() != loadedState[origName].size():
                sys.stderr.write("Wrong parameter length: %s, model: %s, loaded: %s"%(origName, selfState[name].size(), loadedState[origName].size()))
                continue
            selfState[name].copy_(param)
