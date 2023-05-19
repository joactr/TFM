import os, subprocess, glob, pandas, tqdm, cv2, numpy
from scipy.io import wavfile

def init_args(args):
    # The details for the following folders/files can be found in the annotation of the function 'preprocess_AVA' below
    args.modelSavePath    = os.path.join(args.savePath, 'model')
    args.scoreSavePath    = os.path.join(args.savePath, 'score.txt')


    if args.evalDataType == 'val':
        args.evalTrialAVA = os.path.join(args.trialPathAVA, 'val_loader.csv')
        args.evalOrig     = os.path.join(args.trialPathAVA, 'val_orig.csv')  
        args.evalCsvSave  = os.path.join(args.savePath,     'val_res.csv') 
    else:
        args.evalTrialAVA = os.path.join(args.trialPathAVA, 'test_loader.csv')
        args.evalOrig     = os.path.join(args.trialPathAVA, 'test_orig.csv')    
        args.evalCsvSave  = os.path.join(args.savePath,     'test_res.csv')
    
    os.makedirs(args.modelSavePath, exist_ok = True)
    os.makedirs(args.dataPathAVA, exist_ok = True)
    return args
 

def download_pretrain_model_AVA():
    if os.path.isfile('pretrain_AVA.model') == False:
        Link = "1NVIkksrD3zbxbDuDbPc_846bLfPSZcZm"
        cmd = "gdown --id %s -O %s"%(Link, 'pretrain_AVA.model')
        subprocess.call(cmd, shell=True, stdout=None)