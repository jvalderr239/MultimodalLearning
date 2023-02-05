# -*- coding: utf-8 -*-
"""DataPreProcessing.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1iKfrE4WArlkpHOJmVrVuGR8vjg4fZ5Hr
"""

import pickle
import numpy as np
from matplotlib import pyplot as plt
import torch

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
from models import multimodal_model

import logging

# Create a custom logger
log = logging.getLogger(__name__)

# Create handlers
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler('file.log')
i_handler = logging.StreamHandler()
c_handler.setLevel(logging.WARNING)
f_handler.setLevel(logging.ERROR)
i_handler.setLevel(logging.INFO)

# Create formatters and add it to handlers
c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
i_handler.setFormatter(c_format)
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

DATAPATH = "./drive/MyDrive/Multimodal/"

drive_path = './drive/MyDrive/Multimodal/'
model_path = drive_path + 'Models/'
data_path = drive_path + 'Data/'
scripts_path = drive_path + 'Scripts/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataloader = torch.load(data_path + 'audiovisual_dataloader.pkl', pickle_module=pickle)

multimodal = multimodal_model.AudioVisualModel(
    model_path,
    device,
    data_path + 'audiovisual_dataloader.pkl'
)
audio = multimodal.load_model('audio_model.pth')
visual = multimodal.load_model('visual_model.pth')

def plot_conf_multimodal():
    """Plot multimodal metrics"""
    y_pred = []
    y_true = []
    multimodal.model.cuda()
    # iterate over test data
    for audio_inputs, visual_inputs, labels in dataloader['test']:
        with torch.no_grad():
            multimodal.model.eval()
            audio_inputs = multimodal._format_inputs(audio_inputs, 'audio', 'test')
            visual_inputs = multimodal._format_inputs(visual_inputs, 'visual', 'test')
            #inputs = torch.nn.functional.normalize(inputs, p=2.0, dim=1)
            audio_inputs = audio_inputs.to(device, non_blocking=True)
            visual_inputs = visual_inputs.to(device, non_blocking=True)
            labels = (labels.float()).to(device, non_blocking=True)
            preds, _ = multimodal.model.forward(audio_inputs, visual_inputs, labels) # Feed Network

            _, probs = torch.max(preds, 1)
            _, targets = torch.max(labels, 1)
            y_pred.extend(probs.cpu().numpy()) # Save Prediction

            y_true.extend(targets.cpu().numpy()) # Save Truth

    # constant for classes
    classes = ('neutral', 'calm','happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised')

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix, axis=1), index = [i for i in classes],
                        columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('fusion_confusion.png')