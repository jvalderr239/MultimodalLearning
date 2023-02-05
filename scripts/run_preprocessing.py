# -*- coding: utf-8 -*-
""" Module to preprocess audiovisual data into DataLoader format"""
import logging
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.dataset import AudioVisualDataset
from utils import load_processed_data

# Create a custom logger
log = logging.getLogger(__name__)

# Create handlers
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler("file.log")
i_handler = logging.StreamHandler()
c_handler.setLevel(logging.WARNING)
f_handler.setLevel(logging.ERROR)
i_handler.setLevel(logging.INFO)

# Create formatters and add it to handlers
c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
i_handler.setFormatter(c_format)
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

SEED = 239
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATAPATH = "./drive/MyDrive/Multimodal/"

t_a, t_v, t_y = load_processed_data("train")
v_a, v_v, v_y = load_processed_data("val")
test_a, test_v, test_y = load_processed_data("test")

data_sets = {
    "train": AudioVisualDataset(t_a, t_v, t_y),
    "val": AudioVisualDataset(v_a, v_v, v_y),
    "test": AudioVisualDataset(test_a, test_v, test_y),
}

log.info("Making dataloaders")
data_loaders = {
    data_key: DataLoader(data_val, batch_size=30, shuffle=True, num_workers=4)
    for data_key, data_val in data_sets.items()
}
log.info("Saving dataloaders")
torch.save(data_loaders, DATAPATH + "audiovisual_dataloader.pkl")
