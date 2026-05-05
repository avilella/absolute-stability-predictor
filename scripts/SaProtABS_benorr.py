import warnings
warnings.filterwarnings("ignore")

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import MeanSquaredError, R2Score, SpearmanCorrCoef, PearsonCorrCoef
from omegaconf import OmegaConf

from utils.foldseek_util import get_struc_seq
import torch
from torch.utils.data import ConcatDataset
import pandas as pd
import numpy as np
import pickle

from Bio import pairwise2
from math import isnan
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional
from torch.utils.data import ConcatDataset
import numpy as np
import matplotlib.pyplot as plt


ALPHABET = 'ACDEFGHIKLMNPQRSTVWY-'
from config import get_default_config, get_model_configs
from SaProtABS import SaProtABS, SaProtABS_predict

import logging
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

def get_pdb(pdb_code=""):
  if pdb_code is None or pdb_code == "":
    upload_dict = files.upload()
    pdb_string = upload_dict[list(upload_dict.keys())[0]]
    with open("tmp.pdb","wb") as out: out.write(pdb_string)
    return "tmp.pdb"
  elif os.path.isfile(pdb_code):
    return pdb_code
  elif len(pdb_code) == 4:
    os.system(f"wget -qnc https://files.rcsb.org/view/{pdb_code}.pdb")
    return f"{pdb_code}.pdb"
  else:
    os.system(f"wget -qnc https://alphafold.ebi.ac.uk/files/AF-{pdb_code}-F1-model_v3.pdb")
    return f"AF-{pdb_code}-F1-model_v3.pdb"
  
from SaProtABS import SaProtABS, SaProtABS_predict

cfg = get_default_config()
cfg.testing.ddg_scanning = False
WEIGHTS_DIR = "/home/jupyter-yehlin/ESM3_SaProt_dG/saprotdg_weights"
WEIGHT_FILES = [
    "SaProtdG_weights_1_lora.ckpt",
    "SaProtdG_weights_2_lora.ckpt", 
    "SaProtdG_weights_3_lora.ckpt"
]

pdb_ls_path = "/home/jupyter-yehlin/Pairformer/Benorr/round3_ordered_designs_af2_preds"

# Load all models once upfront
models = []
for weight_file in WEIGHT_FILES:
    weight_path = f"{WEIGHTS_DIR}/{weight_file}"
    model = SaProtABS(weight_path, cfg)
    models.append(model)

# Create DataFrame to store results
results_df = pd.DataFrame(columns=['PDB', 'Model1', 'Model2', 'Model3', 'Average'])

for pdb_file in os.listdir(pdb_ls_path):
    if pdb_file.endswith(".pdb"):
        pdb_path = os.path.join(pdb_ls_path, pdb_file)
        PDB_NAME = pdb_file.split(".")[0]
        CHAIN_ID = 'A'

        # Run predictions for each model and collect results
        predictions = []
        for i, (model, weight_file) in enumerate(zip(models, WEIGHT_FILES), 1):
            pred_dg_per_res, pred_dg_avg, combined_seq = SaProtABS_predict(model, pdb_path, CHAIN_ID)
            pred_value = pred_dg_avg[0]
            
            predictions.append(pred_value)
            print(f"🧬 Model {i}: {PDB_NAME} Predicted ΔG (kcal/mol) using {weight_file}: {pred_value:.2f}")

        avg_pred = sum(predictions) / len(predictions)
        std_pred = (sum((x - avg_pred) ** 2 for x in predictions) / len(predictions)) ** 0.5
        print(predictions, avg_pred, std_pred)

        # Add results to DataFrame
        results_df.loc[len(results_df)] = [PDB_NAME] + predictions + [avg_pred]

# Save results to CSV
results_df.to_csv('/home/jupyter-yehlin/ESM3_SaProt_dG/model_predictions_benorr_af2_prediction_r2.csv', index=False)
print("\nResults saved to model_predictions_benorr_af2_prediction_r2.csv")