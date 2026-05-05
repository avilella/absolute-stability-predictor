import warnings
warnings.filterwarnings("ignore")

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
  
cfg = get_default_config()
ddg_scanning = True 
cfg.testing.ddg_scanning = ddg_scanning

PDB_NAME = '1lci'
CHAIN_ID = 'A'
WEIGHTS_DIR = "/home/jupyter-yehlin/ESM3_SaProt_dG/saprotdg_weights"
WEIGHT_FILES = [
    "SaProtdG_weights_1_lora.ckpt",
    "SaProtdG_weights_2_lora.ckpt",
    "SaProtdG_weights_3_lora.ckpt"
]  
pdb_path = get_pdb(PDB_NAME)
frag1='M#E#D#'
frag2='AdKpNqIkKfKaGaPdAaPfFpYdPhLqEdDdGfTfAlGlEqQlLlHlKvAlMlKvRvYqAlLvVpPpGpTlIfAlFeTaDaAlHvIvEgVdNtIdTgYsAvEnYlFlElMlSlVlRlLlAlEqAlMvKvRqYvGpLdNaTlNvHfReIeVeVeCaSeEfNdSdLpQqFrFcMsPnVlLsGsAcLsFlIrGlVhAeVyAePyAqNyDnIvYqNaElRvEcLvLlNlSsMcNvIlSrQlPhTqVeVyFeVySaKvKvGcLvQvKsIvLvNvVsQcKvKvLrPvIsIhQpKaIyIeIhMrDpSdKcTqDaYdQpGnFhQhShMsYnTvFsVsTvSvHgLdPdPpGpFdNdEsYvDvFrVhPgEdStFdDpRqDaKpTgIfAsLyIqMaNkSp'
frag3='S#G#S#T#G#'
frag4='LvPtKfGtVfAtLwPgHsRqTlAlCsVvRlFlSsHqAcRcDgPvIwFnGhNnQpIlIdPpDlTaAeIeLeSeVaVdPgFcHsHhGlFqGnMvFnTnTsLsGnYvLsIsCsGsFyRyVyVyLyMyYrRdFdEdEvEvLsFvLlRvScLlQaDvYsKlIhQqSeAyLeLdVaPqTvLvFlSvFvFqAlKpSdTpLcInDvKvYgDdLnSvNnLhHaEaIyAeShGkGqAdPfLySaKpEvVsGqEvAsVnAcKvRsFnHvLyPpGtIyRkQaGaYhGdLgTvEqTlTsStAgIqLaIiTaPtEpGp'
frag5='D#D#K#'
frag6='PrGrAqVrGhKaVgVhPhFqFkEiAkKfVfVaDaLqDpTpGrKhTtLdGhVaNpQdRkGhEfLiCkVmRaGgPrMsIgMtSpGaYtVrNpNpPvEpAvTrNpAvLqIaDdKpDvGrWiLgHgSpGlDwIiAwYgWaDhEpDvErHgFiFdItVd'
frag7='D#R#L#K#S#'
frag8='LfIfKqYfKqGnYdQtVdAdPqAvEvLlEqSvIlLqLcQpHdPqNqIfFpDgAkGgVwAeGfLqPaDdDvDvArGrEtLaPiAeAmVeVtVaLtEdHpGpKhTdMdTdEqKvErIsVqDvYsVsApSvQpVdTdTvArKhKgLrRpGlGtVyVhFyVdDnEdVqPc'
frag9='K#G#L#T#G#'
frag10='KvLhDsArRvKvIrRnEvIvLsInKvAvKrKd'
frag11='G#G#K#S#K#L#'
# combined_seq_1lci = frag1 + frag2 + frag3 + frag4 + frag5 + frag6 + frag7 + frag8 + frag9 + frag10 + frag11
combined_seq_1lci = frag1 + frag2 + frag3 + frag4 + frag5 + frag6

all_mean_results = []
for i, weight_file in enumerate(WEIGHT_FILES, 1):
    weight_path = f"{WEIGHTS_DIR}/{weight_file}"
    model = SaProtABS(weight_path, cfg)
    pred_mutant_ddg, scaled_pred_mutant_ddg, combined_seq = SaProtABS_predict(model, pdb_path, CHAIN_ID, ddg_scanning, given_seq=combined_seq_1lci)
    mean_results = np.mean(scaled_pred_mutant_ddg.detach().cpu().numpy(), axis=-1).T[0]
    all_mean_results.append(mean_results)
    
    # Save individual model results
    df = pd.DataFrame(mean_results, index=list(ALPHABET[:-1]), columns=range(1, mean_results.shape[1] + 1))
    df.to_csv(f'mutational_scanning_{PDB_NAME}_model{i}_truncated.csv')
    
    # Plot individual model results
    height, width = mean_results.shape
    fig_width = min(max(width/4, 8), 25)
    fig_height = min(max(height/4, 3), 8)
    plt.figure(figsize=(fig_width, fig_height))
    
    vmin = mean_results.min()
    vmax = mean_results.max()
    divergence = max(abs(vmin), abs(vmax))
    
    plt.imshow(mean_results, 
               aspect='auto',
               cmap='bwr',
               vmin=-divergence,
               vmax=divergence)
    
    L = mean_results.shape[1]
    plt.xticks(ticks=np.arange(L), labels=np.arange(1, L + 1), rotation=90)
    plt.yticks(ticks=range(len(ALPHABET)-1), labels=ALPHABET[:-1])
    plt.ylabel("Amino Acids")
    plt.title(f"Mutational Scanning {PDB_NAME} - Model {i}")
    plt.colorbar()
    plt.savefig(f'/home/jupyter-yehlin/ESM3_SaProt_dG/mutational_scanning_{PDB_NAME}_model{i}_truncated.png', dpi=300, bbox_inches='tight')
    plt.show()

# Average results across models
mean_results = np.mean(all_mean_results, axis=0)
height, width = mean_results.shape

# Save ensemble results
df = pd.DataFrame(mean_results, index=list(ALPHABET[:-1]), columns=range(1, width + 1))
df.to_csv(f'/home/jupyter-yehlin/ESM3_SaProt_dG/mutational_scanning_{PDB_NAME}_ensemble_truncated.csv')

# Plot ensemble results
fig_width = min(max(width/4, 8), 25)
fig_height = min(max(height/4, 3), 8)
plt.figure(figsize=(fig_width, fig_height))

vmin = mean_results.min()
vmax = mean_results.max()
divergence = max(abs(vmin), abs(vmax))

plt.imshow(mean_results, 
           aspect='auto',
           cmap='bwr',
           vmin=-divergence,
           vmax=divergence)

L = mean_results.shape[1]
plt.xticks(ticks=np.arange(L), labels=np.arange(1, L + 1), rotation=90)
plt.yticks(ticks=range(len(ALPHABET)-1), labels=ALPHABET[:-1])
plt.ylabel("Amino Acids")
plt.title(f"Mutational Scanning {PDB_NAME} - Ensemble")
plt.colorbar()
plt.show()