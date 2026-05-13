import warnings
warnings.filterwarnings("ignore")

import os, sys
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from ESM3dG import ESM3dG, ESM3dG_predict

import logging
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

ALPHABET = 'ACDEFGHIKLMNPQRSTVWY-'

PDB_NAME = 'nanobody_1zvh'
CHAIN_ID = 'A'
WEIGHTS_DIR = os.path.join(_ROOT, "esm3dg_weights")
WEIGHT_FILES = [
    "ESM3dG_weights_augmented_1_lora.ckpt",
    "ESM3dG_weights_augmented_2_lora.ckpt",
    "ESM3dG_weights_augmented_3_lora.ckpt",
]

pdb_path = os.path.join(_ROOT, "examples", "nanobody_1zvh.cif")

all_mean_results = []
for i, weight_file in enumerate(WEIGHT_FILES, 1):
    weight_path = f"{WEIGHTS_DIR}/{weight_file}"
    model = ESM3dG(weight_path)
    ddg_scan, scaled_ddg_scan, sequence = ESM3dG_predict(model, pdb_path, CHAIN_ID, ddg_scanning=True)
    mean_results = np.mean(scaled_ddg_scan.detach().cpu().numpy(), axis=-1).T[0]
    all_mean_results.append(mean_results)

    df = pd.DataFrame(mean_results, index=list(ALPHABET[:-1]), columns=range(1, mean_results.shape[1] + 1))
    df.to_csv(f'mutational_scanning_{PDB_NAME}_model{i}.csv')

    height, width = mean_results.shape
    fig_width = min(max(width / 4, 8), 25)
    fig_height = min(max(height / 4, 3), 8)
    plt.figure(figsize=(fig_width, fig_height))
    divergence = max(abs(mean_results.min()), abs(mean_results.max()))
    plt.imshow(mean_results, aspect='auto', cmap='bwr', vmin=-divergence, vmax=divergence)
    plt.xticks(ticks=np.arange(width), labels=np.arange(1, width + 1), rotation=90)
    plt.yticks(ticks=range(len(ALPHABET) - 1), labels=ALPHABET[:-1])
    plt.ylabel("Amino Acids")
    plt.title(f"Mutational Scanning {PDB_NAME} - Model {i}")
    plt.colorbar()
    plt.savefig(f'mutational_scanning_{PDB_NAME}_model{i}.png', dpi=300, bbox_inches='tight')
    plt.show()

# Ensemble
mean_results = np.mean(all_mean_results, axis=0)
height, width = mean_results.shape

df = pd.DataFrame(mean_results, index=list(ALPHABET[:-1]), columns=range(1, width + 1))
df.to_csv(f'mutational_scanning_{PDB_NAME}_ensemble.csv')

fig_width = min(max(width / 4, 8), 25)
fig_height = min(max(height / 4, 3), 8)
plt.figure(figsize=(fig_width, fig_height))
divergence = max(abs(mean_results.min()), abs(mean_results.max()))
plt.imshow(mean_results, aspect='auto', cmap='bwr', vmin=-divergence, vmax=divergence)
plt.xticks(ticks=np.arange(width), labels=np.arange(1, width + 1), rotation=90)
plt.yticks(ticks=range(len(ALPHABET) - 1), labels=ALPHABET[:-1])
plt.ylabel("Amino Acids")
plt.title(f"Mutational Scanning {PDB_NAME} - Ensemble")
plt.colorbar()
plt.show()
