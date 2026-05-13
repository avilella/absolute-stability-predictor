import os
import sys
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.chdir("/home/jupyter-yehlin/esm3_attention")
sys.path.append("/home/jupyter-yehlin/esm3_attention")

import sys
import numpy as np
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from torchmetrics import MeanSquaredError, R2Score, SpearmanCorrCoef, PearsonCorrCoef
from omegaconf import OmegaConf

import pandas as pd
import pickle
import argparse
from math import isnan
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional

import random

# ESM3 imports
from huggingface_hub import login
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig
from esm.utils.structure.protein_chain import ProteinChain
from Transfer_esm3_lora_bins_sigmoid_both_full import TransferModel


# Directory containing your nanobody .cif or .pdb files
NANOBODY_PDB_DIR = "/home/jupyter-yehlin/Pairformer/nanobody_boltz2/nanobody_boltz2_results"
OUTPUT_CSV = "/home/jupyter-yehlin/ESM3_SaProt_dG/esm3dG_nanobody_predictions.csv"

# List of ESM3 checkpoint weights for ensembling
WEIGHT_FILES = [
    "/home/jupyter-yehlin/esm3_attention/weights/checkpoint_unfreeze_warum_up_ddg_bins_lora_dg_and_ddg_sigmoid_5e5_model1_resume_model1_norm/Fine_Tune_ESM3_dmsv4_AF_unfreeze_warm_up_ddG_revised_esm3_epoch=03_val_ddG_spearman=0.58_val_ddG_mse=0.2_val_dG_spearman=0.86_val_dG_mse=0.69.ckpt",
    "/home/jupyter-yehlin/esm3_attention/weights/checkpoint_unfreeze_warum_up_ddg_bins_lora_dg_and_ddg_sigmoid_5e5_model2_resume_model1_norm/Fine_Tune_ESM3_dmsv4_AF_unfreeze_warm_up_ddG_revised_esm3_epoch=07_val_ddG_spearman=0.58_val_ddG_mse=0.2_val_dG_spearman=0.87_val_dG_mse=0.68.ckpt",
    "/home/jupyter-yehlin/esm3_attention/weights/checkpoint_unfreeze_warum_up_ddg_bins_lora_dg_and_ddg_sigmoid_5e5_model3_resume_model2_norm/Fine_Tune_ESM3_dmsv4_AF_unfreeze_warm_up_ddG_revised_esm3_epoch=01_val_ddG_spearman=0.57_val_ddG_mse=0.2_val_dG_spearman=0.86_val_dG_mse=0.72.ckpt"

]

def get_esm3_encoding(pdb_path, model_esm_encode, chain_id='A'):
    """Extracts structural and sequence encoding from a local CIF/PDB file."""
    try:
        # 1. Parse coordinates
        coords = parse_CIF(pdb_path, input_chain_list=chain_id)
        if coords is None: return None
        
        # 2. Extract sequence from the structure using gemmi
        st = gemmi.read_structure(pdb_path)
        sequence = ""
        for model in st:
            for chain in model:
                if chain.name == chain_id:
                    sequence = chain.get_polymer().get_sequence()
                    break
        
        if not sequence or len(sequence) != coords.shape[0]:
            return None

        # 3. Create ESMProtein object and encode
        protein_prompt = ESMProtein(sequence=sequence, coordinates=torch.tensor(coords))
        with torch.no_grad():
            encoder = model_esm_encode.encode(protein_prompt)
        
        return {
            'seq': encoder.sequence,
            'struct': encoder.structure,
            'coord': encoder.coordinates
        }
    except Exception as e:
        print(f"Error encoding {pdb_path}: {e}")
        return None

def main():
    # 1. Setup Config (Matches your test_esm3_TED setup)
    base_cfg = Config(
        model={"hidden_dims": [64, 32], "subtract_mut": True, "num_final_layers": 2, 
               "freeze_weights": True, "load_pretrained": True, "lightattn": True},
        training={"learn_rate": 0.001, "warm_up": True}
    )

    # 2. Load ESM3 Encoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading ESM3 encoder...")
    model_esm_encode = ESM3.from_pretrained("esm3_sm_open_v1").to(device).eval()

    # 3. Load Ensemble Models
    prediction_models = []
    for weight_path in WEIGHT_FILES:
        print(f"Loading weights: {os.path.basename(weight_path)}")
        model_pl = TransferModelPL.load_from_checkpoint(
            weight_path, cfg=base_cfg, strict=False
        ).model.to(device).eval()
        prediction_models.append(model_pl)

    # 4. Process Directory
    results = []
    pdb_files = [f for f in os.listdir(NANOBODY_PDB_DIR) if f.endswith(('.cif', '.pdb'))]
    
    print(f"Processing {len(pdb_files)} files...")
    for pdb_file in tqdm(pdb_files):
        pdb_path = os.path.join(NANOBODY_PDB_DIR, pdb_file)
        pdb_name = os.path.splitext(pdb_file)[0]
        
        # Get ESM3 tokens/embeddings
        info_dict = get_esm3_encoding(pdb_path, model_esm_encode, chain_id='A')
        if info_dict is None:
            continue

        # Run inference through each ensemble member
        model_preds = []
        with torch.no_grad():
            for model in prediction_models:
                # model expects a list of info_dicts
                pred_dg, _, mask = model([info_dict])
                
                # Calculate average dG per residue (matching your TED logic)
                mask = mask.cpu()
                avg_dg = (pred_dg.cpu() * mask).sum() / mask.sum()
                model_preds.append(avg_dg.item())

        # Calculate Statistics
        avg_val = np.mean(model_preds)
        results.append([pdb_name] + model_preds + [avg_val])

    # 5. Save Results
    columns = ['PDB'] + [f'Model_{i+1}' for i in range(len(WEIGHT_FILES))] + ['Average']
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone! Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()