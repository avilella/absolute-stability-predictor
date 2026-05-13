import warnings
warnings.filterwarnings("ignore")

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import MeanSquaredError, R2Score, SpearmanCorrCoef, PearsonCorrCoef
from omegaconf import OmegaConf

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

import logging
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

ALPHABET = 'ACDEFGHIKLMNPQRSTVWY-'
from utils.config import get_default_config, get_model_configs
from SaProtABS import SaProtABS, SaProtABS_predict

import logging
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

def cif_to_pdb(cif_file, pdb_out_path):
    """
    Convert an mmCIF file to a PDB using Biopython MMCIFParser and PDBIO.
    This function tries to be independent of the presence/absence of '_atom_site.occupancy'
    by patching missing occupancies to 1.00 when the field is missing or contains 'None'.
    """
    import warnings
    warnings.filterwarnings("ignore")
    from Bio.PDB import MMCIFParser, PDBIO, Select

    # Helper: Patch occupancy if missing or malformed
    def patch_occupancy_in_cif(cif_file, tmp_fixed_cif):
        """
        Reads the cif file, detects _atom_site.occupancy issues, 
        and fills in default values if missing.
        Returns path to patched file (or original if not needed).
        """
        try:
            with open(cif_file, "r") as fin:
                lines = fin.readlines()
            if any("_atom_site.occupancy" in line for line in lines):
                # The occupancy field exists, pass through
                with open(tmp_fixed_cif, "w") as fout:
                    fout.writelines(lines)
                return tmp_fixed_cif
            # Find atom_site loop
            atom_site_start = None
            for ix, line in enumerate(lines):
                if line.strip().startswith("loop_"):
                    # Find all lines in loop
                    look_ahead = 1
                    loop_fields = []
                    while (ix+look_ahead) < len(lines) and lines[ix+look_ahead].startswith("_"):
                        loop_fields.append(lines[ix+look_ahead].strip())
                        look_ahead += 1
                    if "_atom_site.label_atom_id" in loop_fields and "_atom_site.Cartn_x" in loop_fields:
                        atom_site_start = ix
                        break
            if atom_site_start is None:
                # Can't find atom_site loop, just pass through
                with open(tmp_fixed_cif, "w") as fout:
                    fout.writelines(lines)
                return tmp_fixed_cif
            # Insert occupancy after x/y/z/label fields
            atom_site_fields = []
            loop_ix = atom_site_start+1
            while loop_ix < len(lines) and lines[loop_ix].startswith("_"):
                atom_site_fields.append(lines[loop_ix].strip())
                loop_ix += 1
            occupancy_field = "_atom_site.occupancy"
            insert_pos = len(atom_site_fields)
            patched_fields = atom_site_fields.copy()
            if occupancy_field not in patched_fields:
                patched_fields.append(occupancy_field)
            # Patch fields
            n_fields = len(patched_fields)
            # Now patch values: read, add occupancy=1.0
            out_lines = []
            out_lines.extend(lines[:atom_site_start+1])
            for f in patched_fields:
                out_lines.append(f + "\n")
            i = atom_site_start + 1 + len(atom_site_fields)
            # Write data lines
            while i < len(lines):
                l = lines[i]
                if l.strip().startswith("_") or l.strip().startswith("loop_") or l.strip() == "":
                    break
                parts = l.strip().split()
                # If this looks like a data row from atom_site (either matches original field count or patched)
                if len(parts) == len(atom_site_fields):
                    # Add occupancy at the end
                    parts.append("1.00")
                elif len(parts) == n_fields:
                    # Already has an occupancy, pass through
                    pass
                else:
                    # Not an atom_site row, just break
                    break
                out_lines.append(" ".join(parts) + "\n")
                i += 1
            # Finish up any later lines unrelated to atom_site
            out_lines.extend(lines[i:])
            with open(tmp_fixed_cif, "w") as fout:
                fout.writelines(out_lines)
            return tmp_fixed_cif
        except Exception:
            # If any error, just return original (and MMCIFParser may still fail, but that's ok)
            return cif_file

    parser = None
    io = None
    import tempfile
    tmp_fixed_cif = None
    try:
        # Step 1: Patch cif file if necessary to ensure atom_site.occupancy is present
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_fixed_cif = os.path.join(tmpdir, "patched.cif")
            patched_cif = patch_occupancy_in_cif(cif_file, tmp_fixed_cif)
            from Bio.PDB import MMCIFParser, PDBIO
            parser = MMCIFParser(QUIET=True)
            io = PDBIO()
            structure_id = os.path.splitext(os.path.basename(cif_file))[0]
            structure = parser.get_structure(structure_id, patched_cif)
            io.set_structure(structure)
            io.save(pdb_out_path)
            return pdb_out_path
    except Exception as e:
        print(f"  [ERROR] Converting {cif_file} to PDB: {e}")
        return None

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

# === CONFIGURATION ===
cfg = get_default_config()
cfg.testing.ddg_scanning = False
WEIGHTS_DIR = "/home/jupyter-yehlin/ESM3_SaProt_dG/saprotdg_weights"
WEIGHT_FILES = [
    "SaProtdG_weights_1_lora.ckpt",
    "SaProtdG_weights_2_lora.ckpt", 
    "SaProtdG_weights_3_lora.ckpt"
]

CIF_DIR = "/home/jupyter-yehlin/ESM3_SaProt_dG/Adaptyv/cif"
CSV_FILE = "/home/jupyter-yehlin/ESM3_SaProt_dG/Adaptyv/proteinbase_all_data_28_10_2025_with_cif.csv"
CHAIN_ID = "A"

PDB_TMP_DIR = "/home/jupyter-yehlin/ESM3_SaProt_dG/Adaptyv/tmp_pdb"
os.makedirs(PDB_TMP_DIR, exist_ok=True)

# === LOAD CIF FILENAMES FROM CSV ===
df = pd.read_csv(CSV_FILE)
# Only select rows where 'designMethod' == 'boltzgen' and 'cif' column is not null (just in case)
boltzgen_cifs = df.loc[df['designMethod'] == 'boltzgen', 'cif'].dropna().unique()

# === LOAD ALL MODELS ===
models = []
for weight_file in WEIGHT_FILES:
    weight_path = os.path.join(WEIGHTS_DIR, weight_file)
    model = SaProtABS(weight_path, cfg)
    models.append(model)

# === SET UP RESULTS STORAGE ===
results_df = pd.DataFrame(
    columns=['CIF_File', 'PDB_File', 'Model1', 'Model2', 'Model3', 'Average', 'Std']
)

# === PREDICT FOR EACH CIF FILE ===
for cif_file in boltzgen_cifs:
    cif_path = os.path.join(CIF_DIR, cif_file)
    pdb_filename = os.path.splitext(cif_file)[0] + ".pdb"
    pdb_path = os.path.join(PDB_TMP_DIR, pdb_filename)

    # Convert cif to pdb if necessary and possible
    if not os.path.isfile(cif_path):
        print(f"[SKIP] CIF File not found: {cif_path}")
        continue

    # convert CIF to PDB if not already converted
    if not os.path.isfile(pdb_path):
        pdb_actual_path = cif_to_pdb(cif_path, pdb_path)
        if pdb_actual_path is None:
            print(f"[SKIP] Could not convert {cif_file} to PDB.")
            continue
    else:
        pdb_actual_path = pdb_path

    print(f"Processing CIF: {cif_file} → PDB: {os.path.basename(pdb_actual_path)}")

    predictions = []
    for i, model in enumerate(models, 1):
        try:
            pred_dg_per_res, pred_dg_avg, combined_seq = SaProtABS_predict(model, pdb_actual_path, CHAIN_ID)
            pred_value = pred_dg_avg[0]
        except Exception as e:
            print(f"  [ERROR] Model {i} failed on {os.path.basename(pdb_actual_path)}: {e}")
            pred_value = np.nan
        predictions.append(pred_value)
        try:
            pred_value_to_print = float(pred_value)
        except Exception:
            pred_value_to_print = float('nan')
        print(f"    🧬 Model {i}: Predicted ΔG: {pred_value_to_print:.2f} ({WEIGHT_FILES[i-1]})")

    # Compute averages, ignoring NaNs
    pred_clean = [p for p in predictions if not pd.isnull(p)]
    avg_pred = np.mean(pred_clean) if pred_clean else np.nan
    std_pred = np.std(pred_clean) if pred_clean else np.nan

    results_df.loc[len(results_df)] = [cif_file, os.path.basename(pdb_actual_path)] + predictions + [avg_pred, std_pred]

# === SAVE RESULTS AS CSV ===
out_csv = '/home/jupyter-yehlin/ESM3_SaProt_dG/model_predictions_boltzgen.csv'
results_df.to_csv(out_csv, index=False)
print(f"\nResults saved to {out_csv}\n")