import os
import torch
import torch.nn as nn
import numpy as np
from esm.models.esm3 import ESM3
from peft import get_peft_model, LoraConfig, TaskType
import gemmi
from esm.sdk.api import ESMProtein
from huggingface_hub import login
from model_utils import ESM3_Stability_head, SigmoidScaling

# Patch ESM3's tokenize_sequence to handle None mask_token (transformers 4.40+ compat).
# Newer PreTrainedTokenizerFast may return None for mask_token; str.replace() then crashes
# even when the sequence contains no mask characters.
import esm.utils.encoding as _esm_enc
import esm.utils.constants.esm3 as _ESM_C
_orig_tokenize_seq = _esm_enc.tokenize_sequence

def _patched_tokenize_seq(sequence, sequence_tokenizer, add_special_tokens=True):
    mask_tok = sequence_tokenizer.mask_token or "<mask>"
    sequence = sequence.replace(_ESM_C.MASK_STR_SHORT, mask_tok)
    tokens = sequence_tokenizer.encode(sequence, add_special_tokens=add_special_tokens)
    return torch.tensor(tokens, dtype=torch.int64)

_esm_enc.tokenize_sequence = _patched_tokenize_seq


# -----------------------------------------------------------------------------
# Constants & Helpers
# -----------------------------------------------------------------------------

SEQUENCE_VOCAB = [
    "<cls>", "<pad>", "<eos>", "<unk>",
    "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K",
    "Q", "N", "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z",
    "O", ".", "-", "|",
    "<mask>",
]

atom_types = [
    'N', 'CA', 'C', 'CB', 'O', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD',
    'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3',
    'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2',
    'CZ3', 'NZ', 'OXT'
]

atom_order = {atom_type: i for i, atom_type in enumerate(atom_types)}
three_to_one = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G',
    'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V',
    'TRP': 'W', 'TYR': 'Y', 'SEC': 'U', 'PYL': 'O', 'ASX': 'B', 'GLX': 'Z',
    'XLE': 'J', 'XAA': 'X'
}


ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'

def tied_featurize(batch, device=torch.device('cpu')):
    """ Pack and pad batch into torch tensors """
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    B = len(batch)
    lengths = torch.tensor([len(b['seq']) for b in batch], dtype=torch.bfloat16, device=device)  # sum of chain seq lengths
    L_max = max([len(b['seq']) for b in batch])
    sequence_tokens = torch.ones([B, L_max], dtype=torch.int32, device=device) ## sequence pad token is 1
    structure_tokens = torch.full((B, L_max), 4099, dtype=torch.int32, device=device) ## structure pad token is 4099
    mask  = torch.zeros([B, L_max], dtype=torch.int32, device=device)
    coordinates_tokens = torch.full((B, L_max, 37, 3), np.nan, dtype=torch.float32, device=device)

    for i, b in enumerate(batch):
        sequence_tokens[i, :len(b['seq'])] = b['seq'].clone().detach().to(dtype=torch.int32, device=device)
        structure_tokens[i, :len(b['struct'])] = b['struct'].clone().detach().to(dtype=torch.int32, device=device)
        coordinates_tokens[i, :len(b['coord'])] = b['coord'].clone().detach().to(dtype=torch.float32, device=device)
        mask[i, 1:len(b['seq'])-1] = 1
        
    return sequence_tokens, structure_tokens, coordinates_tokens, mask


# -----------------------------------------------------------------------------
# Model Components
# -----------------------------------------------------------------------------

class ModelWrapper(nn.Module):
    def __init__(self, base_model, stability_head, device):
        super(ModelWrapper, self).__init__()
        self.base_model = base_model
        self.stability_head = stability_head.to(device)

    def forward(self, sequence_tokens, structure_tokens, structure_coords):
        # ESM3 Forward pass
        base_outputs = self.base_model(
            sequence_tokens=sequence_tokens, 
            structure_tokens=structure_tokens, 
            structure_coords=structure_coords
        )
        
        features = base_outputs.embeddings  
        stability_output = self.stability_head(features)
        return stability_output

# -----------------------------------------------------------------------------
# Main Factory & Wrapper
# -----------------------------------------------------------------------------

class TransferModel(nn.Module):
    def __init__(self, esm3_stability_model, output_scaling, ddg_scanning=False, device='cuda'):
        super().__init__()
        self.esm3_stability_model = esm3_stability_model
        self.output_scaling = output_scaling
        self.ddg_scanning = ddg_scanning
        self.device = device
        
    def forward(self, batch):
        if not self.ddg_scanning:
            sequence_tokens, structure_tokens, coordinates_tokens, mask = tied_featurize(batch, self.device)
            stability_output = self.esm3_stability_model(
                sequence_tokens=sequence_tokens, 
                structure_tokens=structure_tokens, 
                structure_coords=coordinates_tokens
            )
            dg = stability_output.squeeze(-1)
            scaled_dg = self.output_scaling(dg)
            return dg, scaled_dg, mask
            
        else:
            # Replicated logic from original file for DDG Scanning
            matched_indices = {}
            for char in ALPHABET:
                index = SEQUENCE_VOCAB.index(char)
                matched_indices[char] = index
    
            seq_tokens_original = batch[0]['seq'] # Assuming tensor input from batch logic
            if isinstance(seq_tokens_original, list): seq_tokens_original = torch.tensor(seq_tokens_original)
            
            length = len(seq_tokens_original)
            # print("length", length)
            
            # Create outputs
            dg_scan = torch.zeros(length-2, 20, len(batch), length-2).to(self.device)
            dg_scaled = torch.zeros(length-2, 20, len(batch), length-2).to(self.device)
            
            # WT Pass
            sequence_tokens, structure_tokens, coordinates_tokens, mask = tied_featurize(batch, self.device)
            stability_output = self.esm3_stability_model(
                sequence_tokens=sequence_tokens, 
                structure_tokens=structure_tokens, 
                structure_coords=coordinates_tokens
            )
            dg_wt = stability_output.squeeze(-1)
            dg_scaled_wt = self.output_scaling(dg_wt)
            
            # Mutational Scan Loop
            # Optimization: In a real scenario, this loop is slow. 
            # We clone the batch dict to avoid modifying the original constantly in a way that breaks things.
            base_seq = seq_tokens_original.clone().detach()
            
            for i in range(length-2): ## exclude start and end token 
                for j, A in enumerate(list(ALPHABET)):
                    MUT = base_seq.clone()
                    MUT[i+1] = matched_indices[A] ## first position is start token
                    
                    # Temporarily update batch
                    batch[0]['seq'] = MUT
                    
                    sequence_tokens, structure_tokens, coordinates_tokens, mask = tied_featurize(batch, self.device)
                    stability_output = self.esm3_stability_model(
                        sequence_tokens=sequence_tokens, 
                        structure_tokens=structure_tokens, 
                        structure_coords=coordinates_tokens
                    )
                    
                    dg_mut = stability_output.squeeze(-1)
                    dg_scaled_mut = self.output_scaling(dg_mut)
                    
                    dg_scan[i][j] = (dg_mut - dg_wt)[:, 1:-1]
                    dg_scaled[i][j] = (dg_scaled_mut - dg_scaled_wt)[:, 1:-1]
            
            # Restore original seq
            batch[0]['seq'] = base_seq
            return dg_scan, dg_scaled

def ESM3ABS(additional_layers_path, cfg=None):
    if cfg is None:
        class SimpleConfig:
            def __init__(self):
                self.training = type('obj', (object,), {'rank': 4, 'dropout': 0.15})()
                self.model = type('obj', (object,), {'freeze_weights': True})()
                self.testing = type('obj', (object,), {'ddg_scanning': False})()
        cfg = SimpleConfig()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token, add_to_git_credential=False)


    esm3 = ESM3.from_pretrained("esm3_sm_open_v1").to(device).to(torch.float32)


    rank = cfg.training.rank
    dropout = cfg.training.dropout

    
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        target_modules=["layernorm_qkv.1", "out_proj"],
        r=rank,
        lora_dropout=dropout
    )
    
    _ = get_peft_model(esm3, peft_config)

    stability_head = ESM3_Stability_head(input_dim=1536, output_dim=1).to(device)
    output_scaling = SigmoidScaling(min_val=-1, max_val=5).to(device)
    checkpoint = torch.load(additional_layers_path, map_location=device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    lora_keys = {}
    for k, v in state_dict.items():
        if 'lora' in k:
            clean_k = k
            clean_k = clean_k.replace('model.esm3_stability_model.base_model.', '')
            lora_keys[clean_k] = v
            
    if lora_keys:
        msg = esm3.load_state_dict(lora_keys, strict=False)

    

    head_keys = {}

    for k, v in state_dict.items():
        if 'stability_head' in k:
            clean_k = k
            if 'stability_head.' in k:
                clean_k = k.split('stability_head.')[1]
            head_keys[clean_k] = v
            
    if head_keys:
        msg = stability_head.load_state_dict(head_keys, strict=True)



    scale_keys = {}
    for k, v in state_dict.items():
        if 'output_scaling' in k:
            clean_k = k
            if 'output_scaling.' in k:
                clean_k = k.split('output_scaling.')[1]
            scale_keys[clean_k] = v

    if scale_keys:
        msg = output_scaling.load_state_dict(scale_keys, strict=True)



    esm3_stability_model = ModelWrapper(esm3, stability_head, device)
    
    final_model = TransferModel(
        esm3_stability_model, 
        output_scaling, 
        ddg_scanning=cfg.testing.ddg_scanning, 
        device=device
    )
    
    if cfg.model.freeze_weights:
        final_model.eval()
        for param in final_model.parameters():
            param.requires_grad = False
            
    return final_model



def parse_CIF(path_to_cif, input_chain_list=None, ca_only=False, side_chains=True):
    c = 0
    cif_dict_list = []
    init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                     'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                     'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    extra_alphabet = [str(item) for item in list(np.arange(300))]
    chain_alphabet = init_alphabet + extra_alphabet

    if input_chain_list:
        chain_alphabet = input_chain_list

    biounit_names = [path_to_cif]
    for biounit in biounit_names:
        my_dict = {}
        s = 0
        concat_seq = ''
        coords_dict = {}
        model = gemmi.read_structure(biounit)
        for chain in model[0]:
            letter = chain.name
            if letter not in chain_alphabet:
                continue

            valid_residues = [res for res in chain if res.name in three_to_one]
            chain_length = len(valid_residues)
            # print("Valid chain length:", chain_length)
            
            # print("chain_length", chain_length)
            chain_coords = np.zeros((chain_length, 37, 3), dtype=float)
            seq = []

            for i, res in enumerate(valid_residues):
                # Append the one-letter sequence for valid amino acids
                seq.append(three_to_one[res.name])
                
                for atom in res:
                    if atom.name in atom_order:
                        atom_idx = atom_order[atom.name]
                        if atom_idx < 37:
                            chain_coords[i, atom_idx, :] = atom.pos.tolist()

    return "".join(seq), chain_coords



def get_esm3_input_info_direct(pdb_path, chain_id, esm3_base_model):
    """
    Prepares the input dictionary for ESM3dG using the model's internal encoder.
    """
    # 1. Get raw data from file
    sequence, coords = parse_CIF(pdb_path, chain_id)

    
    if sequence is None or coords is None:
        return None, None

    # 2. Prepare ESMProtein object
    # Ensure coords are tensor
    structure_prompt = torch.tensor(coords, dtype=torch.float32)
    
    protein_prompt = ESMProtein(sequence=sequence, coordinates=structure_prompt)
    
    # 3. Encode using the base ESM3 model
    device = esm3_base_model.device
    esm3_base_model.eval()

    with torch.no_grad():
        encoder = esm3_base_model.encode(protein_prompt)
        
    info_dict = {}
    info_dict['seq'] = encoder.sequence.to(device)
    info_dict['struct'] = encoder.structure.to(device)
    info_dict['coord'] = encoder.coordinates.to(device)
    
    return info_dict, sequence

def ESM3ABS_predict(model, pdb_path, chain_id='A', ddg_scanning=False, sigmoid_on = False):
    """
    Predicts stability (dG) or performs scanning for a given PDB file.

    Args:
        model: Loaded ESM3dG TransferModel
        pdb_path: Path to the PDB/CIF file
        chain_id: Chain ID to parse
        ddg_scanning: Whether to perform mutational scanning
        
    Returns:
        (pred, pred_avg, sequence)
        OR
        (ddg_scan, scaled_ddg_scan, sequence) if scanning
    """
    
    base_esm3 = model.esm3_stability_model.base_model

    # 1. Get Input Info
    info_dict, sequence = get_esm3_input_info_direct(pdb_path, chain_id, base_esm3)

    if info_dict is None or sequence is None:
        print("Error getting input info")
        return None, None, None

    # 2. Run Prediction
    model.eval()
    model.ddg_scanning = ddg_scanning # Dynamically set mode
    
    with torch.no_grad():
        # Pass as a list (batch of 1)
        batch = [info_dict]
        
        if ddg_scanning:
            ddg_scan, scaled_ddg_scan = model(batch)
            return ddg_scan.cpu(), scaled_ddg_scan.cpu(), sequence
            
        else:
            pred_dg, pred_scaled_dg, mask = model(batch)

            # Apply mask and calculate average
            # Outputs are (B, L). Here B=1.

            pred_dg = pred_dg.cpu()
            pred_scaled_dg = pred_scaled_dg.cpu()
            mask = mask.cpu()
            
            pred_dg = pred_dg * mask
            pred_scaled_dg = pred_scaled_dg * mask
            
            
            # Sum and divide by valid length
            # Note: mask excludes cls/eos usually, ensure safe division
            valid_len = mask.sum(dim=-1)
            pred_dg_avg = (pred_dg.sum(dim=-1) / valid_len).tolist()
            pred_scaled_dg_avg = (pred_scaled_dg.sum(dim=-1) / valid_len).tolist()

            if sigmoid_on:
                return pred_scaled_dg, pred_scaled_dg_avg, sequence
            else:
                return pred_dg, pred_dg_avg, sequence

