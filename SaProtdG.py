import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForMaskedLM
from peft import get_peft_model, LoraConfig, TaskType
from model_utils import Stability_classification_head, SigmoidScaling

def SaProtdG(additional_layers_path, cfg=None):
    """
    Create a TransferModel using base SaProt from transformers and merge with additional layers
    (stability head + output scaling + LoRA weights)

    Args:
        additional_layers_path: Path to additional layers checkpoint (stability head + output scaling + LoRA)
        cfg: Configuration object
    """
    
    # Create simple config if none provided
    if cfg is None:
        class SimpleConfig:
            def __init__(self):
                self.training = type('obj', (object,), {
                    'rank': 4,
                    'dropout': 0.15
                })()
                self.model = type('obj', (object,), {
                    'freeze_weights': True
                })()
                self.testing = type('obj', (object,), {
                    'ddg_scanning': False
                })()
        
        cfg = SimpleConfig()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load base SaProt model directly
    print("Loading base SaProt model from transformers...")
    tokenizer = AutoTokenizer.from_pretrained("westlake-repl/SaProt_650M_AF2")
    base_model = AutoModelForMaskedLM.from_pretrained("westlake-repl/SaProt_650M_AF2").to(device)
    
    # Apply LoRA to base model
    print("Applying LoRA to base model...")
    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        target_modules=["query", "key", "value", "intermediate.dense", "output.dense"],
        r=cfg.training.rank,
        lora_dropout=cfg.training.dropout
    )
    
    base_model = get_peft_model(base_model, config)
    additional_checkpoint = torch.load(additional_layers_path, map_location=device)
    
    stability_head = Stability_classification_head(input_dim=1280, output_dim=1).to(device)
    output_scaling = SigmoidScaling(min_val=-1, max_val=5).to(device)
    
    # Create final model that combines everything
    class TransferModel(nn.Module):
        def __init__(self, base_model, tokenizer, stability_head, output_scaling, ddg_scanning=False):
            super().__init__()
            self.base_model = base_model
            self.tokenizer = tokenizer
            self.stability_head = stability_head
            self.output_scaling = output_scaling
            self.ddg_scanning = ddg_scanning
            self.device = device
            self.ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
            self.scan_batch_size = 1

        def forward(self, S):
            # Tokenize input
            inputs = self.tokenizer(S, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Create attention mask
            input_ids = inputs["input_ids"]
            mask = torch.where((input_ids == 0) | (input_ids == 1) | (input_ids == 2), 
                             torch.tensor(0, device=self.device), 
                             torch.tensor(1, device=self.device))
            
            # Get embeddings from base model
            outputs = self.base_model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            
            if self.ddg_scanning:
                length = int(len(S[0])/2)
                ddg_scan = torch.zeros(length, 20, len(S), length).to(self.device)
                scaled_ddg_scan = torch.zeros(length, 20, len(S), length).to(self.device)
                
                # Get wildtype stability
                stability_output = self.stability_head(hidden_states)
                dg_wt = stability_output.squeeze(-1)
                scaled_dg_wt = self.output_scaling(dg_wt)
                
                all_muts = [(i, j, A) for i in range(length) for j, A in enumerate(list(self.ALPHABET))]

                for start in range(0, len(all_muts), self.scan_batch_size):
                    chunk = all_muts[start:start + self.scan_batch_size]
                    mut_seqs = []
                    for (i, j, A) in chunk:
                        MUT = list(S[0])
                        MUT[i*2] = A
                        mut_seqs.append("".join(MUT))

                    mut_inputs = self.tokenizer(mut_seqs, return_tensors="pt", padding=True, truncation=True)
                    mut_inputs = {k: v.to(self.device) for k, v in mut_inputs.items()}
                    mut_outputs = self.base_model(**mut_inputs, output_hidden_states=True)
                    mut_hidden = mut_outputs.hidden_states[-1]
                    mut_stability = self.stability_head(mut_hidden)
                    dg_mut = mut_stability.squeeze(-1)
                    scaled_dg_mut = self.output_scaling(dg_mut)

                    for k, (i, j, _) in enumerate(chunk):
                        ddg_scan[i][j] = (dg_mut[k:k+1] - dg_wt)[:, 1:-1]
                        scaled_ddg_scan[i][j] = (scaled_dg_mut[k:k+1] - scaled_dg_wt)[:, 1:-1]

                return ddg_scan, scaled_ddg_scan
            else:
                # Pass through stability head
                stability_output = self.stability_head(hidden_states)
                dg = stability_output.squeeze(-1)
                scaled_dg = self.output_scaling(dg)
                return dg, scaled_dg, mask
   
    
    additional_state_dict = {}
    for key, value in additional_checkpoint['state_dict'].items():
        new_key = key.replace('model.', '')
        additional_state_dict[new_key] = value
    
    # Load LoRA weights first
    lora_weights = {}
    for key, value in additional_state_dict.items():
        if any(lora_prefix in key for lora_prefix in ['lora_A', 'lora_B']):
            new_key = key.replace('saprot_stability_base_esm.', 'base_model.model.esm.')
            lora_weights[new_key] = value
    
    if lora_weights:
        missing, unexpected = base_model.load_state_dict(lora_weights, strict=False)
        assert len(unexpected) == 0, "Unexpected keys in base model" + str(unexpected)


    # Load stability head weights
    stability_weights = {}
    for key, value in additional_state_dict.items():
        if 'stability_head' in key:
            new_key = key.replace('saprot_stability_stability_head.', '')
            stability_weights[new_key] = value
    
    if stability_weights:
        missing, unexpected = stability_head.load_state_dict(stability_weights, strict=False)
        assert len(missing) == 0, "Missing keys in stability head" + str(missing)
        assert len(unexpected) == 0, "Unexpected keys in stability head" + str(unexpected)

    # Load output scaling weights
    scaling_weights = {}
    for key, value in additional_state_dict.items():
        if key.startswith('output_scaling.'):
            scaling_weights[key.replace('output_scaling.', '')] = value
    
    if scaling_weights:
        missing, unexpected = output_scaling.load_state_dict(scaling_weights, strict=False)
        assert len(missing) == 0, "Missing keys in output scaling" + str(missing)
        assert len(unexpected) == 0, "Unexpected keys in output scaling" + str(unexpected)

    # Freeze base model if needed
    if cfg.model.freeze_weights:
        base_model.eval()
        for param in base_model.parameters():
            param.requires_grad = False
    

    final_model = TransferModel(
        base_model,
        tokenizer,
        stability_head,
        output_scaling,
        ddg_scanning=cfg.testing.ddg_scanning
    )

    return final_model

def SaProtdG_predict(model, pdb_path, chain_id='A', ddg_scanning=False, cdna_rescale=False, foldseek_path="bin/foldseek", given_seq=None, scan_batch_size=1):
    from utils.foldseek_util import get_struc_seq
    import os
    
    def get_saprot_input_info(pdb_file, chain_id='A'):
        process_id = os.getpid()
        struct_seq_data = get_struc_seq(foldseek_path, pdb_file, [chain_id], process_id)
        
        if chain_id not in struct_seq_data:
            print(f"Chain {chain_id} not found in structure sequence data")
            return None
        
        parsed_seqs = struct_seq_data[chain_id]
        _, _, combined_seq = parsed_seqs
        return combined_seq
    
    if given_seq is not None:
        combined_seq = given_seq
    else:
        combined_seq = get_saprot_input_info(pdb_path, chain_id)
    if combined_seq is None:
        return None, None, None
    
    model.eval()
    with torch.no_grad():
        if ddg_scanning:
            print("mutational scanning")
            original_flag = model.ddg_scanning
            model.ddg_scanning = True
            model.scan_batch_size = scan_batch_size
            try:
                ddg_scan, scaled_ddg_scan = model([combined_seq])
            finally:
                model.ddg_scanning = original_flag
            return ddg_scan, scaled_ddg_scan, combined_seq
        else:
            if cdna_rescale:
                _, pred_scaled_dg, mask = model([combined_seq])
                pred_scaled_dg = pred_scaled_dg * mask
                pred_scaled_dg_avg = (pred_scaled_dg.sum(dim=-1) / mask.sum(dim=-1)).tolist()
                return pred_scaled_dg, pred_scaled_dg_avg, combined_seq
            else:
                pred_dg, _, mask = model([combined_seq])
                pred_dg = pred_dg * mask
                pred_dg_avg = (pred_dg.sum(dim=-1) / mask.sum(dim=-1)).tolist()
                return pred_dg, pred_dg_avg, combined_seq