import torch
import os
from pathlib import Path

def extract_saprotdg_layers_from_checkpoint(checkpoint_path, output_path=None):
    """
    Extract only the additional layers (stability head + output scaling) from a merged checkpoint
    EXCLUDING LoRA weights
    
    Args:
        checkpoint_path: Path to the merged checkpoint file
        output_path: Path to save the additional layers (optional, will auto-generate if None)
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract additional layers state dict (excluding LoRA)
    additional_state_dict = {}

    for key, value in checkpoint['state_dict'].items():
        # Look for keys related to additional layers ONLY (not LoRA)
        if 'model.saprot_stability_model.' in key and ('lora_A' in key or 'lora_B' in key):
            additional_state_dict[key] = value
            print(f"Found additional layer key: {key}")
        elif 'model.saprot_stability_model.stability_head.' in key:
            additional_state_dict[key] = value
            print(f"Found additional layer key: {key}")
        elif 'model.output_scaling.' in key:
            additional_state_dict[key] = value
            print(f"Found additional layer key: {key}")

    
    if not additional_state_dict:
        print("Warning: No additional layers found in checkpoint!")
        print("Available keys in checkpoint:")
        for key in checkpoint['state_dict'].keys():
            print(f"  - {key}")
        return None
    
    # Generate output path if not provided
    if output_path is None:
        checkpoint_name = Path(checkpoint_path).stem
        output_path = f"{checkpoint_name}_lora.ckpt"
    
    # Save additional layers weights
    torch.save({
        'state_dict': additional_state_dict,
        'config': checkpoint.get('config', {}),
        'original_checkpoint': checkpoint_path
    }, output_path)
    
    print(f"\nAdditional layers extracted and saved to: {output_path}")
    print(f"Number of additional layer parameters: {len(additional_state_dict)}")
    print(f"Additional layer keys:")
    for key in additional_state_dict.keys():
        print(f"  - {key}")
    
    return output_path
# Example usage for your specific checkpoint
if __name__ == "__main__":
    # checkpoint_path = "/home/jupyter-yehlin/ESM3_SaProt_dG/saprotdg_weights/SaProtdG_weights_3.ckpt"
    
    checkpoint_paths = [
        "/home/jupyter-yehlin/SaProt_dG/weights/Saprot_lora_mgnify_dg_and_ddg_no_confidence_model3_1e4_terminus_trancate_augment_d_sigmoid_both/Saprot_lora_dmsv4_AF_epoch=01_val_ddG_spearman=0.67_val_ddG_mse=0.52_val_dG_spearman=0.72_val_dG_mse=0.8.ckpt",
        "/home/jupyter-yehlin/SaProt_dG/weights/Saprot_lora_mgnify_dg_and_ddg_no_confidence_model7_5e5_terminus_trancate_augment_d_sigmoid_both/Saprot_lora_dmsv4_AF_epoch=11_val_ddG_spearman=0.67_val_ddG_mse=0.52_val_dG_spearman=0.69_val_dG_mse=0.94.ckpt",
        "/home/jupyter-yehlin/SaProt_dG/weights/Saprot_lora_mgnify_dg_and_ddg_no_confidence_model6_5e5_terminus_trancate_augment_d_sigmoid_both/Saprot_lora_dmsv4_AF_epoch=07_val_ddG_spearman=0.67_val_ddG_mse=0.54_val_dG_spearman=0.69_val_dG_mse=0.96.ckpt",
        "/ssd/yehlin/SaProt_dG/weights/Saprot_lora_mgnify_ddg_no_confidence_sigmoid_both_model1_1e4/Saprot_lora_dmsv4_AF_epoch=04_val_ddG_spearman=0.57_val_ddG_mse=0.21__val_dG_spearman=0.84_val_dG_mse=0.86.ckpt",
        "/ssd/yehlin/SaProt_dG/weights/Saprot_lora_mgnify_ddg_no_confidence_sigmoid_both_model2_1e4/Saprot_lora_dmsv4_AF_epoch=02_val_ddG_spearman=0.56_val_ddG_mse=0.23_val_dG_spearman=0.83_val_dG_mse=0.99.ckpt",
        "/data/yehlin/SaprotABS/weights/Saprot_lora_mgnify_ddg_no_confidence_sigmoid_both_model9_1e4/Saprot_lora_dmsv4_AF_epoch=00_val_ddG_spearman=0.56_val_ddG_mse=0.2_val_dG_spearman=0.85_val_dG_mse=0.78.ckpt"
    ]

    output_paths = [
        "/home/jupyter-yehlin/ESM3_SaProt_dG/saprotdg_weights/SaProtdG_weights_augmented_1_lora.ckpt",
        "/home/jupyter-yehlin/ESM3_SaProt_dG/saprotdg_weights/SaProtdG_weights_augmented_2_lora.ckpt",
        "/home/jupyter-yehlin/ESM3_SaProt_dG/saprotdg_weights/SaProtdG_weights_augmented_3_lora.ckpt",
        "/home/jupyter-yehlin/ESM3_SaProt_dG/saprotdg_weights/SaProtdG_weights_1_lora.ckpt",
        "/home/jupyter-yehlin/ESM3_SaProt_dG/saprotdg_weights/SaProtdG_weights_2_lora.ckpt",
        "/home/jupyter-yehlin/ESM3_SaProt_dG/saprotdg_weights/SaProtdG_weights_3_lora.ckpt"
    ]

    for checkpoint_path, output_path in zip(checkpoint_paths, output_paths):
        print(f"\n{'='*80}")
        print(f"=== Extracting additional layers (excluding LoRA) ===")
        print(f"=== Processing: {Path(checkpoint_path).name} ===")
        print(f"{'='*80}")
        additional_layers_path = extract_saprotdg_layers_from_checkpoint(checkpoint_path, output_path)


