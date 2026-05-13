import torch
import os
from pathlib import Path

def extract_esm3dG_layers_from_checkpoint(checkpoint_path, output_path=None):
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

    lora_count = 0
    stability_head_count = 0
    output_scaling_count = 0
    base_esm3_count = 0

    for key, value in checkpoint['state_dict'].items():
        if 'model.esm3_stability_model.' in key and ('lora_A' in key or 'lora_B' in key):
            additional_state_dict[key] = value
            lora_count += 1
        elif 'model.esm3_stability_model.stability_head.' in key:
            additional_state_dict[key] = value
            stability_head_count += 1
        elif 'model.output_scaling.' in key:
            additional_state_dict[key] = value
            output_scaling_count += 1
        elif key.startswith('model.esm3.') and 'lora_A' not in key and 'lora_B' not in key:
            # Fine-tuned base ESM3 weights (unfrozen during "warm-up" training)
            additional_state_dict[key] = value
            base_esm3_count += 1

    print(f"\nSummary:")
    print(f"  LoRA keys: {lora_count}")
    print(f"  Stability head keys: {stability_head_count}")
    print(f"  Output scaling keys: {output_scaling_count}")
    print(f"  Base ESM3 keys: {base_esm3_count}")
    print(f"  Total: {lora_count + stability_head_count + output_scaling_count + base_esm3_count}")

    
    if not additional_state_dict:
        print("Warning: No additional layers found in checkpoint!")
        print("Available keys in checkpoint:")
        # for key in checkpoint['state_dict'].keys():
        #     print(f"  - {key}")
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
    # for key in additional_state_dict.keys():
    #     print(f"  - {key}")
    
    return output_path

# Example usage for your specific checkpoint
if __name__ == "__main__":

    # checkpoint_path='/home/jupyter-yehlin/esm3_attention/weights/checkpoint_unfreeze_warum_up_ddg_bins_lora_dg_and_ddg_sigmoid_5e5_model2_resume_model1_norm/Fine_Tune_ESM3_dmsv4_AF_unfreeze_warm_up_ddG_revised_esm3_epoch=07_val_ddG_spearman=0.58_val_ddG_mse=0.2_val_dG_spearman=0.87_val_dG_mse=0.68.ckpt'
    # checkpoint_path = '/home/jupyter-yehlin/esm3_attention/weights/checkpoint_unfreeze_warum_up_ddg_bins_lora_dg_and_ddg_sigmoid_5e5_model1_resume_model1_norm/Fine_Tune_ESM3_dmsv4_AF_unfreeze_warm_up_ddG_revised_esm3_epoch=03_val_ddG_spearman=0.58_val_ddG_mse=0.2_val_dG_spearman=0.86_val_dG_mse=0.69.ckpt'
    # checkpoint_path='/home/jupyter-yehlin/esm3_attention/weights/checkpoint_unfreeze_warum_up_ddg_bins_lora_dg_and_ddg_sigmoid_5e5_model3_resume_model2_norm/Fine_Tune_ESM3_dmsv4_AF_unfreeze_warm_up_ddG_revised_esm3_epoch=01_val_ddG_spearman=0.57_val_ddG_mse=0.2_val_dG_spearman=0.86_val_dG_mse=0.72.ckpt'

    # checkpoint_path='/home/jupyter-yehlin/esm3_attention/weights/checkpoint_unfreeze_warum_up_ddg_bins_lora_dg_and_ddg_filtered_augmented_sigmoid_5e5_model1_resume_augment_model1_norm/Fine_Tune_ESM3_dmsv4_AF_unfreeze_warm_up_ddG_revised_esm3_epoch=06_val_ddG_spearman=0.69_val_ddG_mse=0.54_val_dG_spearman=0.76_val_dG_mse=0.65.ckpt'
    # checkpoint_path = '/home/jupyter-yehlin/esm3_attention/weights/checkpoint_unfreeze_warum_up_ddg_bins_lora_dg_and_ddg_filtered_augmented_sigmoid_5e5_model2_resume_augment_model1_norm/Fine_Tune_ESM3_dmsv4_AF_unfreeze_warm_up_ddG_revised_esm3_epoch=10_val_ddG_spearman=0.7_val_ddG_mse=0.49_val_dG_spearman=0.75_val_dG_mse=0.64.ckpt'
    # checkpoint_path='/home/jupyter-yehlin/esm3_attention/weights/checkpoint_unfreeze_warum_up_ddg_bins_lora_dg_and_ddg_filtered_augmented_sigmoid_5e5_model3_resume_augment_model1_norm/Fine_Tune_ESM3_dmsv4_AF_unfreeze_warm_up_ddG_revised_esm3_epoch=05_val_ddG_spearman=0.69_val_ddG_mse=0.51_val_dG_spearman=0.75_val_dG_mse=0.66.ckpt'

    # non_aug_jobs — non-augmented models (Config A)
    non_aug_jobs = [
        (
            "/home/jupyter-yehlin/esm3_attention/weights/checkpoint_unfreeze_warum_up_ddg_bins_lora_dg_and_ddg_sigmoid_5e5_model1_resume_model1_norm/Fine_Tune_ESM3_dmsv4_AF_unfreeze_warm_up_ddG_revised_esm3_epoch=03_val_ddG_spearman=0.58_val_ddG_mse=0.2_val_dG_spearman=0.86_val_dG_mse=0.69.ckpt",
            "/home/jupyter-yehlin/ESM3_SaProt_dG/esm3dg_weights/ESM3dG_weights_1_lora.ckpt",
        ),
        (
            "/home/jupyter-yehlin/esm3_attention/weights/checkpoint_unfreeze_warum_up_ddg_bins_lora_dg_and_ddg_sigmoid_5e5_model2_resume_model1_norm/Fine_Tune_ESM3_dmsv4_AF_unfreeze_warm_up_ddG_revised_esm3_epoch=07_val_ddG_spearman=0.58_val_ddG_mse=0.2_val_dG_spearman=0.87_val_dG_mse=0.68.ckpt",
            "/home/jupyter-yehlin/ESM3_SaProt_dG/esm3dg_weights/ESM3dG_weights_2_lora.ckpt",
        ),
        (
            "/home/jupyter-yehlin/esm3_attention/weights/checkpoint_unfreeze_warum_up_ddg_bins_lora_dg_and_ddg_sigmoid_5e5_model3_resume_model2_norm/Fine_Tune_ESM3_dmsv4_AF_unfreeze_warm_up_ddG_revised_esm3_epoch=01_val_ddG_spearman=0.57_val_ddG_mse=0.2_val_dG_spearman=0.86_val_dG_mse=0.72.ckpt",
            "/home/jupyter-yehlin/ESM3_SaProt_dG/esm3dg_weights/ESM3dG_weights_3_lora.ckpt",
        ),
    ]

    # augmented_1/2/3 — norm augmented models (Config B)
    aug_jobs = [
        (
            "/home/jupyter-yehlin/esm3_attention/weights/checkpoint_unfreeze_warum_up_ddg_bins_lora_dg_and_ddg_filtered_augmented_sigmoid_5e5_model1_resume_augment_model1_norm/Fine_Tune_ESM3_dmsv4_AF_unfreeze_warm_up_ddG_revised_esm3_epoch=06_val_ddG_spearman=0.69_val_ddG_mse=0.54_val_dG_spearman=0.76_val_dG_mse=0.65.ckpt",
            "/home/jupyter-yehlin/ESM3_SaProt_dG/esm3dg_weights/ESM3dG_weights_augmented_1_lora.ckpt",
        ),
        (
            "/home/jupyter-yehlin/esm3_attention/weights/checkpoint_unfreeze_warum_up_ddg_bins_lora_dg_and_ddg_filtered_augmented_sigmoid_5e5_model2_resume_augment_model1_norm/Fine_Tune_ESM3_dmsv4_AF_unfreeze_warm_up_ddG_revised_esm3_epoch=10_val_ddG_spearman=0.7_val_ddG_mse=0.49_val_dG_spearman=0.75_val_dG_mse=0.64.ckpt",
            "/home/jupyter-yehlin/ESM3_SaProt_dG/esm3dg_weights/ESM3dG_weights_augmented_2_lora.ckpt",
        ),
        (
            "/home/jupyter-yehlin/esm3_attention/weights/checkpoint_unfreeze_warum_up_ddg_bins_lora_dg_and_ddg_filtered_augmented_sigmoid_5e5_model3_resume_augment_model1_norm/Fine_Tune_ESM3_dmsv4_AF_unfreeze_warm_up_ddG_revised_esm3_epoch=05_val_ddG_spearman=0.69_val_ddG_mse=0.51_val_dG_spearman=0.75_val_dG_mse=0.66.ckpt",
            "/home/jupyter-yehlin/ESM3_SaProt_dG/esm3dg_weights/ESM3dG_weights_augmented_3_lora.ckpt",
        ),
    ]

    for checkpoint_path, output_path in non_aug_jobs:
        print(f"\n=== Extracting {Path(output_path).name} ===")
        extract_esm3dG_layers_from_checkpoint(checkpoint_path, output_path)

    for checkpoint_path, output_path in aug_jobs:
        print(f"\n=== Extracting {Path(output_path).name} ===")
        extract_esm3dG_layers_from_checkpoint(checkpoint_path, output_path)