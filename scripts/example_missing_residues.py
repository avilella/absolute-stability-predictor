#!/usr/bin/env python3
"""
Example script demonstrating different ways to handle missing residues in SaProtdG.

This script shows how to:
1. Analyze missing residues in a PDB file
2. Choose the appropriate handling strategy
3. Run predictions with different missing residue handling methods
"""

import os
from utils.foldseek_util import print_missing_residue_analysis, analyze_missing_residues
from SaProtABS import SaProtABS, SaProtABS_predict

def main():
    # Example PDB file path (replace with your actual file)
    pdb_file = "1lze.pdb"  # Using the example file from the codebase
    
    if not os.path.exists(pdb_file):
        print(f"PDB file {pdb_file} not found. Please provide a valid PDB file path.")
        return
    
    print("=== SaProtdG Missing Residue Handling Example ===\n")
    
    # Step 1: Analyze missing residues in the PDB file
    print("Step 1: Analyzing missing residues...")
    print_missing_residue_analysis(pdb_file, chain_id='A')
    
    # Step 2: Get recommended strategy
    analysis = analyze_missing_residues(pdb_file, chain_id='A')
    recommended_strategy = analysis['recommended_strategy']
    
    print(f"\nStep 2: Using recommended strategy: {recommended_strategy}")
    
    # Step 3: Load the model (you'll need to provide the correct path)
    model_path = "/home/jupyter-yehlin/ESM3_SaProt_dG/SaProtdG_weights_1_additional_layers.ckpt"
    
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Please update the path.")
        return
    
    print(f"\nStep 3: Loading SaProtdG model from {model_path}")
    model = SaProtABS(model_path)
    
    # Step 4: Run predictions with different strategies
    strategies = ["mask", "skip", "interpolate"]
    
    print(f"\nStep 4: Running predictions with different strategies...")
    
    for strategy in strategies:
        print(f"\n--- Testing strategy: {strategy} ---")
        try:
            # Run prediction with current strategy
            pred_dg, pred_dg_avg, combined_seq = SaProtABS_predict(
                model=model,
                pdb_path=pdb_file,
                chain_id='A',
                foldseek_path="bin/foldseek",
                handle_missing_residues=strategy
            )
            
            if pred_dg_avg is not None:
                print(f"Predicted ΔG: {pred_dg_avg[0]:.3f}")
                print(f"Sequence length: {len(combined_seq)}")
                print(f"Strategy {strategy}: SUCCESS")
            else:
                print(f"Strategy {strategy}: FAILED (no prediction returned)")
                
        except Exception as e:
            print(f"Strategy {strategy}: ERROR - {str(e)}")
    
    # Step 5: Compare results
    print(f"\n=== Summary ===")
    print(f"Recommended strategy: {recommended_strategy}")
    print(f"Missing residues: {analysis['missing_residues']}/{analysis['total_residues']} ({analysis['missing_percentage']:.1f}%)")
    
    if analysis['missing_percentage'] > 0:
        print(f"\nRecommendations:")
        if analysis['missing_percentage'] < 5:
            print("- Use 'mask' strategy for best results")
        elif analysis['missing_percentage'] < 15:
            print("- Try 'interpolate' strategy to preserve structure")
        else:
            print("- Consider 'skip' strategy or find a more complete structure")
    else:
        print("- No missing residues detected, any strategy should work")

def compare_strategies(pdb_file, model_path):
    """
    Compare different missing residue handling strategies on the same structure.
    
    Args:
        pdb_file: Path to PDB file
        model_path: Path to SaProtdG model
    """
    print(f"\n=== Detailed Strategy Comparison ===")
    
    # Load model
    model = SaProtABS(model_path)
    
    strategies = ["mask", "skip", "interpolate"]
    results = {}
    
    for strategy in strategies:
        try:
            pred_dg, pred_dg_avg, combined_seq = SaProtABS_predict(
                model=model,
                pdb_path=pdb_file,
                chain_id='A',
                foldseek_path="bin/foldseek",
                handle_missing_residues=strategy
            )
            
            if pred_dg_avg is not None:
                results[strategy] = {
                    'dg': pred_dg_avg[0],
                    'seq_length': len(combined_seq),
                    'success': True
                }
            else:
                results[strategy] = {'success': False, 'error': 'No prediction returned'}
                
        except Exception as e:
            results[strategy] = {'success': False, 'error': str(e)}
    
    # Print comparison table
    print(f"{'Strategy':<12} {'ΔG':<10} {'Length':<8} {'Status':<10}")
    print("-" * 40)
    
    for strategy in strategies:
        if strategy in results and results[strategy]['success']:
            dg = results[strategy]['dg']
            length = results[strategy]['seq_length']
            print(f"{strategy:<12} {dg:<10.3f} {length:<8} {'SUCCESS':<10}")
        else:
            error = results[strategy].get('error', 'Unknown error')
            print(f"{strategy:<12} {'N/A':<10} {'N/A':<8} {'FAILED':<10}")

if __name__ == "__main__":
    main()
    
    # Uncomment the line below to run detailed comparison
    # compare_strategies("1lze.pdb", "/path/to/your/model.ckpt") 