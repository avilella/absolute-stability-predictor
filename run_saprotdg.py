#!/usr/bin/env python3
import argparse
import sys
import os
import csv

# Import SaProtdG as provided
from SaProtdG import SaProtdG, SaProtdG_predict

def log_message(message, verbose_only=False, is_verbose=False):
    """Prints messages to sys.stderr based on verbosity."""
    if not verbose_only or (verbose_only and is_verbose):
        print(message, file=sys.stderr)

def extract_antibody_name(filepath):
    """Reads the first line of the PDB file to extract an antibody name."""
    try:
        with open(filepath, 'r') as f:
            first_line = f.readline().strip()
            # If the file jumps straight into coordinates, there is no name header
            if not first_line or first_line.startswith(("ATOM", "HETATM", "MODEL")):
                return ""
            
            # Extract based on common formats
            if ":" in first_line:
                # E.g., "REMARK Antibody Name: Trastuzumab" -> "Trastuzumab"
                return first_line.split(":", 1)[1].strip()
            elif first_line.startswith("TITLE"):
                return first_line.replace("TITLE", "", 1).strip()
            elif first_line.startswith("HEADER"):
                # PDB standard HEADER usually has the classification/name starting at column 11
                return first_line[10:].strip()
            else:
                # Fallback: return the whole line if it doesn't match standard prefixes
                return first_line
    except Exception:
        return ""

def main():
    parser = argparse.ArgumentParser(description="Calculate protein stability using SaProtdG.")
    parser.add_argument("-i", "--inputfile", required=True, help="Input PDB file")
    parser.add_argument("-c", "--chains", default="A", help="Chain ID(s) to process, separated by ':' (e.g., A or H:L). Default: A")
    parser.add_argument("--tag", default="sapr", help="Tag to append to the output file (default: sapr)")
    parser.add_argument("--outdir", default=None, help="Output directory (default: same as input file directory)")
    parser.add_argument("--refresh", action="store_true", help="Recalculate even if a non-empty output file exists")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print detailed processing steps to STDERR")
    
    args = parser.parse_args()

    # Determine paths and filenames
    input_path = os.path.abspath(args.inputfile)
    input_dir = os.path.dirname(input_path)
    base_name_full = os.path.basename(input_path)
    base_name = os.path.splitext(base_name_full)[0]  # Removes the .pdb or .cif extension

    if args.outdir:
        out_dir = os.path.abspath(args.outdir)
        # Create output directory if it doesn't exist
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = input_dir

    out_filename = f"{base_name}.{args.tag}.csv"
    out_filepath = os.path.join(out_dir, out_filename)

    # Check if we should skip calculation
    if not args.refresh:
        if os.path.exists(out_filepath) and os.path.getsize(out_filepath) > 0:
            log_message(f"Found existing non-empty output file. Skipping calculation.", verbose_only=True, is_verbose=args.verbose)
            print(out_filepath)
            sys.exit(0)

    log_message(f"Starting stability calculation for: {base_name_full}", verbose_only=False, is_verbose=args.verbose)

    # Extract antibody name from the first line
    antibody_name = extract_antibody_name(input_path)
    if antibody_name:
        log_message(f"Found antibody name: {antibody_name}", verbose_only=True, is_verbose=args.verbose)

    # SaProtdG Weights configuration
    WEIGHTS = [
        os.path.expanduser("~/absolute-stability-predictor/saprotdg_weights/SaProtdG_weights_augmented_1_lora.ckpt"),
        os.path.expanduser("~/absolute-stability-predictor/saprotdg_weights/SaProtdG_weights_augmented_2_lora.ckpt"),
        os.path.expanduser("~/absolute-stability-predictor/saprotdg_weights/SaProtdG_weights_augmented_3_lora.ckpt"),
    ]

    # Load Models
    log_message("Loading SaProtdG ensemble models...", verbose_only=True, is_verbose=args.verbose)
    models = [SaProtdG(w) for w in WEIGHTS]

    # Predict for each chain
    log_message("Running predictions on the input structure...", verbose_only=True, is_verbose=args.verbose)
    
    target_chains = args.chains.split(":")
    results = []

    try:
        for chain in target_chains:
            log_message(f"Predicting for chain: {chain}", verbose_only=True, is_verbose=args.verbose)
            preds = [SaProtdG_predict(m, input_path, chain)[1][0] for m in models]
            ensemble_avg = sum(preds) / len(preds)
            results.append((chain, ensemble_avg))
            log_message(f"Chain {chain} Ensemble ΔG: {ensemble_avg:.2f} kcal/mol", verbose_only=False, is_verbose=args.verbose)
    except Exception as e:
        log_message(f"Error during prediction: {e}", verbose_only=False, is_verbose=args.verbose)
        sys.exit(1)

    # Write output to CSV
    log_message(f"Writing results to: {out_filename}", verbose_only=True, is_verbose=args.verbose)
    try:
        with open(out_filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['filename', 'basename', 'name', 'chain', 'saprotdg_stability_kcal_mol'])
            for chain, ensemble_avg in results:
                writer.writerow([input_path, base_name, antibody_name, chain, f"{ensemble_avg:.2f}"])
    except Exception as e:
        log_message(f"Error writing to output file: {e}", verbose_only=False, is_verbose=args.verbose)
        sys.exit(1)

    log_message("Processing complete.", verbose_only=True, is_verbose=args.verbose)

    # STDOUT requirement: Print ONLY the full path to the output file
    print(out_filepath)

if __name__ == "__main__":
    main()
