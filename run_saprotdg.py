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

def main():
    parser = argparse.ArgumentParser(description="Calculate protein stability using SaProtdG.")
    parser.add_argument("-i", "--inputfile", required=True, help="Input PDB file")
    parser.add_argument("--tag", default="sapr", help="Tag to append to the output file (default: sapr)")
    parser.add_argument("--outdir", default=None, help="Output directory (default: same as input file directory)")
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

    log_message(f"Starting stability calculation for: {base_name_full}", verbose_only=False, is_verbose=args.verbose)

    # SaProtdG Weights configuration
    WEIGHTS = [
        "saprotdg_weights/SaProtdG_weights_augmented_1_lora.ckpt",
        "saprotdg_weights/SaProtdG_weights_augmented_2_lora.ckpt",
        "saprotdg_weights/SaProtdG_weights_augmented_3_lora.ckpt",
    ]

    # Load Models
    log_message("Loading SaProtdG ensemble models...", verbose_only=True, is_verbose=args.verbose)
    models = [SaProtdG(w) for w in WEIGHTS]

    # Predict
    log_message("Running predictions on the input structure...", verbose_only=True, is_verbose=args.verbose)
    try:
        # Note: "A" is retained as the chain identifier from the original provided code. 
        # If your PDB files use different chains, you may need to parse or parameterize this.
        preds = [SaProtdG_predict(m, input_path, "A")[1][0] for m in models]
        ensemble_avg = sum(preds) / len(preds)
    except Exception as e:
        log_message(f"Error during prediction: {e}", verbose_only=False, is_verbose=args.verbose)
        sys.exit(1)

    log_message(f"Ensemble ΔG: {ensemble_avg:.2f} kcal/mol", verbose_only=False, is_verbose=args.verbose)

    # Write output to CSV
    log_message(f"Writing results to: {out_filename}", verbose_only=True, is_verbose=args.verbose)
    try:
        with open(out_filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Full Filename', 'Basename', 'Stability (kcal/mol)'])
            writer.writerow([input_path, base_name, f"{ensemble_avg:.2f}"])
    except Exception as e:
        log_message(f"Error writing to output file: {e}", verbose_only=False, is_verbose=args.verbose)
        sys.exit(1)

    log_message("Processing complete.", verbose_only=True, is_verbose=args.verbose)

    # STDOUT requirement: Print ONLY the full path to the output file
    print(out_filepath)

if __name__ == "__main__":
    main()
