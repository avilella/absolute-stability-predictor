"""
After running `python run_paper_examples.py`, call this script to patch
the DEMO_DATA block in index.html with the real predicted ΔG values.

Usage:
    python update_demo_from_results.py
"""
import json, re, os

RESULTS_PATH = "/home/jupyter-yehlin/ESM3_SaProt_dG/paper_example_results.json"
HTML_PATH    = "/home/jupyter-yehlin/ESM3_SaProt_dG/index.html"

with open(RESULTS_PATH) as f:
    results = json.load(f)

# Metadata not in the results file
CHAIN = {
    "3hhr": "B", "1ft8": "A", "1ris": "A", "1div": "A",
    "1bni": "A", "2trx": "A", "1stn": "A", "1ten": "A",
}
DESC = {
    "3hhr": "Human Growth Hormone",
    "1ft8": "TAP/NXF1 RNA-binding domain",
    "1ris": "Ribosomal protein S6",
    "1div": "Ribosomal protein L9",
    "1bni": "Barnase",
    "2trx": "Thioredoxin",
    "1stn": "Staphylococcal nuclease",
    "1ten": "Tenascin III domain",
}
FIG = {
    "3hhr": "4e", "1ft8": "4e", "1ris": "4e", "1div": "4e",
    "1bni": "4f", "2trx": "4f", "1stn": "4f", "1ten": "4f",
}

# Build new DEMO_DATA block
lines = [
    "// ΔG values (kcal/mol) — ensemble mean from run_paper_examples.py",
    "// Run `python run_paper_examples.py` to regenerate with real model predictions",
    "const DEMO_DATA = {",
    "  // Figure 4e — main paper showcase",
]
for pdb in ["3hhr", "1ft8", "1ris", "1div"]:
    if pdb not in results:
        continue
    r = results[pdb]
    esm  = r["esm3"]["mean"]
    sap  = r["saprot"]["mean"]
    L    = r["length"]
    chain = CHAIN[pdb]
    desc  = DESC[pdb]
    lines.append(
        f"  '{pdb}': {{ esm3: {esm:.2f}, saprot: {sap:.2f}, len: {L:3d},"
        f" name: '{pdb}', chain: '{chain}', desc: '{desc}' }},"
    )

lines.append("  // Figure 4f — ThermoMut benchmark")
for pdb in ["1bni", "2trx", "1stn", "1ten"]:
    if pdb not in results:
        continue
    r = results[pdb]
    esm  = r["esm3"]["mean"]
    sap  = r["saprot"]["mean"]
    L    = r["length"]
    chain = CHAIN[pdb]
    desc  = DESC[pdb]
    lines.append(
        f"  '{pdb}': {{ esm3: {esm:.2f}, saprot: {sap:.2f}, len: {L:3d},"
        f" name: '{pdb}', chain: '{chain}', desc: '{desc}' }},"
    )

lines.append("};")
new_block = "\n".join(lines)

with open(HTML_PATH) as f:
    html = f.read()

# Replace the DEMO_DATA block
pattern = r"// ΔG values.*?^const DEMO_DATA = \{.*?^\};"
replacement = new_block
new_html = re.sub(pattern, new_block, html, flags=re.DOTALL | re.MULTILINE)

if new_html == html:
    print("WARNING: DEMO_DATA block not found — no changes made")
else:
    with open(HTML_PATH, "w") as f:
        f.write(new_html)
    print(f"Updated {HTML_PATH} with real predictions from {RESULTS_PATH}")
    for pdb, r in results.items():
        if pdb in DESC:
            print(f"  {pdb.upper()}: ESM3ΔG {r['esm3']['mean']:.3f}, SaProtΔG {r['saprot']['mean']:.3f} kcal/mol")
