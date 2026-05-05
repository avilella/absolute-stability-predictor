# Handling Missing Residues in SaProtABS

This guide explains how SaProtABS handles missing residues in protein structures and provides different strategies for dealing with them.

## Overview

Missing residues in PDB files can occur due to:
- Poor electron density in certain regions
- Flexible loops that couldn't be resolved
- Experimental limitations
- Incomplete structure determination

SaProtABS now provides multiple strategies to handle these missing residues, allowing you to choose the most appropriate approach for your specific case.

## Available Strategies

### 1. **Mask** (Default)
- **What it does**: Replaces missing residues with "#" characters
- **Best for**: Low percentages of missing residues (< 5%)
- **Pros**: Preserves sequence length and position information
- **Cons**: May affect model performance if too many residues are masked

```python
# Use default masking
pred_dg, pred_dg_avg, combined_seq = SaProtABS_predict(
    model=model,
    pdb_path="protein.pdb",
    handle_missing_residues="mask"  # Default behavior
)
```

### 2. **Skip**
- **What it does**: Removes missing residues entirely from the sequence
- **Best for**: High percentages of missing residues (> 15%) or large gaps
- **Pros**: Clean sequence without gaps
- **Cons**: Changes sequence length and position numbering

```python
# Skip missing residues
pred_dg, pred_dg_avg, combined_seq = SaProtABS_predict(
    model=model,
    pdb_path="protein.pdb",
    handle_missing_residues="skip"
)
```

### 3. **Interpolate**
- **What it does**: Fills gaps based on surrounding residues
- **Best for**: Moderate missing residues (5-15%)
- **Pros**: Preserves sequence length and may maintain structural context
- **Cons**: Introduces artificial residues that may not reflect reality

```python
# Interpolate missing residues
pred_dg, pred_dg_avg, combined_seq = SaProtABS_predict(
    model=model,
    pdb_path="protein.pdb",
    handle_missing_residues="interpolate"
)
```

### 4. **Keep**
- **What it does**: Uses the sequence as-is (may cause errors)
- **Best for**: Testing or when you want to see raw output
- **Pros**: No modification of original data
- **Cons**: Likely to cause errors in the model

```python
# Keep missing residues as-is
pred_dg, pred_dg_avg, combined_seq = SaProtABS_predict(
    model=model,
    pdb_path="protein.pdb",
    handle_missing_residues="keep"
)
```

## Analyzing Missing Residues

Before choosing a strategy, you can analyze your PDB file to understand the extent of missing residues:

```python
from utils.foldseek_util import print_missing_residue_analysis, analyze_missing_residues

# Print a detailed analysis
print_missing_residue_analysis("protein.pdb", chain_id='A')

# Get analysis results programmatically
analysis = analyze_missing_residues("protein.pdb", chain_id='A')
print(f"Missing: {analysis['missing_residues']}/{analysis['total_residues']} ({analysis['missing_percentage']:.1f}%)")
print(f"Recommended strategy: {analysis['recommended_strategy']}")
```

Example output:
```
=== Missing Residue Analysis for protein.pdb (Chain A) ===
Total residues: 150
Missing residues: 8
Missing percentage: 5.3%
Recommended strategy: interpolate
Reason: Moderate missing residues, interpolation may preserve structure
Missing residue positions: [45, 46, 67, 68, 69, 120, 121, 122]
Maximum consecutive gaps: 3

Strategy options:
- 'mask': Replace missing residues with '#' (good for low missing rates)
- 'skip': Remove missing residues entirely (good for high missing rates)
- 'interpolate': Fill gaps based on surrounding residues (moderate missing rates)
- 'keep': Use as-is (may cause errors)
============================================================
```

## Choosing the Right Strategy

### Decision Tree

1. **< 5% missing residues**: Use `mask`
   - Safe default that preserves structure
   - Minimal impact on predictions

2. **5-15% missing residues**: Use `interpolate`
   - Balances preservation with completeness
   - Good for moderate gaps

3. **15-30% missing residues**: Use `skip`
   - Removes problematic regions
   - May give cleaner results

4. **> 30% missing residues**: Consider finding a better structure
   - Structure may be too incomplete for reliable predictions
   - Try `skip` as last resort

### Special Cases

- **Large consecutive gaps (>5 residues)**: Use `skip`
- **N-terminal or C-terminal missing**: Use `mask` or `skip`
- **Internal loops missing**: Use `interpolate` or `skip`

## Example Usage

```python
from SaProtABS import SaProtABS, SaProtABS_predict
from utils.foldseek_util import analyze_missing_residues

# Load model
model = SaProtABS("path/to/model.ckpt")

# Analyze missing residues
analysis = analyze_missing_residues("protein.pdb", chain_id='A')
strategy = analysis['recommended_strategy']

print(f"Using recommended strategy: {strategy}")

# Run prediction with recommended strategy
pred_dg, pred_dg_avg, combined_seq = SaProtABS_predict(
    model=model,
    pdb_path="protein.pdb",
    chain_id='A',
    handle_missing_residues=strategy
)

print(f"Predicted ΔG: {pred_dg_avg[0]:.3f}")
```

## Comparing Strategies

You can compare different strategies on the same structure:

```python
strategies = ["mask", "skip", "interpolate"]
results = {}

for strategy in strategies:
    try:
        pred_dg, pred_dg_avg, combined_seq = SaProtABS_predict(
            model=model,
            pdb_path="protein.pdb",
            handle_missing_residues=strategy
        )
        results[strategy] = {
            'dg': pred_dg_avg[0],
            'length': len(combined_seq),
            'success': True
        }
    except Exception as e:
        results[strategy] = {'success': False, 'error': str(e)}

# Print comparison
for strategy, result in results.items():
    if result['success']:
        print(f"{strategy}: ΔG={result['dg']:.3f}, Length={result['length']}")
    else:
        print(f"{strategy}: FAILED - {result['error']}")
```

## Best Practices

1. **Always analyze first**: Use `analyze_missing_residues()` to understand your data
2. **Start with recommended**: Use the automatically recommended strategy
3. **Compare if uncertain**: Run multiple strategies and compare results
4. **Consider the biological context**: Missing residues in functional regions may need special handling
5. **Document your choice**: Note which strategy you used in your analysis

## Troubleshooting

### Common Issues

1. **Model errors with "keep" strategy**: Expected - missing residues can cause tokenization issues
2. **Very different results between strategies**: Normal for structures with many missing residues
3. **No prediction returned**: Try a different strategy or check if the structure is too incomplete

### Error Messages

- `"Invalid handle_missing_residues option"`: Check spelling of strategy name
- `"Pdb file not found"`: Verify file path
- `"Chain X not found"`: Check chain ID in PDB file

## Advanced Usage

For batch processing or custom analysis, you can use the analysis functions directly:

```python
# Batch analysis
pdb_files = ["protein1.pdb", "protein2.pdb", "protein3.pdb"]
for pdb_file in pdb_files:
    analysis = analyze_missing_residues(pdb_file)
    print(f"{pdb_file}: {analysis['missing_percentage']:.1f}% missing")
```

This comprehensive approach to missing residue handling ensures that SaProtABS can work effectively with a wide range of protein structures, from complete high-resolution structures to those with significant missing regions. 