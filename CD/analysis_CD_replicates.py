import pandas as pd
from Bio.SeqUtils import ProtParam
import numpy as np
import matplotlib.pyplot as plt
import os

## full dataset
import pandas as pd
df = pd.read_csv('/home/jupyter-yehlin/Protein_Stability/240404_K50dG_dmsv4_dmsv5_dmsv7_concat240404.csv')

# aa sequence from rocklin_batch2
rocklin_batch2_seq = {
    "DMMG_1": "FKPYLVRVTITNLNIRKGPGTNYDRTGKYTGIGSFTIVDEADGEGASKWGLLKSYQSGRNGWVSLDYAKR",
    "DMMG_02": "TPPKQLRKEILEVFKRQPNKPLNHKQVASTIGIVNHDMRSKISVMLEEMGRDGRLESIGRGKFVLAEMQQD",
    "DMMG_03": "AQRMAVQGTVVDSSGEPVIGANVIVRGSSAGVATDLDGRFRLDVVPDATLVVSYLGYNTQEVPVNNRSQITIVLQENAVA",
    "DMMG_04": "KAKWLDGPDTGYAADHNGQPMVAHGMEGLFTKSGSNDGFIVPFDNAATRNNPMLTMTRLAQAKRLGFTNAP",
    "DMMG_05": "QNQIEYSAFISDVEGGRVQSVSIEGHPLRGQWIKGRRADGSAFATYAPYDPQLVNQLIKSNVRFSAKP",
    "DMMG_06": "TTSYTVKITADVLNVRAGAGTNYKVNTQVRKGEVYTIVGESNGWGKLKSGAGWISLEYTSKN",
    "DMMG_07": "TLSLGQQLKQSREALQLSIEDVVQKTNLKKSHIESLENDIFILQNVAPTFVRGYVRNYLRFLRLPEDLASSVNYGEV",
    "DMMG_08": "KKHILTYAGLKQYEDELQNLKVVRRKEVAQKIKEAREQGDLSENAEYDAAKDEQRDIEARIEQLEAMLKNIEVV",
    "DMMG_09": "PRTVSDQVADQLRTLVASGEFKPGDRIPAERDLAARLGVGRPAVREALRELRAQGLLVTGRGAQGTTVASR",
    "DMMG_10": "STWDHYFSEASIYHAEHGNLDIPRRYKTPMGLSLGAWLQIQRQVRSGKRAGSLTQQQIERLDSIGMRWD",
    "DMMG_11": "SNGKTRFFINIGEMDRIDDRELIELVSMQTRLSIDDISVDKIEDKHSYFEITEQHSEKVMDDFNGFTFRGRDIRVDKAAR",
    "DMMG_12": "DGDKRRKEIMELLNTEKDPLSGTSLAKRLGVSRQVIVQDIALLRATNRNILSTNRGYMLY",
    "DMMG_13": "TKFQAYLRIELADRENDLDFSENLIHQSYNSNNYSYSLGFKNIEDIYKTLDNLEESKILNYSIDKS",
    "DMMG_14": "GDYIKYNGNGILYVNLRPGESAQLYIADLQGRRVASTTMTGSGEIYMDSLAKGIYIVSLTTGSGETAAVKIS",
    "DMMG_15": "RNDRKWNEGYQEAKRYFDAHGDLNVPAEYVSPGGYNLGNWVKRQRYTRQNPEKSGAVLTEERIAKLDAIGMRWG",
    "DMMG_16": "LTDDDWMKFRAVFEKVHPEFIIRQKEQFPDLTPAETRLLALEKLDLSTQEMANMLGVNKNTIHQTRLRMRRKTAG",
    "DMMG_17": "QKTYEKIDEISGELGIKEGEKTIFEIVPTSDPNQMSLSLKSGSWDGVEPWFGIDENQNLHTMVSLKSL",
    "DMMG_18": "GRDEVFSKLLEEIFNQVLLAQSSEQVGAEPYERTEERTAYRNGFRDRQMTTRVGTLTLRVPRHRNGK",
    "DMMG_19": "KQRSQEALTMGNRLLRHALGAVKLDDISQESVDRVVAETKHESFEDLLVDIGLGNELSAIVARRLLG",
    "DMMG_20": "TAETKTIKVGSKVKVKKGAKTYTGGSLASFVYNTVYDVLQINGNRVVIGLKGQVTAAVRLEDLIL",
    "DMMG_21": "FHSKQTLSTDGSTTTFTLDFAVADETSIIVSVGGVLQEPKVAYNLAAGGTQITFTAAPASTDTAYIQFLGQA",
    "DMMG_42": "ELVWQDVINQLAEQNRPSAKSLLEQADYRYSASTNTLTLFFGKTFHRKQAKTGRFQDSLLTTLDELYQVKPTIVISEE",
    "DMMG_44": "YKNFQYTVSMEDFNMSLTLKDYLRQKFNFSSRLMTKIKKNKGLFLNGESAPGWIVPKENDVITVNLPKE",
    "DMMG_45": "SAQDKLHQEFEKINAFVAEHGRLPKNDGDSFQEKLLARTLAKLFSNDAHRQALADADVHGIFSR",
    "DMMG_46": "SEWIYGFHAVEGVIKRSPERITALWLATNRRDKRAQEIEALAQTQGVAVTRVERYQLDMEIDGRHQGIAVSVQ"
}

# aa sequence for CD experiment, with aa length for mean residue ellipticity calculation
cd_experiment_seq = {
    "DMMG_1": {
        "sequence": "GSSHHHHHHSSGLVPRGSHMFKPYLVRVTITNLNIRKGPGTNYDRTGKYTGIGSFTIVDEADGEGASKWGLLKSYQSGRNGWVSLDYAKR",
        "length": 90
    },
    "DMMG_02": {
        "sequence": "GSSHHHHHHSSGLVPRGSHMTPPKQLRKEILEVFKRQPNKPLNHKQVASTIGIVNHDMRSKISVMLEEMGRDGRLESIGRGKFVLAEMQQD",
        "length": 91
    },
    "DMMG_03": {
        "sequence": "GSSHHHHHHSSGLVPRGSHMAQRMAVQGTVVDSSGEPVIGANVIVRGSSAGVATDLDGRFRLDVVPDATLVVSYLGYNTQEVPVNNRSQITIVLQENAVA",
        "length": 100
    },
    "DMMG_04": {
        "sequence": "GSSHHHHHHSSGLVPRGSHMKAKWLDGPDTGYAADHNGQPMVAHGMEGLFTKSGSNDGFIVPFDNAATRNNPMLTMTRLAQAKRLGFTNAP",
        "length": 91
    },
    "DMMG_05": {
        "sequence": "GSSHHHHHHSSGLVPRGSHMQNQIEYSAFISDVEGGRVQSVSIEGHPLRGQWIKGRRADGSAFATYAPYDPQLVNQLIKSNVRFSAKP",
        "length": 88
    },
    "DMMG_06": {
        "sequence": "GSSHHHHHHSSGLVPRGSHMTTSYTVKITADVLNVRAGAGTNYKVNTQVRKGEVYTIVGESNGWGKLKSGAGWISLEYTSKN",
        "length": 82
    },
    "DMMG_07": {
        "sequence": "GSSHHHHHHSSGLVPRGSHMTLSLGQQLKQSREALQLSIEDVVQKTNLKKSHIESLENDIFILQNVAPTFVRGYVRNYLRFLRLPEDLASSVNYGEV",
        "length": 97
    },
    "DMMG_08": {
        "sequence": "GSSHHHHHHSSGLVPRGSHMKKHILTYAGLKQYEDELQNLKVVRRKEVAQKIKEAREQGDLSENAEYDAAKDEQRDIEARIEQLEAMLKNIEVV",
        "length": 94
    },
    "DMMG_09": {
        "sequence": "GSSHHHHHHSSGLVPRGSHMPRTVSDQVADQLRTLVASGEFKPGDRIPAERDLAARLGVGRPAVREALRELRAQGLLVTGRGAQGTTVASR",
        "length": 91
    },
    "DMMG_10": {
        "sequence": "GSSHHHHHHSSGLVPRGSHMSTWDHYFSEASIYHAEHGNLDIPRRYKTPMGLSLGAWLQIQRQVRSGKRAGSLTQQQIERLDSIGMRWD",
        "length": 89
    },
    "DMMG_11": {
        "sequence": "GSSHHHHHHSSGLVPRGSHMSNGKTRFFINIGEMDRIDDRELIELVSMQTRLSIDDISVDKIEDKHSYFEITEQHSEKVMDDFNGFTFRGRDIRVDKAAR",
        "length": 100
    },
    "DMMG_12": {
        "sequence": "GSSHHHHHHSSGLVPRGSHMDGDKRRKEIMELLNTEKDPLSGTSLAKRLGVSRQVIVQDIALLRATNRNILSTNRGYMLY",
        "length": 80
    },
    "DMMG_13": {
        "sequence": "GSSHHHHHHSSGLVPRGSHMTKFQAYLRIELADRENDLDFSENLIHQSYNSNNYSYSLGFKNIEDIYKTLDNLEESKILNYSIDKS",
        "length": 86
    },
    "DMMG_14": {
        "sequence": "GSSHHHHHHSSGLVPRGSHMGDYIKYNGNGILYVNLRPGESAQLYIADLQGRRVASTTMTGSGEIYMDSLAKGIYIVSLTTGSGETAAVKIS",
        "length": 92
    },
    "DMMG_15": {
        "sequence": "GSSHHHHHHSSGLVPRGSHMRNDRKWNEGYQEAKRYFDAHGDLNVPAEYVSPGGYNLGNWVKRQRYTRQNPEKSGAVLTEERIAKLDAIGMRWG",
        "length": 94
    },
    "DMMG_16": {
        "sequence": "GSSHHHHHHSSGLVPRGSHMLTDDDWMKFRAVFEKVHPEFIIRQKEQFPDLTPAETRLLALEKLDLSTQEMANMLGVNKNTIHQTRLRMRRKTAG",
        "length": 95
    },
    "DMMG_17": {
        "sequence": "GSSHHHHHHSSGLVPRGSHMQKTYEKIDEISGELGIKEGEKTIFEIVPTSDPNQMSLSLKSGSWDGVEPWFGIDENQNLHTMVSLKSL",
        "length": 88
    },
    "DMMG_18": {
        "sequence": "GSSHHHHHHSSGLVPRGSHMGRDEVFSKLLEEIFNQVLLAQSSEQVGAEPYERTEERTAYRNGFRDRQMTTRVGTLTLRVPRHRNGK",
        "length": 87
    },
    "DMMG_19": {
        "sequence": "GSSHHHHHHSSGLVPRGSHMKQRSQEALTMGNRLLRHALGAVKLDDISQESVDRVVAETKHESFEDLLVDIGLGNELSAIVARRLLG",
        "length": 87
    },
    "DMMG_20": {
        "sequence": "GSSHHHHHHSSGLVPRGSHMTAETKTIKVGSKVKVKKGAKTYTGGSLASFVYNTVYDVLQINGNRVVIGLKGQVTAAVRLEDLIL",
        "length": 85
    },
    "DMMG_21": {
        "sequence": "GSSHHHHHHSSGLVPRGSHMFHSKQTLSTDGSTTTFTLDFAVADETSIIVSVGGVLQEPKVAYNLAAGGTQITFTAAPASTDTAYIQFLGQA",
        "length": 92
    },
    "DMMG_42": {
        "sequence": "GSSHHHHHHSSGLVPRGSHMELVWQDVINQLAEQNRPSAKSLLEQADYRYSASTNTLTLFFGKTFHRKQAKTGRFQDSLLTTLDELYQVKPTIVISEE",
        "length": 98
    },
    "DMMG_44": {
        "sequence": "GSSHHHHHHSSGLVPRGSHMYKNFQYTVSMEDFNMSLTLKDYLRQKFNFSSRLMTKIKKNKGLFLNGESAPGWIVPKENDVITVNLPKE",
        "length": 89
    },
    "DMMG_45": {
        "sequence": "GSSHHHHHHSSGLVPRGSHMSAQDKLHQEFEKINAFVAEHGRLPKNDGDSFQEKLLARTLAKLFSNDAHRQALADADVHGIFSR",
        "length": 84
    },
    "DMMG_46": {
        "sequence": "GSSHHHHHHSSGLVPRGSHMSEWIYGFHAVEGVIKRSPERITALWLATNRRDKRAQEIEALAQTQGVAVTRVERYQLDMEIDGRHQGIAVSVQ",
        "length": 93
    }
}


matching_rows = df[df['aa_seq'].isin(rocklin_batch2_seq.values())].copy()
seq_to_key = {v: k for k, v in rocklin_batch2_seq.items()}
matching_rows['rocklin_key'] = matching_rows['aa_seq'].map(seq_to_key)


matching_rows[['rocklin_key', 'name']]
matching_rows['id'] = matching_rows['name'].str.split('_').str[-1]


matching_rows['rocklin_key_id'] = matching_rows['rocklin_key'].str.split('_').str[-1].astype(int)
matching_rows = matching_rows.sort_values(by='rocklin_key_id')

# Get all keys/proteins from cd_experiment_seq
all_names = list(cd_experiment_seq.keys())


all_names = matching_rows[['rocklin_key', 'id']]['rocklin_key'].tolist()
print(all_names)

all_names_id = matching_rows[['rocklin_key', 'id']]['id'].tolist()
print(all_names_id)


## handle replicates

import pandas as pd
from Bio.SeqUtils import ProtParam
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pickle

sns.set(style="white", font_scale=1.1)

all_files = os.listdir('CD_supplement')

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from jax import random

def two_baseline_fit(x, y, name=None):
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    log_mval = numpyro.sample("log_mval", dist.Normal(1,0.5))
    mval = numpyro.deterministic("mval", jax.numpy.exp(log_mval))
    cm = numpyro.sample("cm", dist.Normal(2,3))
    if name == "DMMG_16":
        m_u = numpyro.sample("m_u", dist.Normal(0,0.6))
        m_f = numpyro.deterministic("m_f", m_u)
    elif name == "DMMG_15":
        m_u = numpyro.sample("m_u", dist.Normal(0,1.5))
        m_f = numpyro.sample("m_f", dist.Normal(0,1.5))
    else:
        m_u = numpyro.sample("m_u", dist.Normal(0,3))
        m_f = numpyro.sample("m_f", dist.Normal(0,3))
    b_f = numpyro.sample("b_f", dist.Normal(y[0],10))
    rightb_u = numpyro.sample("rightb_u", dist.Normal(y[-1],10))
    b_u = numpyro.deterministic("b_u", rightb_u - (m_u * x[-1]))
    dg0 = numpyro.deterministic("dg0", mval * cm)
    dg = dg0 - mval*x
    temp_c = 20
    temp_k = temp_c + 273.15
    kbt = numpyro.deterministic('kbt', 0.0019872036 * temp_k)
    Ku = jax.numpy.exp(-dg / kbt)
    f_u = numpyro.deterministic("f_u", Ku / (1 + Ku))
    f_f = 1 - f_u
    signal_u = ((m_u * x) + b_u)
    signal_f = ((m_f * x) + b_f)
    obs_f_u = numpyro.deterministic("obs_f_u", (y - signal_f) / (signal_u - signal_f))
    expected_y = numpyro.deterministic("fit_y", (f_u * signal_u) + (f_f * signal_f))
    numpyro.sample("obs", dist.Normal(expected_y, sigma), obs=y)

# Precompute valid proteins and filepaths for all replicates
protein_file_info = []
valid_names = []

# For all matched names/IDs
for name, name_id in zip(all_names, all_names_id):
    # All files for this protein
    protein_files = [f for f in all_files if f.startswith(name + "_")]
    # Separate by condition
    denature_files = sorted([f for f in protein_files if 'denature' in f.lower()])
    melt_files     = sorted([f for f in protein_files if 'melt' in f.lower()])
    native_files   = sorted([f for f in protein_files if 'native' in f.lower()])
    num_reps = max(len(denature_files), len(melt_files), len(native_files))
    found_any = False
    for rep_idx in range(num_reps):
        # Try to get file for each condition for this replicate number (if fewer, use None)
        f_denature = os.path.join('CD_supplement', denature_files[rep_idx]) if rep_idx < len(denature_files) else None
        f_melt     = os.path.join('CD_supplement', melt_files[rep_idx])     if rep_idx < len(melt_files)     else None
        f_native   = os.path.join('CD_supplement', native_files[rep_idx])   if rep_idx < len(native_files)   else None
        if f_denature and f_melt and f_native:
            valid_names.append((name, rep_idx + 1)) # rep_idx+1 for human-readable replicate number
            protein_file_info.append((name, name_id, f_denature, f_melt, f_native, rep_idx + 1))
            found_any = True
        else:
            # Print warning on missing replicate file
            print(f"Skipping {name} replicate {rep_idx+1}: missing " +
                  f"denature={bool(f_denature)}, melt={bool(f_melt)}, native={bool(f_native)}")
    if not found_any and num_reps == 0:
        print(f"Skipping {name}: no matching files for any replicate.")

d_list = []
all_fit_datapoints = {}

if len(protein_file_info) == 0:
    print("No proteins have the required files. No plots made.")
else:
    fig_rows = len(protein_file_info)
    fig, axes = plt.subplots(nrows=fig_rows, ncols=3, figsize=(14, 3.8 * fig_rows), squeeze=False)

    for idx, (name, name_id, f_denature, f_melted, f_native, replicate_number) in enumerate(protein_file_info):
        try:
            # if name != "DMMG_16":
            #     continue

            df_denature = pd.read_csv(f_denature)
            df_melted   = pd.read_csv(f_melted)
            df_native   = pd.read_csv(f_native)

            seq = cd_experiment_seq[name]['sequence']
            delta_G_cdna = matching_rows.loc[matching_rows['rocklin_key'] == name, 'deltaG'].values[0]
            number_of_residues = len(seq)
            protein_conc_mg_per_ml = 0.03
            cuvette_path_length_cm = 1.0

            molecular_weight = ProtParam.ProteinAnalysis(seq).molecular_weight()
            protein_conc_g_per_l = protein_conc_mg_per_ml * 1.0
            molar_protein_concentration = protein_conc_g_per_l / molecular_weight
            molar_residue_concentration = molar_protein_concentration * number_of_residues

            def add_MRE_column(df, mdeg_col):
                theta_deg = np.array(df[mdeg_col]) / 1000.0
                MRE = theta_deg / (10 * molar_residue_concentration * cuvette_path_length_cm)
                df['MRE'] = MRE
                return df

            df_denature = add_MRE_column(df_denature, 'mDeg')
            df_melted   = add_MRE_column(df_melted, 'mDeg')
            df_native   = add_MRE_column(df_native, 'mDeg')

            # Recalibrate at 260 nm for denatured
            mask_260_denature = df_denature['Wavelength_nm'] == 260.0
            offset_260 = df_denature.loc[mask_260_denature, 'MRE'].iloc[0] if mask_260_denature.any() else 0.0
            df_denature['MRE'] = df_denature['MRE'] - offset_260
            df_native['MRE']   = df_native['MRE'] - offset_260

            # --- Fit two-state model using numpyro ---
            x = np.array(df_melted.get('urea_M', None))
            y = np.array(df_melted.get('MRE', None))
            rng_key = random.PRNGKey(0)
            rng_key, rng_key_ = random.split(rng_key)
            num_samples = 1000

            kernel = NUTS(lambda x, y: two_baseline_fit(x, y, name=name))
            mcmc = MCMC(kernel, num_warmup=1000, num_samples=num_samples)
            mcmc.run(rng_key, x, y)
            samples_1 = mcmc.get_samples()

            # ---- Left panel: original data traces (as before) ----
            ax_traces = axes[idx, 0]
            native_color = 'orangered'
            denatured_color = 'royalblue'
            ax_traces.scatter(df_native['Wavelength_nm'], df_native['MRE'], label='Native', color=native_color, s=10)
            ax_traces.scatter(df_denature['Wavelength_nm'], df_denature['MRE'], label='Denatured', color=denatured_color, s=10)
            ax_traces.plot(df_native['Wavelength_nm'], df_native['MRE'], color=native_color)
            ax_traces.plot(df_denature['Wavelength_nm'], df_denature['MRE'], color=denatured_color)
            ax_traces.set_title(f'ID: {name_id}  Replicate {replicate_number}  Native & Denatured')
            ax_traces.set_xlabel('Wavelength (nm)')
            ax_traces.set_ylabel('MRE\n(10³ deg cm²/dmol)')
            ax_traces.grid(alpha=0.3)
            if idx == 0:
                ax_traces.legend(frameon=False, loc="lower right")
            ax_traces.spines['top'].set_visible(False)
            ax_traces.spines['right'].set_visible(False)

            # ---- Middle panel: show as in single-protein example (data, fit_y, baselines + credible intervals) ----
            ax_middle = axes[idx, 1]

            # DATA POINTS
            ax_middle.scatter(x, y, color='royalblue', s=22, zorder=9, label='Data')

            # Baseline fits: unfolded
            unfolded_med_m = np.median(samples_1['m_u'], axis=0)
            unfolded_med_b = np.median(samples_1['b_u'], axis=0)
            y_unfolded = unfolded_med_m * x + unfolded_med_b
            ax_middle.plot(x, y_unfolded, color='black', lw=1.6, label='Unfolded baseline ($m_u$)', zorder=12)

            all_upperbase = np.array([m * x + b for m, b in zip(samples_1['m_u'], samples_1['b_u'])])
            top_ubase = np.percentile(all_upperbase, 97.5, axis=0)
            bottom_ubase = np.percentile(all_upperbase, 2.5, axis=0)
            ax_middle.fill_between(x, bottom_ubase, top_ubase, color='gray', alpha=0.18, zorder=2)

            # Baseline fits: folded
            folded_med_m = np.median(samples_1['m_f'], axis=0)
            folded_med_b = np.median(samples_1['b_f'], axis=0)
            y_folded = folded_med_m * x + folded_med_b
            ax_middle.plot(x, y_folded, color='black', lw=1.6, linestyle=':', label='Folded baseline ($m_f$)', zorder=13)

            all_lowerbase = np.array([m * x + b for m, b in zip(samples_1['m_f'], samples_1['b_f'])])
            top_lbase = np.percentile(all_lowerbase, 97.5, axis=0)
            bottom_lbase = np.percentile(all_lowerbase, 2.5, axis=0)
            ax_middle.fill_between(x, bottom_lbase, top_lbase, color='gray', alpha=0.18, zorder=2)

            # Median model fit_y (expected y), with no 95% band (as single panel).
            fit_y_median = np.median(samples_1['fit_y'], axis=0)
            ax_middle.plot(x, fit_y_median, color='black', lw=2.4, label="Fit", zorder=14)

            # Vertical line at Cm (median)
            cm_median = np.median(samples_1['cm'])
            ax_middle.axvline(cm_median, color='black', lw=1.7, ls='--', alpha=0.9, zorder=20, label="Cm")

            ax_middle.set_xlabel('[GuHCl], M')
            ax_middle.set_ylabel('CD signal (MRE)')
            ax_middle.set_title(f'CD signal (Replicate {replicate_number})')
            if idx == 0:
                ax_middle.legend(frameon=False, fontsize=8, loc='upper right')
            ax_middle.spines['top'].set_visible(False)
            ax_middle.spines['right'].set_visible(False)
            ax_middle.grid(alpha=0.3)
            
            # Set y-limits with 20% gap
            height = max(y) - min(y)
            ax_middle.set_ylim(min(y) - (height*0.2), max(y) + (height*0.2))

            middle_panel_data = {
                'melted_urea_M': np.array(df_melted['urea_M']),
                'melted_MRE': np.array(df_melted['MRE'])
            }

            # ---- Right panel: (remains unchanged from original selection) ----
            ax_dgu = axes[idx, 2]

            # As before, plot ΔG_U vs denaturant settings
            kbt_median = np.median(samples_1['kbt'])
            obs_f_u = np.median(samples_1['obs_f_u'], axis=0)
            obs_f_u_min = np.percentile(samples_1['obs_f_u'], 2.5, axis=0)
            obs_f_u_max = np.percentile(samples_1['obs_f_u'], 97.5, axis=0)

            mask = (obs_f_u > 0.01) & (obs_f_u < 0.99)
            mask_min = (obs_f_u_min > 0.01) & (obs_f_u_min < 0.99)
            mask_max = (obs_f_u_max > 0.01) & (obs_f_u_max < 0.99)
            obs_fu_filter = mask
            obs_fu_minmax_filter = mask & mask_min & mask_max

            # --- START FILTERING FOR CONTINUITY (no gaps) ACROSS Cm ---
            # Only allow contiguous points around the median Cm
            cm_median = np.median(samples_1['cm'])
            # Defensive: ensure x is sorted
            sorted_indices = np.argsort(x)
            x_sorted = x[sorted_indices]
            obs_fu_minmax_filter_sorted = obs_fu_minmax_filter[sorted_indices]
            # Now, find the index of Cm or closest value in x
            cm_index = np.searchsorted(x_sorted, cm_median)
            # Traverse forward from cm_index: if a False, mask out all after
            for i in range(cm_index, len(obs_fu_minmax_filter_sorted)):
                if not obs_fu_minmax_filter_sorted[i]:
                    obs_fu_minmax_filter_sorted[i:] = False
                    break
            # Traverse backward from cm_index-1: if a False, mask out all before
            for i in range(cm_index-1, -1, -1):
                if not obs_fu_minmax_filter_sorted[i]:
                    obs_fu_minmax_filter_sorted[:i+1] = False
                    break
            # Unsort to match original order for plotting (x, etc)
            reverse_indices = np.argsort(sorted_indices)
            obs_fu_minmax_filter = obs_fu_minmax_filter_sorted[reverse_indices]
            # --- END FILTERING (enforces only contiguous block through Cm_median) ---

            x_plot = x[obs_fu_minmax_filter]
            obs_f_u_plot = obs_f_u[obs_fu_minmax_filter]
            obs_f_u_min_plot = obs_f_u_min[obs_fu_minmax_filter]
            obs_f_u_max_plot = obs_f_u_max[obs_fu_minmax_filter]

            legit_obs_deltaG = -kbt_median * np.log(obs_f_u_plot / (1-obs_f_u_plot))
            legit_obs_deltaG_min = -kbt_median * np.log(obs_f_u_min_plot / (1-obs_f_u_min_plot))
            legit_obs_deltaG_max = -kbt_median * np.log(obs_f_u_max_plot / (1-obs_f_u_max_plot))

            # scatter obs deltaG
            ax_dgu.scatter(x_plot, legit_obs_deltaG, color='royalblue', s=22,  zorder=8, label="Obs. ΔG")
            for idx_in, xind in enumerate(x_plot):
                ax_dgu.plot(
                    [xind, xind],
                    [legit_obs_deltaG_min[idx_in], legit_obs_deltaG_max[idx_in]],
                    linewidth=1,
                    color=sns.color_palette()[0],
                    alpha=0.87,
                    zorder=6
                )

            # Overplot fit line credible interval gray zone, median fit (as in right-panel of 231)
            all_linear_DG = np.array([np.polyval([-m, b], x) for m, b in zip(samples_1['mval'], samples_1['dg0'])])
            dg_fit_upper = np.percentile(all_linear_DG, 97.5, axis=0)
            dg_fit_lower = np.percentile(all_linear_DG, 2.5, axis=0)
            dg0_median = np.median(samples_1['dg0'])
            mval_median = np.median(samples_1['mval'])
            fit_curve = dg0_median - mval_median * x
            ax_dgu.fill_between(x, dg_fit_lower, dg_fit_upper, color='gray', alpha=0.3, label="95% CI", zorder=-2)
            ax_dgu.plot(x, fit_curve, color='black', lw=2, zorder=3, label='Model fit')

            cm_median = np.median(samples_1['cm'])
            ax_dgu.axvline(
                cm_median,
                color='black',
                lw=1.5,
                ls='--',
                alpha=0.85,
                zorder=12,
                label="Cm"
            )

            ax_dgu.set_xlabel('[GuHCl], M')
            ax_dgu.set_ylabel('ΔG(kcal/mol)')
            ax_dgu.set_title('ΔG_H2O = %.1f, 95CI %.1f - %.1f \n ΔG_cdna = %.1f (Rep %d)' % (
                    dg0_median,
                    np.percentile(samples_1['dg0'], 2.5),
                    np.percentile(samples_1['dg0'], 97.5),
                    delta_G_cdna if delta_G_cdna is not None else float('nan'),
                    replicate_number
                ))
            if idx == 0:
                ax_dgu.legend(frameon=False, fontsize=9)
            ax_dgu.grid(alpha=0.3)
            ax_dgu.spines['top'].set_visible(False)
            ax_dgu.spines['right'].set_visible(False)

            d_list.append(dict(
                deltaG_cdna=delta_G_cdna,
                deltaG_fit=dg0_median,
                deltaG_fit_025=np.percentile(samples_1['dg0'], 2.5),
                deltaG_fit_975=np.percentile(samples_1['dg0'], 97.5),
                protein=name,
                replicate=replicate_number
            ))

            all_fit_datapoints[f"{name}_rep{replicate_number}"] = {
                'x_plot': x_plot,
                'obs_deltaG': legit_obs_deltaG,
                'obs_deltaG_min': legit_obs_deltaG_min,
                'obs_deltaG_max': legit_obs_deltaG_max,
                'fit_x': x,
                'fit_curve': fit_curve,
                'cm_median': cm_median,
                'dg0_median': dg0_median,
                'dg0_025': np.percentile(samples_1['dg0'], 2.5),
                'dg0_975': np.percentile(samples_1['dg0'], 97.5),
                'deltaG_cdna': delta_G_cdna,
                'middle_panel_data': middle_panel_data,
                'replicate': replicate_number
            }

        except Exception as e:
            print(f"Failed on {name} (Replicate {replicate_number}): {e}")
            continue

    plt.tight_layout(rect=(0, 0, 1, 1), h_pad=3.2, w_pad=2.2)
    plt.savefig('CD_all_fit_stacked_replicates.png', dpi=300)
    plt.show()

    with open("CD_all_fit_datapoints_replicates.pkl", "wb") as f:
        pickle.dump(all_fit_datapoints, f)
    print(f"Saved datapoints for quick plotting to CD_all_fit_datapoints.pkl")

df_deltaG_summary = pd.DataFrame(d_list)
df_deltaG_summary.to_csv('CD_deltaG_summary_replicates.csv')