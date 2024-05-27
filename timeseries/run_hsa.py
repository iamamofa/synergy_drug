from synergy.combination import HSA
from synergy.utils import plots

from scipy.stats import ttest_1samp
import pandas as pd
import numpy as np

figsize=(4.5,3.4)
vmin=-0.6
vmax=0.6

def square_dose_effect(d1, d2, E):
    DD1, DD2 = np.meshgrid(np.unique(d1), np.unique(d2), indexing='ij')
    DD1 = DD1.flatten()
    DD2 = DD2.flatten()
    out_E = DD1*0
    for _i in range(len(DD1)): 
        dd1 = DD1[_i]
        dd2 = DD2[_i]
        indices = np.where((d1==dd1) & (d2==dd2))[0]
        out_E[_i] = np.mean(E[indices])
    return DD1, DD2, out_E

cline = "BxPC-3"
for _time in [24, 48, 72]:

    df = pd.read_csv("data/mtt_gem_MSeA_%d.csv"%_time)
    d1 = df['drug1.conc']
    d2 = df['drug2.conc']
    E = df['effect']

    hsa = HSA()
    synergy = hsa.fit(d1, d2, E)
    synergy[(d1==0) | (d2==0)] = np.nan # clearer for visualization that there can be no synergy here

    # Write csv
    syn = pd.DataFrame({'drug1.conc':d1, 'drug2.conc':d2, 'effect':E, 'hsa':synergy})
    syn.to_csv("results/hsa/%d.csv"%_time, index=None)

    # Heatmaps can complain with non-equal numbers of repliates
    # square_dose_effect makes d1, d2, and synergy suitable for heatmaps
    # by averaging synergy values at each dose
    X1, X2, Z = square_dose_effect(d1, d2, synergy)
    Z[(X1==0) | (X2==0)] = np.nan
    hsa_pval = Z*0+1
    for _i in range(len(Z)):
        _d1 = X1[_i]
        _d2 = X2[_i]
        syn = synergy[(d1==_d1)&(d2==_d2)]
        hsa_pval[_i] = ttest_1samp(syn,0).pvalue

    # Plot heatmaps
    for fmt in ['pdf', 'png']:
        # Plot HSA heatmaps
        plots.plot_heatmap(X1, X2, Z, fname="results/hsa/%s.%s"%(_time,fmt),
            xlabel="Gemcitabine (μM)", ylabel="MSeA (μM)", title="%s HSA synergy at %d hours"%(cline, _time),
            center_on_zero=True, figsize=figsize, vmin=vmin, vmax=vmax
        )

        # Plot HSA pvals
        plots.plot_heatmap(X1, X2, -np.log10(hsa_pval), fname="results/hsa/%s_pvals.%s"%(_time,fmt),
            xlabel="Gemcitabine (μM)", ylabel="MSeA (μM)", title="HSA (-log10(p-val)) at %d hours"%_time,
            center_on_zero=False, figsize=figsize, vmin=0, vmax=-2*np.log10(0.05)
        )