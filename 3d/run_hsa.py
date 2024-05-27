from synergy.combination import MuSyC, HSA
from synergy.utils.dose_tools import grid
from synergy.utils import plots

import pandas as pd
import numpy as np
from scipy.stats import ttest_1samp

df = pd.read_csv("data/data.csv")

d1 = df['drug1.conc']
d2 = df['drug2.conc']
E = df['effect']

hsa = HSA()
synergy = hsa.fit(d1, d2, E)

syn = pd.DataFrame({'drug1.conc':d1, 'drug2.conc':d2, 'effect':E, 'hsa':synergy})
syn.to_csv("results/hsa/BxPC-3_3D_culture_hsa.csv")

# The rest deals with the fact that some (2) datapoints from Panc1 were removed
# the current synergy library doesn't like plotting that using hsa.plot_heatmap()
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

X1, X2, Z = square_dose_effect(d1, d2, synergy)
Z[(X1==0) | (X2==0)] = np.nan
plots.plot_heatmap(X1, X2, Z, fname="results/hsa/BxPC-3_3D_culture_hsa.pdf", xlabel="Gemcitabine (μM)", ylabel="MSeA (μM)", title="BxPC-3 3D Culture - HSA synergy ", center_on_zero=True, figsize=(4.5,3.4))
plots.plot_heatmap(X1, X2, Z, fname="results/hsa/BxPC-3_3D_culture_hsa.png", xlabel="Gemcitabine (μM)", ylabel="MSeA (μM)", title="BxPC-3 3D Culture - HSA synergy ", center_on_zero=True, figsize=(4.5,3.4))


hsa_pval = Z*0+1
for _i in range(len(Z)):
    _d1 = X1[_i]
    _d2 = X2[_i]
    syn = synergy[(d1==_d1)&(d2==_d2)]
    hsa_pval[_i] = ttest_1samp(syn,0).pvalue


plots.plot_heatmap(X1, X2, -np.log10(hsa_pval), fname="results/hsa/BxPC-3_3D_culture_pvals.pdf", xlabel="Gemcitabine (μM)", ylabel="MSeA (μM)", title="HSA (-log10(p-val))", center_on_zero=False, figsize=(4.5,3.4), vmin=0, vmax=-2*np.log10(0.05))
plots.plot_heatmap(X1, X2, -np.log10(hsa_pval), fname="results/hsa/BxPC-3_3D_culture_pvals.png", xlabel="Gemcitabine (μM)", ylabel="MSeA (μM)", title="HSA (-log10(p-val))", center_on_zero=False, figsize=(4.5,3.4), vmin=0, vmax=-2*np.log10(0.05))