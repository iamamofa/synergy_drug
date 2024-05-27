from synergy.combination import MuSyC
from synergy.utils.dose_tools import grid
import pandas as pd
import numpy as np

params = pd.read_csv("results/musyc/musyc_parameters.csv", index_col=0)['0']
E0 = float(params['E0'])
E1 = float(params['E1'])
E2 = float(params['E2'])
E3 = float(params['E3'])
h1 = float(params['h1'])
h2 = float(params['h2'])
C1 = float(params['C1'])
C2 = float(params['C2'])
alpha12 = float(params['alpha12'])
alpha21 = float(params['alpha21'])
gamma12 = float(params['gamma12'])
gamma21 = float(params['gamma21'])


df = pd.read_csv("data/data.csv")
d1 = df['drug1.conc']
d2 = df['drug2.conc']
E = df['effect']

d1_min = min(d1[d1>0])/10
d2_min = min(d2[d2>0])/10
d1_max = max(d1)*10
d2_max = max(d2)*10

model = MuSyC(
    E0=E0,
    E1=E1,
    E2=E2,
    E3=E3,
    h1=h1,
    h2=h2,
    C1=C1,
    C2=C2,
    alpha12=alpha12,
    alpha21=alpha21,
    gamma12=gamma12,
    gamma21=gamma21
)
d1_min = d1_min / 1000.
D1, D2 = grid(d1_min, d1_max, d2_min, d2_max, 40, 40, include_zero=True)

for fmt in ['pdf', 'png', 'html']:
    model.plot_surface_plotly(D1, D2,
        xlabel="log(Gemcitabine) (μM)", ylabel="log(MSeA) (μM)", zlabel="MTT",
        title="BxPC-3 3D Culture MuSyC Fit",
        scatter_points=df,# figsize=(800,600),
        fname="results/musyc/BxPC-3_3D.%s"%fmt
    )