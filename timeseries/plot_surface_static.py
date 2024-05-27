from synergy.combination import MuSyC
from synergy.utils.dose_tools import grid
import pandas as pd
import numpy as np

cline = 'BxPC-3'
for _time in [24,48,72]:
    params = pd.read_csv("results/musyc/%d_parameters.csv"%_time, index_col=0)['0']
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

    df = pd.read_csv("data/mtt_gem_MSeA_%d.csv"%_time)
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

    # dose=0 cannot be properly shown on log-scaled axes
    # Here d1_min and d2_min are used to plot the data, and d=0 is put there
    # However for some zero doses, the visualized dose needs to be lower than what
    # this does by default. Thus we manually adjust d1_min and d2_min here in those
    # cases. This changes no data, just makes the visualization show the actual E0
    # asymptote.
    if _time>=48: d1_min = d1_min/100.
    D1, D2 = grid(d1_min, d1_max, d2_min, d2_max, 40, 40, include_zero=True)

    for fmt in ['pdf','png']:
        model.plot_surface_plotly(D1, D2,
            xlabel="log(Gemcitabine) (μM)", ylabel="log(MSeA) (μM)", zlabel="MTT",
            title="%s 2D Culture at %d Hours MuSyC Fit"%(cline, _time),
            scatter_points=df,
            fname="results/musyc/%d_2D.%s"%(_time,fmt)
        )