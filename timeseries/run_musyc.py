from synergy.combination import MuSyC
from synergy.single import Hill
from synergy.utils.dose_tools import grid
from synergy.utils import remove_zeros

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

plot_n = 40

def plot_single(_d, E, mask, model, drug_index,
    xlabel="xlabel", ylabel="MTT", title="Response", fname="f.png", figsize=None):
    
    # dose=0 cannot be properly shown on log-scaled axes
    # remove_zeros() attempts to replace 0's with reasonable doses for visualization
    # However for some zero doses, the visualized dose needs to be lower than what
    # remove_zeros() does by default. Thus we manually set min_buffer here in those
    # cases. This changes no data, just makes the visualization show the actual E0
    # asymptote.
    min_buffer = 0.5
    if "24_gem" in fname: min_buffer=1.5
    elif "48_gem" in fname: min_buffer=2
    elif "72_gem" in fname: min_buffer=1.75
    _d = remove_zeros(_d[mask], min_buffer=min_buffer)
    E = E[mask]

    d = np.logspace(np.log10(min(_d)), np.log10(max(_d)))
    params = model.get_parameters()

    E0 = params['E0'][0]


    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    E0 = params['E0'][0]
    Emax = params['E%d'%drug_index][0]
    h = params['h%d'%drug_index][0]
    C = params['C%d'%drug_index][0]
    if drug_index==1:
        Emax_i = 1
        h_i = 4
        C_i = 6
    else:
        Emax_i = 2
        h_i = 5
        C_i = 7

    hill_best = Hill(E0=E0, Emax=Emax, h=h, C=C).E(d)

    n_bootstraps = model.bootstrap_parameters.shape[0]
    hill_bootstrap = np.zeros((n_bootstraps, len(d)))

    for _i, bootstrap in enumerate(model.bootstrap_parameters):
        E0 = bootstrap[0]
        Emax = bootstrap[Emax_i]
        h = bootstrap[h_i]
        C = bootstrap[C_i]
        hill_bootstrap[_i] = Hill(E0=E0, Emax=Emax, h=h, C=C).E(d)

    hill_lower, hill_upper = np.quantile(hill_bootstrap, q=(0.025, 0.975), axis=0)
        

    ax.plot(d, hill_lower, 'k--')
    ax.plot(d, hill_best, c='k', lw=3)
    ax.plot(d, hill_upper, 'k--')
    ax.scatter(_d, E, s=30, facecolors='none', edgecolors='k', marker='o')
    ax.set_xscale('log')

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.tight_layout()
    plt.savefig(fname)

def plot_heatmaps(d1, d2, model, xlabel, ylabel, fname, cline, time, figsize=(8,3)):
    fig = plt.figure(figsize=figsize)
    ax_fit = fig.add_subplot(131)
    ax_ref = fig.add_subplot(132)
    ax_delta = fig.add_subplot(133)
    plt.title("%s in 2D culture at %d hours"%(cline,time))
    model.plot_heatmap(d1, d2, xlabel=xlabel, ylabel=ylabel, ax=ax_fit, title="MuSyC Best Fit", aggfunc=np.mean)
    model.plot_reference_heatmap(d1, d2, xlabel=xlabel, ylabel=ylabel, ax=ax_ref, title="MuSyC Zero-synergy Reference ", aggfunc=np.mean)
    model.plot_delta_heatmap(d1, d2, xlabel=xlabel, ylabel=ylabel, ax=ax_delta, title="Net Synergy", aggfunc=np.mean)
    plt.tight_layout()
    plt.savefig(fname)

def musyc_to_series(model):
        d = dict()
        params = model.get_parameters()
        for key in params:
            val = params[key]
            if not hasattr(val,"__iter__"): return pd.Series(params)
            d["%s_lower"%key] = val[1][0]
            d[key] = val[0]
            d["%s_upper"%key] = val[1][1]
        d = pd.Series(d)
        d['Converged'] = str(model.converged)
        d['R^2'] = model.r_squared
        d['RSS'] = model.sum_of_squares_residuals
        d['BIC'] = model.bic
        d['AIC'] = model.aic
        d['Bootstrap Iterations'] = model.bootstrap_parameters.shape[0]
        return d

cline = "BxPC-3"
for _time in [24,48,72]:
    df = pd.read_csv("data/mtt_gem_MSeA_%d.csv"%_time)

    d1 = df['drug1.conc']
    d2 = df['drug2.conc']
    E = df['effect']

    # for plotting and EC50 bounds
    d1_min = min(d1[d1>0])/10
    d2_min = min(d2[d2>0])/10
    d1_max = max(d1)*10
    d2_max = max(d2)*10

    C1_bounds = (d1_min, d1_max)
    C2_bounds = (d2_min, d2_max)
    h_bounds = (1e-3,1e3)
    alpha_bounds=(1e-4,1e4)
    gamma_bounds=(1e-4,1e4)

    model = MuSyC(
        E0_bounds=(1,2.5),
        E1_bounds=(0,1.5),
        E2_bounds=(0,1),
        E3_bounds=(0,1),
        C1_bounds=C1_bounds,
        C2_bounds=C2_bounds,
        h1_bounds=h_bounds,
        h2_bounds=h_bounds,
        alpha12_bounds=alpha_bounds,
        alpha21_bounds=alpha_bounds,
        gamma12_bounds=gamma_bounds,
        gamma21_bounds=gamma_bounds
    )

    model.fit(d1, d2, E, bootstrap_iterations=500, seed=8943)

    # dose=0 cannot be properly shown on log-scaled axes
    # Here d1_min and d2_min are used to plot the data, and d=0 is put there
    # However for some zero doses, the visualized dose needs to be lower than what
    # this does by default. Thus we manually adjust d1_min and d2_min here in those
    # cases. This changes no data, just makes the visualization show the actual E0
    # asymptote.
    if _time>=48: d1_min = d1_min/100.

    D1, D2 = grid(d1_min, d1_max, d2_min, d2_max, plot_n, plot_n, include_zero=True)

    #model.plot_surface_plotly(D1, D2, fname="results/musyc/%d.html"%_time, scatter_points=df, xlabel="log(|Gemcitabine|) (μM)", ylabel="log(|MSeA|) (μM)", zlabel="MTT", title="")

    s = musyc_to_series(model)
    s.to_csv("results/musyc/%d_parameters.csv"%_time)


    plot_single(d1, E, d2==0, model, 1,
                xlabel="Gemcitabine (μM)", ylabel="MTT", title="Gemcitabine alone in %s at %d hours"%(cline, _time),
                fname="results/musyc/%d_gem_alone.pdf"%_time, figsize=(4.5,3.4)
    )

    plot_single(d2, E, d1==0, model, 2,
                xlabel="MSeA (μM)", ylabel="MTT", title="MSeA alone in %s at %d hours"%(cline, _time),
                fname="results/musyc/%d_msea_alone.pdf"%_time, figsize=(4.5,3.4)
    )

    plot_heatmaps(d1, d2, model, xlabel="Gemcitabine (log[μM])", ylabel="MSeA (log[μM])", fname="results/musyc/%d_delta_heatmap.pdf"%_time, cline=cline, time=_time, figsize=(10,4))