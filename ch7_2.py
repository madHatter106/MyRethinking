import numpy as np
import pandas as pd
import scipy.stats as st
import pymc3 as pm
import arviz as ar
import matplotlib.pyplot as pl

"""Using the hominin brain example:"""

sppnames = ["afarensis", "africanus", "habilis", "boisei", "rudolfensis",
            "ergaster", "sapiens", ]
brainvolcc = [438, 452, 612, 521, 752, 871, 1350]
masskg = [37.0, 35.5, 34.5, 41.5, 55.5, 61.0, 53.5]

d = pd.DataFrame(dict(species=sppnames, brain=brainvolcc, mass=masskg))
# standardize inputs:
d['mass_s'] = (d.mass - d.mass.mean()) / d.mass.std()
d['brain_s'] = (d.brain - d.brain.mean()) / d.brain.std()
d.describe()
pd.plotting.scatter_matrix(d[['mass_s', 'brain_s']]);

with pm.Model() as model:
    α = pm.Normal('α', mu=0, sd=1)
    β = pm.Normal('β', 0, sd=1)
    μ =  α + β * d.mass_s
    σ = pm.Exponential('σ', 1)
    brain = pm.Normal('brain', mu=μ, sd=σ, observed=d.brain)
    #trace = pm.sample(1000, tune=1000)
    theta = pm.find_MAP(start=dict(α=d.brain.mean(),
                                   β=0, σ=d.brain.std())
                        )
#theta = pm.summary(trace)['mean']
theta
log_lik = st.norm.logpdf(d.brain.values,
                         loc=theta['α'] + theta['β'] * d.mass_s.values,
                         scale=theta['σ'])
-2 * log_lik.sum()

"""Running in-sample/out-sample simulation"""c

from sim_train_test import simulate

n_sim = 10000
n_params = 5
dev_trn = np.empty((n_sim, n_params))
mae_trn = np.empty_like(dev_trn)
r2_trn = np.empty_like(dev_trn)
dev_tst = np.empty_like(dev_trn)
mae_tst = np.empty_like(dev_trn)
r2_tst = np.empty_like(dev_trn)

for i in range(n_sim):
    dict_trn, dict_tst = simulate()
    dev_trn[i, :] = dict_trn['deviance']
    mae_trn[i, :] = dict_trn['mae']
    r2_trn[i, :] = dict_trn['r2']
    dev_tst[i, :] = dict_tst['deviance']
    mae_tst[i, :] = dict_tst['mae']
    r2_tst[i, :] = dict_tst['r2']
def plot_train_test(ax, trn, tst, title=None):

    ax.errorbar(np.arange(1, 6), trn.mean(axis=0), yerr=trn.std(axis=0),
                linestyle='', label='train', marker='o')
    ax.errorbar(np.arange(1, 6), tst.mean(axis=0), yerr=tst.std(axis=0),
                linestyle='', label='test', marker='o')
    ax.legend()
    ax.set_title(title, fontsize=12)

res = dict(train=dict(dev=dev_trn, mae=mae_trn, r2=r2_trn),
           test=dict(dev=dev_tst, mae=mae_tst, r2=r2_tst)
           )

import pickle
with open('./ch7_2_sim_results.pkl', 'wb') as fb:
    pickle.dump(res, fb, protocol=pickle.HIGHEST_PROTOCOL)

#with open('./ch7_2_sim_results.pkl', 'rb') as fb:#
#   res = pickle.load(fb)

f, ax = pl.subplots(nrows=3, figsize=(6, 12), sharex=True)

plot_train_test(ax[0], dev_trn, dev_tst, title='deviance')
plot_train_test(ax[1], r2_trn, r2_tst, title='r2')
plot_train_test(ax[2], mae_trn, mae_tst, title='mae')
