import numpy as np
import pandas as pd
from causalgraphicalmodels import CausalGraphicalModel
import arviz as ar
import pymc3 as pm
import theano.tensor as tt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as pl


sppnames = ["afarensis", "africanus", "habilis", "boisei", "rudolfensis",
            "ergaster", "sapiens", ]
brainvolcc = [438, 452, 612, 521, 752, 871, 1350]
masskg = [37.0, 35.5, 34.5, 41.5, 55.5, 61.0, 53.5]

d = pd.DataFrame(dict(species=sppnames, brain=brainvolcc, mass=masskg))
# standardize inputs:
d['mass_s'] = (d.mass - d.mass.mean()) / d.mass.std()
d['brain_s'] = d.brain / d.brain.max()
d.head()

def compute_r2(trace, model, obs_y, likelihood_name='b'):
    lkl = pm.sample_posterior_predictive(trace=trace, model=model)[likelihood_name]
    return r2_score(obs_y, lkl.mean(axis=0))

def compute_maes(trace, model, obs_y, likelihood_name='b'):
    lkl = pm.sample_posterior_predictive(trace=trace, model=model)[likelihood_name]
    return mean_absolute_error(obs_y, lkl.mean(axis=0))


x = d.mass_s.values.reshape(-1, 1)
y = d.brain_s.values
X = np.array([x**i for i in range(1,7)]).squeeze()
with pm.Model() as m1:
    σ = pm.Exponential('σ', 1)
    β = pm.Normal('β', 0, 10)
    α = pm.Normal('α', 0.5, 1)
    μi = α + β * X[0]
    b = pm.Normal('b', mu=μi, sd=σ, observed=d.brain_s.values)
    trc1 = pm.sample(tune=1000)
r2_scores = []
mae = []
r2_scores.append(compute_r2(trc1, m1, y,))
mae.append(compute_maes(trc1, m1, y))

with pm.Model() as m2:
    σ = pm.Exponential('σ', 1)
    β = pm.Normal('β', 0, 10, shape=2)
    α = pm.Normal('α', 0.5, 1)
    μi = α + tt.dot(β, X[:2])
    b = pm.Normal('b', mu=μi, sd=σ, observed=y)
    trc2 = pm.sample(tune=2000)
r2_scores.append(compute_r2(trc2, m2, y))
mae.append(compute_maes(trc2, m2, y))

pwr = 3
with pm.Model() as m3:
    σ = pm.Exponential('σ', 1)
    β = pm.Normal('β', 0, 10, shape=pwr)
    α = pm.Normal('α', 0.5, 1)
    μi = α + tt.dot(β, X[:pwr])
    b = pm.Normal('b', mu=μi, sd=σ, observed=y)
    trc3 = pm.sample(tune=2000)
r2_scores.append(compute_r2(trc3, m3, y))
mae.append(compute_maes(trc3, m3, y))

pwr=4
with pm.Model() as m4:
    σ = pm.Exponential('σ', 1)
    β = pm.Normal('β', 0, 10, shape=pwr)
    α = pm.Normal('α', 0.5, 1)
    μi = α + tt.dot(β, X[:pwr])
    b = pm.Normal('b', mu=μi, sd=σ, observed=y)
    trc4 = pm.sample(tune=2000)
r2_scores.append(compute_r2(trc4, m4, y))
mae.append(compute_maes(trc4, m4, y))

pwr=5
with pm.Model() as m5:
    σ = pm.Exponential('σ', 1)
    β = pm.Normal('β', 0, 10, shape=pwr)
    α = pm.Normal('α', 0.5, 1)
    μi = α + tt.dot(β, X[:pwr])
    b = pm.Normal('b', mu=μi, sd=σ, observed=y)
    trc5 = pm.sample(tune=2000)
r2_scores.append(compute_r2(trc5, m5, y))
mae.append(compute_maes(trc5, m5, y))

pwr=6
with pm.Model() as m6:
    β = pm.Normal('β', 0, 10, shape=pwr)
    α = pm.Normal('α', 0.5, 1)
    μi = α + tt.dot(β, X[:pwr])
    b = pm.Normal('b', mu=μi, sd=0.001, observed=y)
    trc6 = pm.sample(tune=2000)
r2_scores.append(compute_r2(trc6, m6, y))
mae.append(compute_maes(trc6, m6, y))


_, axs = pl.subplots(ncols=2)
axs[0].plot(np.arange(1, 7), r2_scores, marker='.', color='k')
axs[1].plot(np.arange(1, 7), mae, marker='.', color='k')
axs[0].set_xlabel(r'$^\circ$ Polynomial', fontsize=12)
axs[1].set_xlabel(r'$^\circ$ Polynomial', fontsize=12);



traces = [trc1, trc2, trc3, trc4, trc5, trc6]
x_dummy = np.linspace(x.min(), x.max(), 100)
x
X_dummy = np.vstack([x_dummy**(i) for i in range(1, 7)])
_, axs = pl.subplots(ncols=2, nrows=3, figsize=(10, 8),
                     sharex=True,)
for i, (trc, ax) in enumerate(zip(traces, axs.flat)):
    α_pos = trc.get_values('α').reshape(-1,1 )
    β_pos = trc.get_values('β', combine=True, squeeze=False)[0]
    if β_pos.ndim == 1:
        β_pos = β_pos.reshape(-1, 1)
    lines = α_pos + β_pos.dot(X_dummy[:i+1])
    line = lines.mean(axis=0)
    line_hpd = pm.hpd(lines)
    lbl = 'R^2: %.2f \n MAE: %.2f' %(r2_scores[i], mae[i])
    ax.scatter(x, y, color='k')
    ax.plot(x_dummy, line, color='b', label=lbl)
    ax.fill_between(x_dummy.squeeze(), line_hpd[:,0], line_hpd[:, 1], alpha=0.5, color='k')
    ax.legend()
    ax.set_xticklabels(['%.1f' % (ai * d.mass.std() + d.mass.mean()) for ai in ax.get_xticks()])
    ax.set_yticklabels(['%.1f' % (ai * d.brain.std() + d.brain.mean()) for ai in ax.get_yticks()])
