import numpy as np
import pandas as pd
import scipy.stats as st
import pymc3 as pm

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
    α = pm.Normal('α', mu=0, sd=0.2)
    β = pm.Normal('β', 0, sd=0.2)
    μ = α + β * d.mass_s
    #σ = pm.Uniform('σ', 0, d.brain.std() * 10)
    σ = pm.Exponential('σ', 1)
    brain = pm.Normal('brain', mu=μ, sd=σ, observed=d.brain_s)
    trace = pm.sample(1000, tune=1000)
theta = pm.summary(trace)['mean']
theta

log_lik = st.norm.logpdf(d.brain_s.values,
                         loc=theta['α'] + theta['β'] * d.mass_s.values, scale=theta['σ'])
-2 * log_lik.sum()
waic = pm.waic(trace, model=model)
waic
waic.WAIC - 2 * waic.p_WAIC
loo = pm.loo(trace, model=model)
