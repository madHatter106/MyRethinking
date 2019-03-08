import numpy as np
import pandas as pd
import scipy.stats as st
import pymc3 as pm
import pymc3.glm as lm

"""Using the hominin brain example:"""

sppnames = ["afarensis", "africanus", "habilis", "boisei", "rudolfensis",
            "ergaster", "sapiens", ]
brainvolcc = [438, 452, 612, 521, 752, 871, 1350]
masskg = [37.0, 35.5, 34.5, 41.5, 55.5, 61.0, 53.5]

d = pd.DataFrame(dict(species=sppnames, brain=brainvolcc, mass=masskg))
# standardize inputs:
d['mass_s'] = (d.mass - d.mass.mean()) / d.mass.std()
d['brain_s'] = d.brain / d.brain.max()
d.head()
with pm.Model() as model:
    α = pm.Normal('α', 0, 1)
    β = pm.Normal('β', 0, 1)
    μ = α + β * d.mass_s.values
    σ = pm.Flat('σ')
    brain = pm.Normal('brain', mu=μ, sd=σ, observed=d.brain.values)

with model:
    quap = pm.find_MAP(start=dict(α=d.brain.mean(), β=0, σ=d.brain.std()),
                                  method="Nelder-Mead")
quap.keys()
quap['α']

log_lik = st.norm.logpdf(d.brain.values,
                         loc=quap['α'] + quap['β'] * d.mass_s.values, scale=quap['σ'])
-2 * log_lik.sum()
