import numpy as np
import pandas as pd
from causalgraphicalmodels import CausalGraphicalModel
import arviz as ar
import pymc3 as pm


sppnames = ["afarensis", "africanus", "habilis", "boisei", "rudolfensis",
            "ergaster", "sapiens", ]
brainvolcc = [438, 452, 612, 521, 752, 871, 1350]
masskg = [37.0, 35.5, 34.5, 41.5, 55.5, 61.0, 53.5]

d = pd.DataFrame(dict(species=sppnames, brain=brainvolcc, mass=masskg))
# standardize inputs:
d['mass_s'] = (d.mass - d.mass.mean()) / d.mass.std()
d['brain_s'] = d.brain / d.brain.max()
d.head()

with pm.Model() as m1:
    σ = pm.Exponential('σ', 1)
    β = pm.Normal('β', 0, 10)
    α = pm.Normal('α', 0.5, 1)
    μi = α + β * d.mass_s.values
    b = pm.Normal('b', mu=μi, sd=σ, observed=d.brain_s.values)
