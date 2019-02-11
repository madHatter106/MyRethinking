import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as ar
from sklearn.preprocessing import scale
import matplotlib.pyplot as pl
from causalgraphicalmodels import CausalGraphicalModel

np.random.seed(1914)
N = 200 # num grant proposals
p = 0.1 # selection proportion
# uncorrelated newsworthiness and trustworthiness
nw = np.random.randn(N)
tw = np.random.randn(N)
# select top 10% of combined scores
s = nw + tw # total score
q = np.quantile(s, 1-p) # top 10% threshold
selected = np.where(s>=q, True, False )
s[selected]
np.corrcoef(tw[selected], nw[selected])
pl.scatter(nw, tw, color='k')
pl.scatter(nw[selected], tw[selected], color='m')
pl.xlabel('Newsworthiness')
pl.ylabel('Trustworthiness');
"""
Multi colinearity
1. Multi colinear legs
"""
N = 100
np.random.seed(909)
# simulate height
height = np.random.normal(10, 2, size=N)
# leg as proportion of height
leg_prop = np.random.uniform(low=0.4, high=0.5, size=N)
leg_left = leg_prop * height + np.random.normal(scale=0.02, size=N)
leg_right = leg_prop * height + np.random.normal(scale=0.02, size=N)
d = pd.DataFrame(dict(height=height, leg_left=leg_left,
                 leg_right=leg_right))
d.head(2)
pd.plotting.scatter_matrix(d);
with pm.Model() as m1:
    br = pm.Normal('br', 2, 10)
    bl = pm.Normal('bl', 2, 10)
    sigma = pm.Exponential('sigma', 1)
    a = pm.Normal('a', 10, 100)
    mu = a + bl * d.leg_left + br * d.leg_right
    height = pm.Normal('height', mu=mu, sd=sigma,
                       observed=d.height)
    trc1 = pm.sample()
pm.summary(trc1, alpha=0.11)
ar.plot_forest(trc1, var_names=['a', 'bl', 'br'],
combined=True, figsize=(5, 2));
pl.axvline(color='k', ls=':');
pm.__version__
