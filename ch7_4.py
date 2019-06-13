import pathlib

import numpy as np
import pandas as pd
import pymc3 as pm
import scipy.stats as stats
from scipy.special import logsumexp

pwd
home = pathlib.Path.home()
root = home / 'DEV-ALL/BAYES/rethinking_2ndEd/rethinking-Experimental/data'
df = pd.read_csv(root / 'cars.csv')

df.head()
with pm.Model() as m:
    α = pm.Normal('α', 0, 100)
    β = pm.Normal('β', 0, 10)
    σ = pm.Uniform('σ', 0, 30)
    μ = α + β * df.speed.values
    dist = pm.Normal('dist', mu=μ, sd=σ, observed=df.dist.values)

with m:
    post = pm.sample(1000, tune=1200)

n_samples = 1000
μ_post = post['α'][:n_samples].reshape(1, -1) + post['β'][:n_samples].reshape(1, -1) * df.speed.values.reshape(-1, 1)
ll = stats.norm.logpdf(df.dist.values.reshape(-1, 1), loc=μ_post, scale=post['σ'][:n_samples].reshape(1, -1))
lppd = logsumexp(ll, axis=1) - np.log(n_samples)
pwaic = np.var(ll, axis=1)
lppd2 = np.log(np.sum(np.exp(ll), axis=1)) - np.log(n_samples)
np.testing.assert_array_almost_equal(lppd2, lppd)
waic = -2*(lppd.sum() - pwaic.sum())
waic
