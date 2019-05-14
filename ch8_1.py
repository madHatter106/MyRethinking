import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
import pymc3 as pm
import arviz as az
import matplotlib.pyplot as pl

da = pd.read_csv('./rethinking-Experimental/data/rugged.csv', delimiter=';')

da.head()
d = da[['rgdppc_2000', 'cont_africa', 'rugged']].copy()
d['log_gdp'] = np.log(d.rgdppc_2000)
dd = d.dropna().copy()
dd[['A0', 'A1']] = pd.get_dummies(d.cont_africa, dtype='bool')
dd.info()
dd.head()
dd['rugged_s'] = dd.rugged / dd.rugged.max()
dd['log_gdp_s'] = dd.log_gdp / dd.log_gdp.mean()

dfinal = dd[['rugged_s', 'A0', 'A1', 'log_gdp_s']].copy()
dfinal.describe()
rbar = dfinal.rugged_s.mean()
dA1 = dfinal[dfinal.A1]
dA0 = dfinal[dfinal.A0]

def sim_prior_lin(model, x, vars=['α', 'β']):
    prior_ = pm.sample_prior_predictive(vars=vars, model=model)
    prior_array = pd.DataFrame(prior_).to_numpy()
    prior_mu = prior_array.dot(x)
    return prior_mu

def link(sim, x):
    sim_array = pd.DataFrame(sim).to_numpy()
    return sim_array.dot(x)


with pm.Model() as m1:
    α = pm.Normal('α', 1, 1)
    β = pm.Normal('β', 0, 1)
    σ = pm.Exponential('σ',  1)
    μi = α + β * (dA1.rugged_s.values - rbar)
    log_yi = pm.Normal('log_yi', μi, σ, observed=dA1.log_gdp_s)

rugged_seq = np.linspace(-0.1, 1.1, num=30).reshape(-1, 1)
rugged_seq.T.shape
x = np.r_[np.ones((1,30)), rugged_seq.T]

m1_μ = sim_prior_lin(m1, x, )
m1_μ.shape

with pm.Model() as m1i:
    α = pm.Normal('α', 1, 0.1)
    β = pm.Normal('β', 0, 0.3)
    σ = pm.Exponential('σ',  1)
    μi = α + β * (dA1.rugged_s.values - rbar)
    log_yi = pm.Normal('log_yi', μi, σ, observed=dA1.log_gdp_s)

m1i_μ = sim_prior_lin(m1i, x)

_, axs = pl.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
for ax, m_μ, ttl in zip(axs, [m1_μ, m1i_μ], ['vague priors', 'informative priors']):
    ax.set_ylim(0.5, 1.5)
    ax.set_title(ttl)
    ax.set_xlabel('ruggedness (normalized)')
    ax.set_ylabel('log GDP (stdized)')
    ax.plot(rugged_seq, m_μ.T, color='k', alpha=0.1);
    ax.axhline(y=dd.log_gdp_s.min(), color='r', ls='--')
    ax.axhline(y=dd.log_gdp_s.max(), color='r', ls='--');


with pm.Model() as m2i:
    α = pm.Normal('α', 1, 0.1)
    β = pm.Normal('β', 0, 0.3)
    σ = pm.Exponential('σ',  1)
    μi = α + β * (dA0.rugged_s.values - rbar)
    log_yi = pm.Normal('log_yi', μi, σ, observed=dA0.log_gdp_s)


with m1i:
    trace_m1i = pm.sample()
with m2i:
    trace_m2i = pm.sample()


post_m1i = pm.trace_to_dataframe(trace_m1i, varnames=['α', 'β'])
post_m1i_μ = link(post_m1i, x)

pm.summary(trace_m1i)
pm.summary(trace_m2i)

dfinal.head()

dfinal['cont_africa'] = dfinal.A1.astype('i')


with pm.Model() as m3:
    # pooled model
    α = pm.Normal('α', 0, 0.1, )
    β = pm.Normal('β', 0, 0.3, )
    σ = pm.Exponential('σ', 1)
    μ = α + β * (dfinal.rugged_s.values - rbar)
    log_gdp_s_i = pm.Normal('log_gdp_s_i', μ, σ, observed=dfinal.log_gdp_s.values)


with pm.Model() as m4:
    # unpooled model
    α = pm.Normal('α', 0, 0.1, shape=2)
    β = pm.Normal('β', 0, 0.3, )
    σ = pm.Exponential('σ', 1)
    μ = α[dfinal.cont_africa.values] + β * (dfinal.rugged_s.values - rbar)
    log_gdp_s_i = pm.Normal('log_gdp_s_i', μ, σ, observed=dfinal.log_gdp_s.values)

with m3:
    trace_m3 = pm.sample()
with m4:
    trace_m4 = pm.sample()
m3.name = 'm3'
m4.name = 'm4'
pm.compare({m3: trace_m3, m4: trace_m4})

pm.summary(trace_m4, alpha=0.11)


# Making the slope conditional
with pm.Model() as m5:
    α = pm.Normal('α', 0, 0.1, shape=2)
    β = pm.Normal('β', 0, 0.3, shape=2)
    σ = pm.Exponential('σ', 1)
    μ = α[dfinal.cont_africa.values] + β[dfinal.cont_africa.values] * (dfinal.rugged_s.values - rbar)
    log_gdp_s_i = pm.Normal('log_gdp_s_i', μ, σ, observed=dfinal.log_gdp_s.values)
    trace_m5 = pm.sample()
pm.summary(trace_m5, alpha=0.11).round(decimals=2)
m5.name = 'm5'
pm.compare({m3: trace_m3, m4: trace_m4, m5: trace_m5}, ic='LOO')
waic_list = pm.waic(trace_m5, model=m5, pointwise=True)

loo_list = pm.loo(trace_m5, model=m5, pointwise=True)

pl.plot(waic_list.WAIC_i, marker='.', ls='', color='k');
pl.plot(loo_list.LOO_i, marker='s', ls='', markeredgecolor='r');


dfinal.head()
# Plotting the interaction

% matplotlib inline
_, axs = pl.subplots(ncols=2, figsize=(8,4))
ttls = ['Non-African', 'African']
df_m5 = pm.trace_to_dataframe(trace_m5)
for i, (axi, ttl) in enumerate(zip(axs, ttls)):
    axs[i].scatter(dfinal[dfinal.cont_africa==i].rugged_s, dfinal[dfinal.cont_africa==i].log_gdp_s)
    m5_μ = link(df_m5.filter(regex=f'{i}', axis=1), x)
    m5_μ_mean = m5_μ.mean(axis=0)
    m5_ci = pm.hpd(m5_μ, alpha=0.03)
    axs[i].plot(rugged_seq, m5_μ_mean, lw=1.5, color='k')
    axs[i].set_title(f'{ttl} nations')
    axs[i].set_xlabel('ruggedness (standardized)')
    axs[i].set_ylabel('log GDP (as prop. of mean)')
    #axs[i].fill_between(rugged_seq.flatten(), y1=m5_ci[:, 0], y2=m5_ci[:, 1], alpha=0.5)
    az.plot_hpd(rugged_seq, m5_μ, credible_interval=0.97, ax=axs[i], smooth=True)

rugged_seq.shape
df_m5_A0 = pm.trace_to_dataframe(trace_m5).filter(regex='0', axis=1)
m5_μ = link(df_m5_A0, x)
pm.hpd(m5_μ, alpha=0.03).shape
