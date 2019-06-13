import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
import pymc3 as pm
import arviz as az


da = pd.read_csv('./rethinking-Experimental/data/tulips.csv', delimiter=';')
da.head()



# Two models that predict blooms:
#   1. model with both water and shade but no interaction
#   2. model that also contains interaction of water and shade

# Center or max-scaling variables

da['blooms_scl'] = da.blooms / da.blooms.max()
da['water_cent'] = da.water - da.water.mean()
da['shade_cent'] = da.shade - da.shade.mean()

da.describe().loc[['min','max', 'mean']]

with pm.Model() as m1:
    α = pm.Normal('α', 0.5, .25)
    βw = pm.Normal('βw', 0, .25)
    βs = pm.Normal('βs', 0, .25)
    μi = α + βw * da.water_cent + βs * da.shade_cent
    σ = pm.Exponential('σ', 1)
    bloomsi = pm.Normal('blooms', μi, σ, observed=da.blooms_scl)

pm.model_to_graphviz(m1)

with pm.Model() as m2:
    α = pm.Normal('α', 0.5, .25)
    βw = pm.Normal('βw', 0, .25)
    βs = pm.Normal('βs', 0, .25)
    βws = pm.Normal('βws', 0, .25)
    σ = pm.Exponential('σ', 1)
    μi = α + βw * da.water_cent + βs * da.shade_cent + βws * da.water_cent * da.shade_cent
    bloomsi = pm.Normal('blooms', μi, σ, observed=da.blooms_scl)

    pm.model_to_graphviz(m2)
    
