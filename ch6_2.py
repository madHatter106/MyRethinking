"""
POST-TREATMENT BIAS - This occurs when variables that are consequences of
other variables are including in an analysis.

E.g.: Plants grown in a greenhouse subject to growth-inhibiting fungi.
Goal is to measure effects of anti fungal treatment on plant height.
1. Initial height, h0, measured
2. Treatment is applied
3. Measure final height, h1, and presence of fungus.

There are four variables: h0, h1, whether treatment was applied,
fungus presence. h1 is the outcome of interest. What else to include?
"""

import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as ar
from sklearn.preprocessing import scale
import matplotlib.pyplot as pl
from causalgraphicalmodels import CausalGraphicalModel

np.random.seed(71)
n_plants = 100

# simulate initial height
h0 = np.random.normal(10, 2, size=n_plants)

# assign treatments and simulate fungus and growth
treatment = np.repeat([0, 1], n_plants/2)
fungus = np.random.binomial(n=1, p=0.5-treatment*0.4, size=n_plants)
h1 = h0 + np.random.normal(5 - 3 * fungus)
d = pd.DataFrame(dict(h0=h0, h1=h1, treatment=treatment, fungus=fungus))
d.head()
pd.plotting.scatter_matrix(d);
d.describe()

"""
Put the parameters on a scale of proportion of height at tme t=0.
h1,i ~ Normal(μi, σ)
μi = h0,i * p
p = h1,i / h0,i
p = 1 → the plant's height hasn't changed
p = 2 → the plant's height has doubled
1>p>0 → plant has degraded or died

so i'll use a lognormal prior for p
"""

with pm.Model() as m6:
    σ = pm.Exponential('σ', 1)
    p = pm.Lognormal('p', mu=0, sd=0.25)
    μ = d.h0.values * p
    h1 = pm.Normal('h1', mu=μ, sd=σ, observed=d.h1.values)
    trc6 = pm.sample(tune=1000)
pm.summary(trc6)
"""
The above suggests about 40% average growth. Now to include both treatment and
fungus variables with the intention of measuring the effect of both. They will
be part of the proportionality model like so:
"""

with pm.Model() as m7:
    σ = pm.Exponential('σ', 1)
    βT = pm.Normal('βT', 0, 0.5)
    βF = pm.Normal('βF', 0, 0.5)
    α = pm.Lognormal('α', 0, 0.25)
    p = α + βT * d.treatment.values + βF * d.fungus.values
    μi = d.h0.values * p
    h1 = pm.Normal('h1', mu=μi, sd=σ, observed=d.h1.values)
    trc7 = pm.sample(tune=1000)
pm.summary(trc7)
"""
Treatment appears to have negligible effect even though βF posterior indicates fungus impacts
growth.
The problem is that fungus is a consequence of treatment; i.e. fungus is a post-treatment variable.
The model asked the question "Once we know fungus is present does treatment matter?" ⇒ No.
The next model ignores the fungus variable
"""

with pm.Model() as m8:
    σ = pm.Exponential('σ', 1)
    α = pm.Lognormal('α', 0, 0.2)
    βT = pm.Normal('βT', 0, 0.5)
    p = α + βT * d.treatment.values
    μ = d.h0.values * p
    h1 = pm.Normal('h1', mu=μ, sd=σ, observed=d.h1.values)
    trc8 = pm.sample(tune=1000)

pm.summary(trc8)
"""
Now the treatment effect is plain to see. Note that:
1. It makes sense to control for pre-treatment differences such as initial height, h0, here.
2. Including post-treatment variables can mask the treatment itself.
3. Note that model m7 is still useful to identify the causal mechanism!
"""
plant_dag = CausalGraphicalModel(nodes=['H0', 'H1', 'T', 'F'],
                                 edges=[('H0', 'H1'), ('T', 'F'), ('F', 'H1')])
plant_dag.draw()
plant_dag.is_d_separated('T', 'H1')
plant_dag.is_d_separated('T', 'H1', 'F')
plant_dag.get_all_independence_relationships()
