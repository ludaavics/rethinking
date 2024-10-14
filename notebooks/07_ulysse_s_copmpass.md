---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Chapter 7: Ulysse's Compass

```python
%load_ext jupyter_black

import arviz as az
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpyro
import numpyro.distributions as dist
import pandas as pd
import statsmodels.api as sm

from jax.scipy.special import logsumexp
from numpyro.infer import SVI
from numpyro.infer.autoguide import AutoLaplaceApproximation

seed = 84735
jrng = jax.random.key(seed)
plt.rcParams["figure.figsize"] = [10, 6]

optim = numpyro.optim.Adam(step_size=1)
loss = numpyro.infer.Trace_ELBO()
```

```python
def prune_return_sites(posterior_samples, depth=1, exclude=[]):
    return {
        k: v
        for k, v in posterior_samples.items()
        if len(posterior_samples[k].shape) <= depth and k not in exclude
    }


def scale(x):
    return (x - x.mean()) / x.std()
```

## Code


### Code 7.1

```python
df = pd.DataFrame(
    {
        "species": [
            "afarensis",
            "africanus",
            "habilis",
            "boisei",
            "rudolfensis",
            "ergaster",
            "sapiens",
        ],
        "brain": [438, 452, 612, 521, 752, 871, 1350],
        "mass": [37.0, 35.5, 34.5, 41.5, 55.5, 61.0, 53.5],
    }
)
df
```

### Code 7.2

```python
df["mass_std"] = scale(df["mass"])
df["brain_std"] = df["brain"] / df["brain"].max()
```

### Code 7.3

```python
def model_7_1(mass, brain):
    a = numpyro.sample("a", dist.Normal(0.5, 1))
    b = numpyro.sample("b", dist.Normal(0, 10))
    mu = numpyro.deterministic("mu", a + b * mass)
    sigma = numpyro.sample("sigma", dist.LogNormal(0, 1))
    brain = numpyro.sample("brain", dist.Normal(mu, sigma), obs=brain)
    return brain


guide_7_1 = AutoLaplaceApproximation(model_7_1)
svi_7_1 = SVI(model=model_7_1, guide=guide_7_1, optim=optim, loss=loss).run(
    jrng,
    mass=df["mass_std"].values,
    brain=df["brain_std"].values,
    num_steps=2_500,
)
posterior_samples_7_1 = guide_7_1.sample_posterior(
    jrng, svi_7_1.params, sample_shape=(10_500,)
)
numpyro.diagnostics.print_summary(
    prune_return_sites(posterior_samples_7_1), prob=0.89, group_by_chain=False
)
```

### Code 7.4

```python
model_7_1_OLS = sm.OLS(df["brain"], df[["mass_std"]]).fit()
# posterior_samples_7_1_OLS = dist.Normal(model_7_1_OLS.params['brain'],
```

### Code 7.5

```python
residuals_7_1 = df["brain_std"] - posterior_samples_7_1["mu"].mean(axis=0)
df["predictions_7_1"] = posterior_samples_7_1["mu"].mean(axis=0)
df["residuals_7_1"] = residuals_7_1
r2 = 1 - df["residuals_7_1"].var() / df["brain_std"].var()
display(df)
print(f"R2: {r2}")
```

### Code 7.6

```python
def r2_is_bad(posterior_samples):
    residuals = df["brain_std"] - posterior_samples["mu"].mean(axis=0)
    return 1 - residuals.var() / df["brain_std"].var()


r2_is_bad(posterior_samples_7_1)
```

### Code 7.7

```python
def model_7_2(mass, brain):
    a = numpyro.sample("a", dist.Normal(0.5, 1))
    b = numpyro.sample("b", dist.Normal(0, 10).expand((2,)))
    mu = numpyro.deterministic("mu", a + jnp.dot(b, jnp.array([mass, mass**2])))
    sigma = numpyro.sample("sigma", dist.LogNormal(0, 1))
    brain = numpyro.sample("brain", dist.Normal(mu, sigma), obs=brain)
    return brain


guide_7_2 = AutoLaplaceApproximation(model_7_2)
svi_7_2 = SVI(model=model_7_2, guide=guide_7_2, optim=optim, loss=loss).run(
    jrng,
    mass=df["mass_std"].values,
    brain=df["brain_std"].values,
    num_steps=10_000,
)
posterior_samples_7_2 = guide_7_2.sample_posterior(
    jrng, svi_7_2.params, sample_shape=(5000,)
)
numpyro.diagnostics.print_summary(
    prune_return_sites(posterior_samples_7_2, depth=2, exclude=["mu"]),
    prob=0.89,
    group_by_chain=False,
)
```

### Code 7.8

```python
def model_factory(order):
    def model(mass, brain):
        a = numpyro.sample("a", dist.Normal(0.5, 1))
        b = numpyro.sample("b", dist.Normal(0, 10).expand((order,)))
        mu = numpyro.deterministic(
            "mu",
            a
            + jnp.dot(
                b, jnp.array([jnp.power(mass, _order + 1) for _order in range(order)])
            ),
        )
        sigma = (
            numpyro.sample("sigma", dist.LogNormal(0, 1))
            if order < brain.shape[0]
            else 0.001
        )
        brain = numpyro.sample("brain", dist.Normal(mu, sigma), obs=brain)
        return brain

    guide = AutoLaplaceApproximation(model)
    svi = SVI(
        model=model, guide=guide, optim=numpyro.optim.Adam(step_size=0.1), loss=loss
    ).run(
        jrng,
        mass=df["mass_std"].values,
        brain=df["brain_std"].values,
        num_steps=5_000,
    )
    posterior_samples = guide.sample_posterior(jrng, svi.params, sample_shape=(5000,))
    numpyro.diagnostics.print_summary(
        prune_return_sites(posterior_samples, depth=2, exclude=["mu"]),
        prob=0.89,
        group_by_chain=False,
    )
    return {
        "model": model,
        "guide": guide,
        "svi": svi,
        "posterior_samples": posterior_samples,
    }


_7_3 = model_factory(order=3)
_7_4 = model_factory(order=4)
_7_5 = model_factory(order=5)
```

### Code 7.9

```python
_7_6 = model_factory(order=6)
```

### Code 7.10

```python
mass_std = jnp.linspace(df["mass_std"].min(), df["mass_std"].max(), num=100)
posterior_predictive_7_1 = numpyro.infer.Predictive(model_7_1, posterior_samples_7_1)
posterior_predictive_samples_7_1 = posterior_predictive_7_1(
    jrng, mass=mass_std, brain=None
)
mu_mean = posterior_predictive_samples_7_1["mu"].mean(axis=0)
mu_ci = numpyro.diagnostics.hpdi(posterior_predictive_samples_7_1["mu"])
plt.plot(mass_std, mu_mean)
plt.fill_between(mass_std, mu_ci[0], mu_ci[1], color="k", alpha=0.5)
```

### Code 7.11

```python
i = 1
df_minus_i = df.drop(df.index[i])
```

### Code 7.12

```python
p = jnp.array([0.3, 0.7])
-jnp.sum(p * jnp.log(p))
```

### Code 7.13

```python
def lppd(jrng, guide, params, inputs, return_site, num_samples=1000):
    posterior_samples = guide.sample_posterior(
        jrng, params, sample_shape=(num_samples,)
    )
    logprob = numpyro.infer.log_likelihood(
        guide.model,
        posterior_samples,
        **inputs,
    )[return_site]
    return logsumexp(logprob, 0) - jnp.log(logprob.shape[0])


_, jrng = jax.random.split(jrng)
lppd(
    jrng,
    guide=guide_7_1,
    params=svi_7_1.params,
    inputs={"mass": df["mass_std"].values, "brain": df["brain_std"].values},
    return_site="brain",
    num_samples=10_000,
)
```

### Code 7.14

```python
def lppd(jrng, guide, params, inputs, return_site, num_samples=1000):
    posterior_samples = guide.sample_posterior(
        jrng, params, sample_shape=(num_samples,)
    )
    logprob = numpyro.infer.log_likelihood(
        guide.model,
        posterior_samples,
        **inputs,
    )[return_site]
    return logsumexp(logprob, 0) - jnp.log(logprob.shape[0])
```

### Code 7.15

```python
lppds = []
for order in range(1, 7):
    _model = model_factory(order=order)
    # _, jrng = jax.random.split(jrng)
    _lppd = lppd(
        jrng,
        guide=_model["guide"],
        params=_model["svi"].params,
        inputs={"mass": df["mass_std"].values, "brain": df["brain_std"].values},
        return_site="brain",
        num_samples=10_000,
    )
    lppds.append(float(jnp.sum(_lppd)))
lppds
```

### Code 7.16

Can't be bothered

### Code 7.17

Can't be bothered

### Code 7.18

Can't be bothered


### Code 7.19

```python
df = pd.read_csv("../data/cars.csv", sep=",", index_col=[0]).rename(
    columns={"dist": "distance"}
)
df.head()
```

```python
def model_waic(speed, distance):
    a = numpyro.sample("a", dist.Normal(0, 100))
    b = numpyro.sample("b", dist.Normal(0, 10))
    mu = numpyro.deterministic("mu", a + b * speed)
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    distance = numpyro.sample("distance", dist.Normal(mu, sigma), obs=distance)
    return distance


guide_waic = AutoLaplaceApproximation(model_waic)
svi_waic = SVI(model=model_waic, guide=guide_waic, optim=optim, loss=loss).run(
    jrng,
    speed=df["speed"].values,
    distance=df["distance"].values,
    num_steps=5_000,
)
posterior_samples_waic = guide_waic.sample_posterior(
    jrng, svi_waic.params, sample_shape=(1_000,)
)
numpyro.diagnostics.print_summary(
    prune_return_sites(posterior_samples_waic),
    prob=0.89,
    group_by_chain=False,
)
```

### Code 7.20

```python
logprob = (
    dist.Normal(posterior_samples_waic["mu"], posterior_samples_waic["sigma"][:, None])
    .log_prob(df["distance"].values)
    .T
)
pd.DataFrame(logprob).head()
```

### Code 7.21

```python
lppd = logsumexp(logprob, axis=1) - jnp.log(posterior_samples_waic["mu"].shape[0])
lppd
```

### Code 7.22

```python
pWAIC = jnp.var(logprob, axis=1)
pWAIC
```

### Code 7.23

```python
-2 * (jnp.sum(lppd) - jnp.sum(pWAIC))
```

### Code 7.24

```python
waic_vec = -2 * (lppd - pWAIC)
jnp.sqrt(df.shape[0] * jnp.var(waic_vec))
```

### Code 7.25

```python
# number of plants
N = 100

# simulate initial heights
h0 = dist.Normal(10, 2).sample(jrng, sample_shape=(N,))

# assign treatmens and simulate fungus and growth
_, jrng = jax.random.split(jrng)
treatment = dist.Categorical(probs=jnp.array([0.5, 0.5])).sample(
    jrng, sample_shape=(N,)
)
_, jrng = jax.random.split(jrng)
fungus = dist.Binomial(total_count=1, probs=0.5 - treatment * 0.4).sample(jrng)
h1 = h0 + dist.Normal(5 - 3 * fungus).sample(jrng)
df = pd.DataFrame({"h0": h0, "h1": h1, "treatment": treatment, "fungus": fungus})
display(df.head())
df.describe()


def model_6_6(h0, h1):
    p = numpyro.sample("p", dist.LogNormal(0, 0.25))
    mu = numpyro.deterministic("mu", h0 * p)
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    h1 = numpyro.sample("h1", dist.Normal(mu, sigma), obs=h1)
    return h1


_, jrng = jax.random.split(jrng)
guide_6_6 = AutoLaplaceApproximation(model_6_6)
svi_6_6 = SVI(
    model=model_6_6,
    guide=guide_6_6,
    optim=optim,
    loss=loss,
    h0=df["h0"].values,
    h1=df["h1"].values,
).run(jrng, 2_500)
posterior_samples_6_6 = guide_6_6.sample_posterior(
    jrng, svi_6_6.params, sample_shape=(1_000,)
)
numpyro.diagnostics.print_summary(
    prune_return_sites(posterior_samples_6_6),
    prob=0.89,
    group_by_chain=False,
)


def model_6_7(h0, treatment, fungus, h1):
    alpha = numpyro.sample("alpha", dist.LogNormal(0, 0.25))
    beta_treatment = numpyro.sample("beta_treatment", dist.Normal(0, 0.5))
    beta_fungus = numpyro.sample("beta_fungus", dist.Normal(0, 0.5))
    p = numpyro.deterministic(
        "p", alpha + beta_treatment * treatment + beta_fungus * fungus
    )
    mu = numpyro.deterministic("mu", h0 * p)
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    h1 = numpyro.sample("h1", dist.Normal(mu, sigma), obs=h1)
    return h1


_, jrng = jax.random.split(jrng)
guide_6_7 = AutoLaplaceApproximation(model_6_7)
svi_6_7 = SVI(
    model=model_6_7,
    guide=guide_6_7,
    optim=optim,
    loss=loss,
    h0=df["h0"].values,
    treatment=df["treatment"].values,
    fungus=df["fungus"].values,
    h1=df["h1"].values,
).run(jrng, 2_500)
posterior_samples_6_7 = guide_6_7.sample_posterior(
    jrng, svi_6_7.params, sample_shape=(1_000,)
)
numpyro.diagnostics.print_summary(
    prune_return_sites(posterior_samples_6_7),
    prob=0.89,
    group_by_chain=False,
)


def model_6_8(h0, treatment, h1):
    alpha = numpyro.sample("alpha", dist.LogNormal(0, 0.25))
    beta_treatment = numpyro.sample("beta_treatment", dist.Normal(0, 0.5))
    p = numpyro.deterministic("p", alpha + beta_treatment * treatment)
    mu = numpyro.deterministic("mu", h0 * p)
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    h1 = numpyro.sample("h1", dist.Normal(mu, sigma), obs=h1)
    return h1


_, jrng = jax.random.split(jrng)
guide_6_8 = AutoLaplaceApproximation(model_6_8)
svi_6_8 = SVI(
    model=model_6_8,
    guide=guide_6_8,
    optim=optim,
    loss=loss,
    h0=df["h0"].values,
    treatment=df["treatment"].values,
    h1=df["h1"].values,
).run(jrng, 2_500)
posterior_samples_6_8 = guide_6_8.sample_posterior(
    jrng, svi_6_8.params, sample_shape=(1_000,)
)
numpyro.diagnostics.print_summary(
    prune_return_sites(posterior_samples_6_8),
    prob=0.89,
    group_by_chain=False,
)
```

```python
logprob_6_7 = numpyro.infer.log_likelihood(
    model_6_7,
    posterior_samples_6_7,
    treatment=df["treatment"].values,
    fungus=df["fungus"].values,
    h0=df["h0"].values,
    h1=df["h1"].values,
)
az_6_7 = az.from_dict(sample_stats={"log_likelihood": logprob_6_7["h1"][None, ...]})
az.waic(az_6_7, scale="deviance")
```

### Code 7.26

```python
logprob_6_6 = numpyro.infer.log_likelihood(
    model_6_6,
    posterior_samples_6_6,
    h0=df["h0"].values,
    h1=df["h1"].values,
)
az_6_6 = az.from_dict(sample_stats={"log_likelihood": logprob_6_6["h1"][None, ...]})
logprob_6_8 = numpyro.infer.log_likelihood(
    model_6_8,
    posterior_samples_6_8,
    treatment=df["treatment"].values,
    h0=df["h0"].values,
    h1=df["h1"].values,
)
az_6_8 = az.from_dict(sample_stats={"log_likelihood": logprob_6_8["h1"][None, ...]})
az.compare(
    {"model_6_6": az_6_6, "model_6_7": az_6_7, "model_6_8": az_6_8},
    ic="waic",
    scale="deviance",
)
```

## Code 7.27

```python
waic_6_7 = az.waic(az_6_7, scale="deviance", pointwise=True)
waic_6_8 = az.waic(az_6_8, scale="deviance", pointwise=True)
waic_6_7_minus_6_8 = waic_6_7["waic_i"].values - waic_6_8["waic_i"].values
num_posterior_samples = waic_6_7["waic_i"].shape[0]
jnp.sqrt(num_posterior_samples * jnp.var(waic_6_7_minus_6_8))
```

### Code 7.28

```python
55 + jnp.array([-1, 1]) * 11 * 2.6
```

### Code 7.29

```python
compare_7_1 = az.compare(
    {"model_6_6": az_6_6, "model_6_7": az_6_7, "model_6_8": az_6_8},
    ic="waic",
    scale="deviance",
)
az.plot_compare(compare_7_1)
```

### Code 7.30

```python
waic_6_6 = az.waic(az_6_6, scale="deviance", pointwise=True)
waic_6_6_minus_6_8 = waic_6_6["waic_i"].values - waic_6_8["waic_i"].values
num_posterior_samples = waic_6_6["waic_i"].shape[0]
jnp.sqrt(num_posterior_samples * jnp.var(waic_6_6_minus_6_8))
```

### Code 7.31

```python
waics = [waic_6_6, waic_6_7, waic_6_8]
dse = jnp.full((3, 3), jnp.nan)
for i in range(len(waics)):
    for j in range(len(waics)):
        diff = waics[i]["waic_i"].values - waics[j]["waic_i"].values
        dse = dse.at[i, j].set(jnp.sqrt(num_posterior_samples * jnp.var(diff)))
pd.DataFrame(
    dse,
    index=["model_6_6", "model_6_7", "model_6_8"],
    columns=["model_6_6", "model_6_7", "model_6_8"],
)
```

### Code 7.32

```python
df = pd.read_csv("../data/WaffleDivorce.csv", sep=";")
df["A"] = scale(df["MedianAgeMarriage"])
df["D"] = scale(df["Divorce"])
df["M"] = scale(df["Marriage"])


def model_5_1(
    age,
    *,
    alpha_prior={"loc": 0, "scale": 0.2},
    beta_age_prior={"loc": 0, "scale": 0.5},
    sigma_prior={"rate": 1},
    divorce=None,
):
    alpha = numpyro.sample("alpha", dist.Normal(**alpha_prior))
    beta_age = numpyro.sample("beta_age", dist.Normal(**beta_age_prior))
    mu = numpyro.deterministic("mu", alpha + beta_age * age)
    sigma = numpyro.sample("sigma", dist.Exponential(**sigma_prior))
    divorce = numpyro.sample("divorce", dist.Normal(mu, sigma), obs=divorce)
    return divorce


guide_5_1 = AutoLaplaceApproximation(model_5_1)
svi_5_1 = numpyro.infer.SVI(
    model=model_5_1,
    guide=guide_5_1,
    optim=optim,
    loss=loss,
    age=df["A"].values,
    divorce=df["D"].values,
).run(jrng, 2_000)
posterior_samples_5_1 = guide_5_1.sample_posterior(
    jrng, svi_5_1.params, sample_shape=(1_000,)
)
numpyro.diagnostics.print_summary(
    prune_return_sites(posterior_samples_5_1),
    prob=0.89,
    group_by_chain=False,
)
logprob_5_1 = numpyro.infer.log_likelihood(
    model_5_1,
    posterior_samples_5_1,
    age=df["A"].values,
    divorce=df["D"].values,
)
az_5_1 = az.from_dict(
    posterior={k: v[None, ...] for k, v in posterior_samples_5_1.items()},
    log_likelihood={"divorce": logprob_5_1["divorce"][None, ...]},
)


def model_5_2(
    marriage,
    *,
    alpha_prior={"loc": 0, "scale": 0.2},
    beta_marriage_prior={"loc": 0, "scale": 0.5},
    sigma_prior={"rate": 1},
    divorce=None,
):
    alpha = numpyro.sample("alpha", dist.Normal(**alpha_prior))
    beta_marriage = numpyro.sample("beta_marriage", dist.Normal(**beta_marriage_prior))
    mu = numpyro.deterministic("mu", alpha + beta_marriage * marriage)
    sigma = numpyro.sample("sigma", dist.Exponential(**sigma_prior))
    divorce = numpyro.sample("divorce", dist.Normal(mu, sigma), obs=divorce)
    return divorce


guide_5_2 = AutoLaplaceApproximation(model_5_2)
svi_5_2 = numpyro.infer.SVI(
    model=model_5_2,
    guide=guide_5_2,
    optim=optim,
    loss=loss,
    marriage=df["M"].values,
    divorce=df["D"].values,
).run(jrng, 2_000)
posterior_samples_5_2 = guide_5_2.sample_posterior(
    jrng, svi_5_2.params, sample_shape=(1_000,)
)
numpyro.diagnostics.print_summary(
    prune_return_sites(posterior_samples_5_2),
    prob=0.89,
    group_by_chain=False,
)
logprob_5_2 = numpyro.infer.log_likelihood(
    model_5_2,
    posterior_samples_5_2,
    marriage=df["M"].values,
    divorce=df["D"].values,
)
az_5_2 = az.from_dict(
    posterior={k: v[None, ...] for k, v in posterior_samples_5_2.items()},
    log_likelihood={"divorce": logprob_5_2["divorce"][None, ...]},
)


def model_5_3(
    age,
    marriage,
    *,
    alpha_prior={"loc": 0, "scale": 0.2},
    beta_age_prior={"loc": 0, "scale": 0.5},
    beta_marriage_prior={"loc": 0, "scale": 0.5},
    sigma_prior={"rate": 1},
    divorce=None,
):
    alpha = numpyro.sample("alpha", dist.Normal(**alpha_prior))
    beta_age = numpyro.sample("beta_age", dist.Normal(**beta_age_prior))
    beta_marriage = numpyro.sample("beta_marriage", dist.Normal(**beta_marriage_prior))
    mu = numpyro.deterministic("mu", alpha + beta_age * age + beta_marriage * marriage)
    sigma = numpyro.sample("sigma", dist.Exponential(**sigma_prior))
    divorce = numpyro.sample("divorce", dist.Normal(mu, sigma), obs=divorce)
    return divorce


guide_5_3 = AutoLaplaceApproximation(model_5_3)
svi_5_3 = numpyro.infer.SVI(
    model=model_5_3,
    guide=guide_5_3,
    optim=numpyro.optim.Adam(step_size=0.5),
    loss=numpyro.infer.Trace_ELBO(),
    age=df["A"].values,
    marriage=df["M"].values,
    divorce=df["D"].values,
).run(jrng, 2_000)
posterior_samples_5_3 = guide_5_3.sample_posterior(
    jrng, svi_5_3.params, sample_shape=(10_000,)
)
numpyro.diagnostics.print_summary(
    prune_return_sites(posterior_samples_5_3),
    prob=0.89,
    group_by_chain=False,
)
logprob_5_3 = numpyro.infer.log_likelihood(
    model_5_3,
    posterior_samples_5_3,
    age=df["A"].values,
    marriage=df["M"].values,
    divorce=df["D"].values,
)
az_5_3 = az.from_dict(
    posterior={k: v[None, ...] for k, v in posterior_samples_5_3.items()},
    log_likelihood={"divorce": logprob_5_3["divorce"][None, ...]},
)
```

### Code 7.33

```python
compare_7_2 = az.compare(
    {"model_5_1": az_5_1, "model_5_2": az_5_2, "model_5_3": az_5_3},
    ic="waic",
    scale="deviance",
)
display(compare_7_2)
az.plot_compare(compare_7_2)
```

### Code 7.34

```python
waic_5_3 = az.waic(az_5_3, pointwise=True, scale="deviance")
psis_5_3 = az.loo(az_5_3, pointwise=True, scale="deviance")
penalty_5_3 = az_5_3.log_likelihood.stack(sample=("chain", "draw")).var(dim="sample")
display(waic_5_3)
plt.plot(psis_5_3.pareto_k.values, penalty_5_3["divorce"].values, "o", mfc="none")
plt.xlabel("PSIS k value")
plt.ylabel("pWAIC")
```

### Code 7.35

```python
def model_5_3t(
    age,
    marriage,
    *,
    alpha_prior={"loc": 0, "scale": 0.2},
    beta_age_prior={"loc": 0, "scale": 0.5},
    beta_marriage_prior={"loc": 0, "scale": 0.5},
    sigma_prior={"rate": 1},
    divorce=None,
):
    alpha = numpyro.sample("alpha", dist.Normal(**alpha_prior))
    beta_age = numpyro.sample("beta_age", dist.Normal(**beta_age_prior))
    beta_marriage = numpyro.sample("beta_marriage", dist.Normal(**beta_marriage_prior))
    mu = numpyro.deterministic("mu", alpha + beta_age * age + beta_marriage * marriage)
    sigma = numpyro.sample("sigma", dist.Exponential(**sigma_prior))
    divorce = numpyro.sample(
        "divorce",
        dist.StudentT(
            2,
            mu,
            sigma,
        ),
        obs=divorce,
    )
    return divorce


guide_5_3t = AutoLaplaceApproximation(model_5_3t)
svi_5_3t = numpyro.infer.SVI(
    model=model_5_3t,
    guide=guide_5_3t,
    optim=optim,
    loss=loss,
    age=df["A"].values,
    marriage=df["M"].values,
    divorce=df["D"].values,
).run(jrng, 2_000)
posterior_samples_5_3t = guide_5_3t.sample_posterior(
    jrng, svi_5_3t.params, sample_shape=(10_000,)
)
numpyro.diagnostics.print_summary(
    prune_return_sites(posterior_samples_5_3t),
    prob=0.89,
    group_by_chain=False,
)
logprob_5_3t = numpyro.infer.log_likelihood(
    model_5_3t,
    posterior_samples_5_3t,
    age=df["A"].values,
    marriage=df["M"].values,
    divorce=df["D"].values,
)
az_5_3t = az.from_dict(
    posterior={k: v[None, ...] for k, v in posterior_samples_5_3t.items()},
    log_likelihood={"divorce": logprob_5_3t["divorce"][None, ...]},
)
az.loo(az_5_3t)
```

## Easy


### 7E1

The measure of uncertainty should:
 - be continuous
 - increase as the number of possibilities increase
 - additive: the uncertainty on two events should be the sum of the uncertainty of eache event


### 7E2

```python
p = jnp.array([0.7, 0.3])
-jnp.sum(p * jnp.log(p))
```

### 7E3

```python
p = jnp.array([0.2, 0.25, 0.25, 0.3])
-jnp.sum(p * jnp.log(p))
```

### 7E4

```python
p = jnp.array([1 / 3, 1 / 3, 1 / 3])
-jnp.sum(p * jnp.log(p))
```

## Medium


### 7M1

`AIC = -2 (log_score - p)` where `p` is the number of parameters
`WAIC = -2 (lppd - waic_penalty)` where `waic_penalty` is the variance of the log likelihood of the observation over the posterior distribution.

`AIC` assumes flat priors and that oos is drawn from same distribution as is.


### 7M2

Model selection picks a "best" mode, vs model comparison tries to see where models agree or disagree.
Model selection discard the uncertainty as to which is the right model.


### 7M3

Because the information criterion are a sum (rather than an average) of log likelihoods.


### 7M4

A stronger prior reduces the amount of freedoem in the training process, thus reducing the effective number of parameters.


### 7M5

Informative priors make the training process more skeptical of otherwise surprising data, reigning in tendancy of the model to draw to much from it.


### 7M6

Overly informative priors prevent the model from adjusting to contradictory data.


## Hard


### 7H1

```python
df = pd.read_csv("../data/Laffer.csv", sep=";")
df["rate"] = scale(df["tax_rate"])
df["revenue"] = scale(df["tax_revenue"])
rates = jnp.linspace(df["rate"].min(), df["rate"].max(), 30)


def model_7h1_linear(rate, revenue):
    a = numpyro.sample("a", dist.Normal(0, 0.2))
    b = numpyro.sample("b", dist.Normal(0, 0.5))
    mu = numpyro.deterministic("mu", a + b * rate)
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    revenue = numpyro.sample("revenue", dist.Normal(mu, sigma), obs=revenue)
    return revenue


guide_7h1_linear = AutoLaplaceApproximation(model_7h1_linear)
svi_7h1_linear = numpyro.infer.SVI(
    model=model_7h1_linear,
    guide=guide_7h1_linear,
    optim=optim,
    loss=loss,
    rate=df["rate"].values,
    revenue=df["revenue"].values,
).run(jrng, 2_000)
posterior_samples_7h1_linear = guide_7h1_linear.sample_posterior(
    jrng, svi_7h1_linear.params, sample_shape=(1_000,)
)
numpyro.diagnostics.print_summary(
    prune_return_sites(posterior_samples_7h1_linear),
    prob=0.89,
    group_by_chain=False,
)
posterior_predictive_7h1_linear = numpyro.infer.Predictive(
    model=model_7h1_linear, posterior_samples=posterior_samples_7h1_linear
)
posterior_predictive_samples_7h1_linear = posterior_predictive_7h1_linear(
    jrng,
    rate=rates,
    revenue=None,
)
mu_mean_7h1_linear = posterior_predictive_samples_7h1_linear["mu"].mean(axis=0)
logprob_7h1_linear = numpyro.infer.log_likelihood(
    model_7h1_linear,
    posterior_samples_7h1_linear,
    rate=df["rate"].values,
    revenue=df["revenue"].values,
)
idata_7h1_linear = az.from_dict(
    posterior={k: v[None, ...] for k, v in posterior_samples_7h1_linear.items()},
    log_likelihood={"revenue": logprob_7h1_linear["revenue"][None, ...]},
)


def model_7h1_curved(rate, revenue):
    a = numpyro.sample("a", dist.Normal(0, 0.2))
    b = numpyro.sample("b", dist.Normal(0, 0.5))
    b2 = numpyro.sample("b2", dist.Normal(0, 0.5))
    mu = numpyro.deterministic("mu", a + b * rate + b2 * rate**2)
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    revenue = numpyro.sample("revenue", dist.Normal(mu, sigma), obs=revenue)
    return revenue


guide_7h1_curved = AutoLaplaceApproximation(model_7h1_curved)
svi_7h1_curved = numpyro.infer.SVI(
    model=model_7h1_curved,
    guide=guide_7h1_curved,
    optim=optim,
    loss=loss,
    rate=df["rate"].values,
    revenue=df["revenue"].values,
).run(jrng, 2_000)
posterior_samples_7h1_curved = guide_7h1_curved.sample_posterior(
    jrng, svi_7h1_curved.params, sample_shape=(1_000,)
)
numpyro.diagnostics.print_summary(
    prune_return_sites(posterior_samples_7h1_curved),
    prob=0.89,
    group_by_chain=False,
)
posterior_predictive_7h1_curved = numpyro.infer.Predictive(
    model=model_7h1_curved, posterior_samples=posterior_samples_7h1_curved
)
posterior_predictive_samples_7h1_curved = posterior_predictive_7h1_curved(
    jrng,
    rate=rates,
    revenue=None,
)
mu_mean_7h1_curved = posterior_predictive_samples_7h1_curved["mu"].mean(axis=0)
logprob_7h1_curved = numpyro.infer.log_likelihood(
    model_7h1_curved,
    posterior_samples_7h1_curved,
    rate=df["rate"].values,
    revenue=df["revenue"].values,
)
idata_7h1_curved = az.from_dict(
    posterior={k: v[None, ...] for k, v in posterior_samples_7h1_curved.items()},
    log_likelihood={"divorce": logprob_7h1_curved["revenue"][None, ...]},
)
```

```python
df.plot(kind="scatter", x="rate", y="revenue")
plt.plot(rates, mu_mean_7h1_linear, "k")
plt.plot(rates, mu_mean_7h1_curved, "k--")
```

```python
compare_7h1 = az.compare(
    {"linear": idata_7h1_linear, "curved": idata_7h1_curved},
    ic="loo",
    scale="deviance",
)
display(compare_7h1)
az.plot_compare(compare_7h1)
```

Can't distinguish betwee linear and curved model.


### 7H2

```python
pd.DataFrame(az.loo(idata_7h1_linear, pointwise=True)["pareto_k"]).sort_values(
    by=0, ascending=False
).head()
```

```python
pd.DataFrame(az.loo(idata_7h1_curved, pointwise=True)["pareto_k"]).sort_values(
    by=0, ascending=False
).head()
```

The high tax revenue country (# 11) is an outlier in both models.

```python
df = pd.read_csv("../data/Laffer.csv", sep=";")
df["rate"] = scale(df["tax_rate"])
df["revenue"] = scale(df["tax_revenue"])
rates = jnp.linspace(df["rate"].min(), df["rate"].max(), 30)


def model_7h2_linear(rate, revenue):
    a = numpyro.sample("a", dist.Normal(0, 0.2))
    b = numpyro.sample("b", dist.Normal(0, 0.5))
    mu = numpyro.deterministic("mu", a + b * rate)
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    revenue = numpyro.sample("revenue", dist.StudentT(2, mu, sigma), obs=revenue)
    return revenue


guide_7h2_linear = AutoLaplaceApproximation(model_7h2_linear)
svi_7h2_linear = numpyro.infer.SVI(
    model=model_7h2_linear,
    guide=guide_7h2_linear,
    optim=optim,
    loss=loss,
    rate=df["rate"].values,
    revenue=df["revenue"].values,
).run(jrng, 2_000)
posterior_samples_7h2_linear = guide_7h2_linear.sample_posterior(
    jrng, svi_7h2_linear.params, sample_shape=(1_000,)
)
numpyro.diagnostics.print_summary(
    prune_return_sites(posterior_samples_7h2_linear),
    prob=0.89,
    group_by_chain=False,
)
posterior_predictive_7h2_linear = numpyro.infer.Predictive(
    model=model_7h2_linear, posterior_samples=posterior_samples_7h2_linear
)
posterior_predictive_samples_7h2_linear = posterior_predictive_7h2_linear(
    jrng,
    rate=rates,
    revenue=None,
)
mu_mean_7h2_linear = posterior_predictive_samples_7h2_linear["mu"].mean(axis=0)
logprob_7h2_linear = numpyro.infer.log_likelihood(
    model_7h2_linear,
    posterior_samples_7h2_linear,
    rate=df["rate"].values,
    revenue=df["revenue"].values,
)
idata_7h2_linear = az.from_dict(
    posterior={k: v[None, ...] for k, v in posterior_samples_7h2_linear.items()},
    log_likelihood={"revenue": logprob_7h2_linear["revenue"][None, ...]},
)


def model_7h2_curved(rate, revenue):
    a = numpyro.sample("a", dist.Normal(0, 0.2))
    b = numpyro.sample("b", dist.Normal(0, 0.5))
    b2 = numpyro.sample("b2", dist.Normal(0, 0.5))
    mu = numpyro.deterministic("mu", a + b * rate + b2 * rate**2)
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    revenue = numpyro.sample("revenue", dist.StudentT(2, mu, sigma), obs=revenue)
    return revenue


guide_7h2_curved = AutoLaplaceApproximation(model_7h2_curved)
svi_7h2_curved = numpyro.infer.SVI(
    model=model_7h2_curved,
    guide=guide_7h2_curved,
    optim=optim,
    loss=loss,
    rate=df["rate"].values,
    revenue=df["revenue"].values,
).run(jrng, 2_000)
posterior_samples_7h2_curved = guide_7h2_curved.sample_posterior(
    jrng, svi_7h2_curved.params, sample_shape=(1_000,)
)
numpyro.diagnostics.print_summary(
    prune_return_sites(posterior_samples_7h2_curved),
    prob=0.89,
    group_by_chain=False,
)
posterior_predictive_7h2_curved = numpyro.infer.Predictive(
    model=model_7h2_curved, posterior_samples=posterior_samples_7h2_curved
)
posterior_predictive_samples_7h2_curved = posterior_predictive_7h2_curved(
    jrng,
    rate=rates,
    revenue=None,
)
mu_mean_7h2_curved = posterior_predictive_samples_7h2_curved["mu"].mean(axis=0)
logprob_7h2_curved = numpyro.infer.log_likelihood(
    model_7h2_curved,
    posterior_samples_7h2_curved,
    rate=df["rate"].values,
    revenue=df["revenue"].values,
)
idata_7h2_curved = az.from_dict(
    posterior={k: v[None, ...] for k, v in posterior_samples_7h2_curved.items()},
    log_likelihood={"divorce": logprob_7h1_curved["revenue"][None, ...]},
)
```

```python
df.plot(kind="scatter", x="rate", y="revenue")
plt.plot(rates, mu_mean_7h1_linear, "k", alpha=0.2)
plt.plot(rates, mu_mean_7h1_curved, "k--", alpha=0.2)
plt.plot(rates, mu_mean_7h2_linear, "k")
plt.plot(rates, mu_mean_7h2_curved, "k--")
```

```python
compare_7h2 = az.compare(
    {
        "linear": idata_7h1_linear,
        "curved": idata_7h1_curved,
        "linear_t": idata_7h2_linear,
        "curved_t": idata_7h2_curved,
    },
    ic="loo",
    scale="deviance",
)
display(compare_7h2)
az.plot_compare(compare_7h2)
```

### 7H3

```python
df = pd.DataFrame(
    [
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.8, 0.1, 0.05, 0.025, 0.025],
        [0.05, 0.15, 0.7, 0.05, 0.05],
    ],
    index=["Island 1", "Island 2", "Island 3"],
    columns=["Species A", "Species B", "Species C", "Species D", "Species D"],
)
df
```

```python
-jnp.sum(df.values * jnp.log(df.values), axis=1)
```

The first  (second) island has the most (least) uncertainty as to the distribution of species.

```python
kl_divergences = jnp.full(df.shape[:1] * 2, jnp.nan)
for i in range(df.shape[0]):
    for j in range(df.shape[0]):
        p = df.iloc[i, :].values
        q = df.iloc[j, :].values
        kl_divergences = kl_divergences.at[i, j].set(jnp.sum(p * jnp.log(p / q)))
kl_divergences = pd.DataFrame(kl_divergences, index=df.index, columns=df.index)
kl_divergences
```

Island 1 predicts the other best because it has the higher entropy. So it is the least surprised when it's contradicted.


### 7H3

```python
def sim_happiness(seed=1977, N_years=1000, max_age=65, N_births=20, aom=18):
    # age existing individuals & newborns
    A = jnp.repeat(jnp.arange(1, N_years + 1), N_births)
    # sim happiness trait - never changes
    H = jnp.repeat(jnp.linspace(-2, 2, N_births)[None, :], N_years, 0).reshape(-1)
    # not yet married
    M = jnp.zeros(N_years * N_births, dtype=jnp.int32)

    def update_M(i, M):
        # for each person over 17, chance get married
        married = dist.Bernoulli(logits=(H - 4)).sample(jax.random.PRNGKey(seed + i))
        return jnp.where((A >= i) & (M == 0), married, M)

    M = jax.lax.fori_loop(aom, max_age + 1, update_M, M)
    # mortality
    deaths = A > max_age
    A = A[~deaths]
    H = H[~deaths]
    M = M[~deaths]

    d = pd.DataFrame({"age": A, "married": M, "happiness": H})
    return d


df = sim_happiness(seed=1977, N_years=1000)
df2 = df.loc[df["age"] > 17, :].copy()
df2["A"] = (df2["age"] - 18) / (65 - 18)


def model_6_9(age, married, happiness):
    with numpyro.plate(name="married", size=2):
        alpha = numpyro.sample("alpha", dist.Normal(0, 1))
    beta_age = numpyro.sample("beta_age", dist.Normal(0, 2))
    mu = numpyro.deterministic("mu", alpha[married] + beta_age * age)
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    happiness = numpyro.sample("happiness", dist.Normal(mu, sigma), obs=happiness)
    return happiness


guide_6_9 = AutoLaplaceApproximation(model_6_9)
svi_6_9 = SVI(
    model=model_6_9,
    guide=guide_6_9,
    optim=optim,
    loss=loss,
    age=df2["A"].values,
    married=df2["married"].values,
    happiness=df2["happiness"].values,
).run(jrng, 5_000)
posterior_samples_6_9 = guide_6_9.sample_posterior(
    jrng, svi_6_9.params, sample_shape=(2_500,)
)
numpyro.diagnostics.print_summary(
    prune_return_sites(posterior_samples_6_9, depth=2, exclude=["mu"]),
    prob=0.89,
    group_by_chain=False,
)
log_likelihood_6_9 = numpyro.infer.log_likelihood(
    model_6_9,
    posterior_samples_6_9,
    age=df2["A"].values,
    married=df2["married"].values,
    happiness=df2["happiness"].values,
)
idata_6_9 = az.from_dict(
    posterior={k: v[None, ...] for k, v in posterior_samples_6_9.items()},
    log_likelihood={"happiness": log_likelihood_6_9["happiness"][None, ...]},
)


def model_6_10(age, happiness):
    alpha = numpyro.sample("alpha", dist.Normal(0, 1))
    beta_age = numpyro.sample("beta_age", dist.Normal(0, 2))
    mu = numpyro.deterministic("mu", alpha + beta_age * age)
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    happiness = numpyro.sample("happiness", dist.Normal(mu, sigma), obs=happiness)
    return happiness


guide_6_10 = AutoLaplaceApproximation(model_6_10)
svi_6_10 = SVI(
    model=model_6_10,
    guide=guide_6_10,
    optim=optim,
    loss=loss,
    age=df2["A"].values,
    happiness=df2["happiness"].values,
).run(jrng, 5_000)
posterior_samples_6_10 = guide_6_10.sample_posterior(
    jrng, svi_6_10.params, sample_shape=(2_500,)
)
numpyro.diagnostics.print_summary(
    prune_return_sites(posterior_samples_6_10),
    prob=0.89,
    group_by_chain=False,
)
log_likelihood_6_10 = numpyro.infer.log_likelihood(
    model_6_10,
    posterior_samples_6_10,
    age=df2["A"].values,
    happiness=df2["happiness"].values,
)
idata_6_10 = az.from_dict(
    posterior={k: v[None, ...] for k, v in posterior_samples_6_10.items()},
    log_likelihood={"happiness": log_likelihood_6_10["happiness"][None, ...]},
)
```

```python
compare_7h3 = az.compare(
    {
        "model_6_9": idata_6_9,
        "model_6_10": idata_6_10,
    },
    ic="loo",
    scale="deviance",
)
display(compare_7h3)
az.plot_compare(compare_7h3)
```

The confounded model is much better at predicting. Even though marriage doesn't cause happiness, being married is statistically associated with being happy.


### 7H5

```python
df = pd.read_csv("../data/foxes.csv", sep=";")
df["F"] = scale(df["avgfood"])
df["G"] = scale(df["groupsize"])
df["A"] = scale(df["area"])
df["W"] = scale(df["weight"])

Xs = [df[["F", "G", "A"]], df[["F", "G"]], df[["G", "A"]], df[["F"]], df[["A"]]]
idata_7h5 = []
for X in Xs:

    def model_7h5(X, y):
        a = numpyro.sample("a", dist.Normal(0, 0.2))
        b = numpyro.sample("b", dist.Normal(0, 0.5).expand(X.shape[1:2]))
        mu = numpyro.deterministic("mu", a + jnp.dot(b, X.T))
        sigma = numpyro.sample("sigma", dist.Exponential(1))
        y = numpyro.sample("y", dist.Normal(mu, sigma), obs=y)
        return y

    guide_7h5 = AutoLaplaceApproximation(model_7h5)
    svi_7h5 = SVI(
        model=model_7h5,
        guide=guide_7h5,
        optim=optim,
        loss=loss,
        X=X.values,
        y=df["W"].values,
    ).run(jrng, 2_500)
    posterior_samples_7h5 = guide_7h5.sample_posterior(
        jrng, svi_7h5.params, sample_shape=(2_500,)
    )
    numpyro.diagnostics.print_summary(
        prune_return_sites(posterior_samples_7h5, depth=2, exclude=["mu"]),
        prob=0.89,
        group_by_chain=False,
    )

    log_likelihood_7h5 = numpyro.infer.log_likelihood(
        model_7h5,
        posterior_samples_7h5,
        X=X.values,
        y=df["W"].values,
    )
    idata_7h5.append(
        az.from_dict(
            posterior={k: v[None, ...] for k, v in posterior_samples_7h5.items()},
            log_likelihood={"y": log_likelihood_7h5["y"][None, ...]},
        )
    )
```

```python
compare_7h5 = az.compare(
    dict(zip(["".join(X.columns) for X in Xs], idata_7h5)),
    ic="loo",
    scale="deviance",
)
display(compare_7h5)
az.plot_compare(compare_7h5)
```

Models using just food or just area have the weakest predictive power since they're confounded (negatively correlated and each correlated to the outcome)
The remaining models are hard to distinguish from one another. 




