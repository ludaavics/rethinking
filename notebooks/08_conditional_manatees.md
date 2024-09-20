---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.2
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Chapter 8: Conditional Manatees

```python
%load_ext jupyter_black

import arviz as az
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpyro
import pandas as pd

from numpyro import distributions as dist
from numpyro.infer import SVI
from numpyro.infer.autoguide import AutoLaplaceApproximation
from sklearn.preprocessing import LabelEncoder

seed = 84735
jrng = jax.random.key(seed)
plt.rcParams["figure.figsize"] = [10, 6]

optim = numpyro.optim.Adam(step_size=0.1)
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
### Code 8.1

```python
_df = pd.read_csv("../data/rugged.csv", sep=";")
_df["log_gdp"] = jnp.log(_df["rgdppc_2000"].values)
df = _df.dropna(subset=["rgdppc_2000"]).copy()
df["log_gdp_std"] = df["log_gdp"] / df["log_gdp"].mean()
df["rugged_std"] = df["rugged"] / df["rugged"].max()
df.head()
```

### Code 8.2

```python
rugged_bar = df["rugged_std"].mean()


def model_8_1(rugged, log_gdp):
    a = numpyro.sample("a", dist.Normal(1, 1))
    b = numpyro.sample("b", dist.Normal(0, 1))
    mu = numpyro.deterministic("mu", a + b * (rugged - rugged_bar))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    log_gdp = numpyro.sample("log_gdp", dist.Normal(mu, sigma), obs=log_gdp)
    return log_gdp
```

### Code 8.3

```python
ruggeds = jnp.linspace(-0.1, 1, 30)
prior_predictive_8_1 = numpyro.infer.Predictive(model=model_8_1, num_samples=1_000)
prior_predictive_samples_8_1 = prior_predictive_8_1(
    jrng,
    rugged=ruggeds,
    log_gdp=None,
)
plt.subplot(xlim=(0, 1), ylim=(0.5, 1.5), xlabel="ruggedness", ylabel="log GDP")
for mu_prior in prior_predictive_samples_8_1["mu"][:50]:
    plt.plot(ruggeds, mu_prior, color="k", alpha=0.3)
plt.axhline(df["log_gdp_std"].min(), color="b", ls="--")
plt.axhline(df["log_gdp_std"].max(), color="b", ls="--")
```

### Code 8.4

```python
jnp.sum(
    jnp.abs(prior_predictive_samples_8_1["b"]) > 0.6
) / prior_predictive_samples_8_1["b"].shape[0]
```

### Code 8.5

```python
def model_8_1(rugged, log_gdp):
    a = numpyro.sample("a", dist.Normal(1, 0.1))
    b = numpyro.sample("b", dist.Normal(0, 0.3))
    mu = numpyro.deterministic("mu", a + b * (rugged - rugged_bar))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    log_gdp = numpyro.sample("log_gdp", dist.Normal(mu, sigma), obs=log_gdp)
    return log_gdp


guide_8_1 = AutoLaplaceApproximation(model_8_1)
svi_8_1 = SVI(
    model=model_8_1,
    guide=guide_8_1,
    optim=optim,
    loss=loss,
    rugged=df["rugged_std"].values,
    log_gdp=df["log_gdp_std"].values,
).run(jrng, 2000)
posterior_samples_8_1 = guide_8_1.sample_posterior(
    jrng,
    svi_8_1.params,
    sample_shape=(1_000,),
)
log_likelihood_8_1 = numpyro.infer.log_likelihood(
    model=model_8_1,
    posterior_samples=posterior_samples_8_1,
    rugged=df["rugged_std"].values,
    log_gdp=df["log_gdp_std"].values,
)
idata_8_1 = az.from_dict(
    posterior={k: v[None] for k, v in posterior_samples_8_1.items()},
    log_likelihood={k: v[None] for k, v in log_likelihood_8_1.items()},
)
```

### Code 8.6

```python
numpyro.diagnostics.print_summary(
    prune_return_sites(posterior_samples_8_1),
    prob=0.89,
    group_by_chain=False,
)
```

### Code 8.7

```python
# make variable to index Africa (0) or not (1)
df["cid"] = jnp.logical_not(df["cont_africa"].values).astype(int)
```

### Code 8.8

```python
def model_8_2(rugged, cid, log_gdp):
    with numpyro.plate(name="continent", size=2):
        a = numpyro.sample("a", dist.Normal(1, 0.1))
    b = numpyro.sample("b", dist.Normal(0, 0.3))
    mu = numpyro.deterministic("mu", a[cid] + b * (rugged - rugged_bar))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    log_gdp = numpyro.sample("log_gdp", dist.Normal(mu, sigma), obs=log_gdp)
    return log_gdp


guide_8_2 = AutoLaplaceApproximation(model_8_2)
svi_8_2 = SVI(
    model=model_8_2,
    guide=guide_8_2,
    optim=optim,
    loss=loss,
    rugged=df["rugged_std"].values,
    cid=df["cid"].values,
    log_gdp=df["log_gdp_std"].values,
).run(jrng, 2000)
posterior_samples_8_2 = guide_8_2.sample_posterior(
    jrng,
    svi_8_2.params,
    sample_shape=(1_000,),
)
log_likelihood_8_2 = numpyro.infer.log_likelihood(
    model=model_8_2,
    posterior_samples=posterior_samples_8_2,
    rugged=df["rugged_std"].values,
    cid=df["cid"].values,
    log_gdp=df["log_gdp_std"].values,
)
idata_8_2 = az.from_dict(
    posterior={k: v[None] for k, v in posterior_samples_8_2.items()},
    log_likelihood={k: v[None] for k, v in log_likelihood_8_2.items()},
)
```

### Code 8.9

```python
compare_8_1 = az.compare(
    {"model_8_1": idata_8_1, "model_8_2": idata_8_2},
    ic="loo",
    scale="deviance",
)
az.plot_compare(compare_8_1)
```

### Code 8.10

```python
numpyro.diagnostics.print_summary(
    prune_return_sites(posterior_samples_8_2, depth=2, exclude=["mu"]),
    prob=0.89,
    group_by_chain=False,
)
```

### Code 8.11

```python
diff_a1_s2 = posterior_samples_8_2["a"][:, 0] - posterior_samples_8_2["a"][:, 1]
numpyro.diagnostics.hpdi(diff_a1_s2, prob=0.89)
```

### Code 8.12

```python
ruggeds = jnp.linspace(-0.1, 1, 30)
posterior_predictive_8_2 = numpyro.infer.Predictive(
    model=model_8_2, posterior_samples=posterior_samples_8_2
)
posterior_predictive_africa_samples_8_2 = posterior_predictive_8_2(
    jrng,
    rugged=ruggeds,
    cid=0,
    log_gdp=None,
)
mu_mean_africa_8_2 = posterior_predictive_africa_samples_8_2["mu"].mean(axis=0)
mu_hpdi_africa_8_2 = numpyro.diagnostics.hpdi(
    posterior_predictive_africa_samples_8_2["mu"], axis=0
)
posterior_predictive_not_africa_samples_8_2 = posterior_predictive_8_2(
    jrng,
    rugged=ruggeds,
    cid=1,
    log_gdp=None,
)
mu_mean_not_africa_8_2 = posterior_predictive_not_africa_samples_8_2["mu"].mean(axis=0)
mu_hpdi_not_africa_8_2 = numpyro.diagnostics.hpdi(
    posterior_predictive_not_africa_samples_8_2["mu"], axis=0
)
```

```python
plt.subplot(xlim=(0, 1), ylim=(0.5, 1.5), xlabel="ruggedness", ylabel="log GDP")
plt.scatter(
    df.loc[df["cid"] == 0, "rugged_std"],
    df.loc[df["cid"] == 0, "log_gdp_std"],
    color="purple",
    facecolors="none",
)
plt.plot(ruggeds, mu_mean_africa_8_2, color="purple", alpha=0.4)
plt.fill_between(
    ruggeds,
    mu_hpdi_africa_8_2[0, :],
    mu_hpdi_africa_8_2[1, :],
    color="purple",
    alpha=0.2,
)

plt.scatter(
    df.loc[df["cid"] == 1, "rugged_std"],
    df.loc[df["cid"] == 1, "log_gdp_std"],
    color="k",
    facecolors="none",
)
plt.plot(ruggeds, mu_mean_not_africa_8_2, color="k", alpha=0.4)
plt.fill_between(
    ruggeds,
    mu_hpdi_not_africa_8_2[0, :],
    mu_hpdi_not_africa_8_2[1, :],
    color="k",
    alpha=0.2,
)
```

### Code 8.13

```python
def model_8_3(rugged, cid, log_gdp):
    with numpyro.plate(name="continent", size=2):
        a = numpyro.sample("a", dist.Normal(1, 0.1))
        b = numpyro.sample("b", dist.Normal(0, 0.3))
    mu = numpyro.deterministic("mu", a[cid] + b[cid] * (rugged - rugged_bar))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    log_gdp = numpyro.sample("log_gdp", dist.Normal(mu, sigma), obs=log_gdp)
    return log_gdp


guide_8_3 = AutoLaplaceApproximation(model_8_3)
svi_8_3 = SVI(
    model=model_8_3,
    guide=guide_8_3,
    optim=optim,
    loss=loss,
    rugged=df["rugged_std"].values,
    cid=df["cid"].values,
    log_gdp=df["log_gdp_std"].values,
).run(jrng, 2000)
posterior_samples_8_3 = guide_8_3.sample_posterior(
    jrng,
    svi_8_3.params,
    sample_shape=(1_000,),
)
log_likelihood_8_3 = numpyro.infer.log_likelihood(
    model=model_8_3,
    posterior_samples=posterior_samples_8_3,
    rugged=df["rugged_std"].values,
    cid=df["cid"].values,
    log_gdp=df["log_gdp_std"].values,
)
idata_8_3 = az.from_dict(
    posterior={k: v[None] for k, v in posterior_samples_8_3.items()},
    log_likelihood={k: v[None] for k, v in log_likelihood_8_3.items()},
)
```

### Code 8.14

```python
numpyro.diagnostics.print_summary(
    prune_return_sites(posterior_samples_8_3, depth=2, exclude=["mu"]),
    prob=0.89,
    group_by_chain=False,
)
```

### Code 8.15

```python
compare_8_2 = az.compare(
    {
        "model_8_1": idata_8_1,
        "model_8_2": idata_8_2,
        "model_8_3": idata_8_3,
    },
    ic="loo",
    scale="deviance",
)
display(compare_8_2)
az.plot_compare(compare_8_2)
```

### Code 8.16

```python
pd.DataFrame(
    az.loo(idata_8_3, pointwise=True, scale="deviance")["pareto_k"],
    index=df["isocode"],
    columns=["pareto_k"],
).sort_values(["pareto_k"], ascending=False)
```

### Code 8.17

```python
ruggeds = jnp.linspace(-0.1, 1, 30)
posterior_predictive_8_3 = numpyro.infer.Predictive(
    model=model_8_3, posterior_samples=posterior_samples_8_3
)
posterior_predictive_africa_samples_8_3 = posterior_predictive_8_3(
    jrng,
    rugged=ruggeds,
    cid=0,
    log_gdp=None,
)
mu_mean_africa_8_3 = posterior_predictive_africa_samples_8_3["mu"].mean(axis=0)
mu_hpdi_africa_8_3 = numpyro.diagnostics.hpdi(
    posterior_predictive_africa_samples_8_3["mu"], axis=0
)
posterior_predictive_not_africa_samples_8_3 = posterior_predictive_8_3(
    jrng,
    rugged=ruggeds,
    cid=1,
    log_gdp=None,
)
mu_mean_not_africa_8_3 = posterior_predictive_not_africa_samples_8_3["mu"].mean(axis=0)
mu_hpdi_not_africa_8_3 = numpyro.diagnostics.hpdi(
    posterior_predictive_not_africa_samples_8_3["mu"], axis=0
)
```

```python
plt.subplot(xlim=(0, 1), ylim=(0.5, 1.5), xlabel="ruggedness", ylabel="log GDP")
plt.scatter(
    df.loc[df["cid"] == 0, "rugged_std"],
    df.loc[df["cid"] == 0, "log_gdp_std"],
    color="purple",
    facecolors="none",
)
plt.plot(ruggeds, mu_mean_africa_8_3, color="purple", alpha=0.4)
plt.fill_between(
    ruggeds,
    mu_hpdi_africa_8_3[0, :],
    mu_hpdi_africa_8_3[1, :],
    color="purple",
    alpha=0.2,
)

plt.scatter(
    df.loc[df["cid"] == 1, "rugged_std"],
    df.loc[df["cid"] == 1, "log_gdp_std"],
    color="k",
    facecolors="none",
)
plt.plot(ruggeds, mu_mean_not_africa_8_3, color="k", alpha=0.4)
plt.fill_between(
    ruggeds,
    mu_hpdi_not_africa_8_3[0, :],
    mu_hpdi_not_africa_8_3[1, :],
    color="k",
    alpha=0.2,
)
```

### Code 8.18

```python
delta = mu_mean_africa_8_3 - mu_mean_not_africa_8_3
delta
```

### Code 8.19

```python
df = pd.read_csv("../data/tulips.csv", sep=";")
df.head()
```

### Code 8.20

```python
df["blooms_std"] = df["blooms"] / df["blooms"].max()
df["water_cent"] = df["water"] - 2
df["shade_cent"] = df["shade"] - 2
```

### Code 8.21

```python
a = dist.Normal(0.5, 1).sample(jrng, sample_shape=(10_000,))
jnp.sum(jnp.logical_or(a < 0, a > 1)) / a.shape[0]
```

### Code 8.22

```python
a = dist.Normal(0.5, 0.25).sample(jrng, sample_shape=(10_000,))
jnp.sum(jnp.logical_or(a < 0, a > 1)) / a.shape[0]
```

### Code 8.23

```python
def model_8_4(water, shade, blooms):
    a = numpyro.sample("a", dist.Normal(0.5, 0.25))
    b_water = numpyro.sample("b_water", dist.Normal(0, 0.25))
    b_shade = numpyro.sample("b_shade", dist.Normal(0, 0.25))
    mu = numpyro.deterministic("mu", a + b_water * water + b_shade * shade)
    sigma = numpyro.sample("sigam", dist.Exponential(1))
    blooms = numpyro.sample("blooms", dist.Normal(mu, sigma), obs=blooms)
    return blooms


guide_8_4 = AutoLaplaceApproximation(model_8_4)
svi_8_4 = SVI(
    model=model_8_4,
    guide=guide_8_4,
    optim=optim,
    loss=loss,
    water=df["water_cent"].values,
    shade=df["shade_cent"].values,
    blooms=df["blooms_std"].values,
).run(jrng, 2000)
posterior_samples_8_4 = guide_8_4.sample_posterior(
    jrng,
    svi_8_4.params,
    sample_shape=(1_000,),
)
posterior_predictive_8_4 = numpyro.infer.Predictive(
    model=model_8_4, posterior_samples=posterior_samples_8_4
)
log_likelihood_8_4 = numpyro.infer.log_likelihood(
    model=model_8_4,
    posterior_samples=posterior_samples_8_4,
    water=df["water_cent"].values,
    shade=df["shade_cent"].values,
    blooms=df["blooms_std"].values,
)
idata_8_4 = az.from_dict(
    posterior={k: v[None] for k, v in posterior_samples_8_4.items()},
    log_likelihood={k: v[None] for k, v in log_likelihood_8_4.items()},
)
```

### Code 8.24

```python
def model_8_5(water, shade, blooms):
    a = numpyro.sample("a", dist.Normal(0.5, 0.25))
    b_water = numpyro.sample("b_water", dist.Normal(0, 0.25))
    b_shade = numpyro.sample("b_shade", dist.Normal(0, 0.25))
    b_water_shade = numpyro.sample("b_water_shade", dist.Normal(0, 0.25))
    mu = numpyro.deterministic(
        "mu",
        a + b_water * water + b_shade * shade + b_water_shade * water * shade,
    )
    sigma = numpyro.sample("sigam", dist.Exponential(1))
    blooms = numpyro.sample("blooms", dist.Normal(mu, sigma), obs=blooms)
    return blooms


guide_8_5 = AutoLaplaceApproximation(model_8_5)
svi_8_5 = SVI(
    model=model_8_5,
    guide=guide_8_5,
    optim=optim,
    loss=loss,
    water=df["water_cent"].values,
    shade=df["shade_cent"].values,
    blooms=df["blooms_std"].values,
).run(jrng, 2000)
posterior_samples_8_5 = guide_8_5.sample_posterior(
    jrng,
    svi_8_5.params,
    sample_shape=(1_000,),
)
posterior_predictive_8_5 = numpyro.infer.Predictive(
    model=model_8_5, posterior_samples=posterior_samples_8_5
)
log_likelihood_8_5 = numpyro.infer.log_likelihood(
    model=model_8_5,
    posterior_samples=posterior_samples_8_5,
    water=df["water_cent"].values,
    shade=df["shade_cent"].values,
    blooms=df["blooms_std"].values,
)
idata_8_5 = az.from_dict(
    posterior={k: v[None] for k, v in posterior_samples_8_5.items()},
    log_likelihood={k: v[None] for k, v in log_likelihood_8_5.items()},
)
```

### Code 8.25

```python
_, axs = plt.subplots(2, 3, sharey=True)
waters = jnp.linspace(-1, 1, 20)
for col, _shade in enumerate([-1, 0, 1]):
    _df = df.loc[df["shade_cent"] == _shade]
    for row, (model_name, _posterior_predictive) in enumerate(
        zip(
            ["model 8.4", "model 8.5"],
            [posterior_predictive_8_4, posterior_predictive_8_5],
        )
    ):
        ax = axs[row, col]
        ax.set(
            xlim=(-1.1, 1.1),
            ylim=(-0.1, 1.1),
            xlabel="water",
            ylabel="blooms",
            title=f"{model_name} | shade={_shade}",
        )
        ax.scatter(_df["water_cent"], _df["blooms_std"])
        _posterior_predictive_samples = _posterior_predictive(
            jrng, water=waters, shade=_shade, blooms=None
        )
        for _posterior_predictive_sample in _posterior_predictive_samples["mu"][:20]:
            ax.plot(waters, _posterior_predictive_sample, "k", alpha=0.3)

plt.tight_layout()
```

### Code 8.26

```python
prior_predictive_samples_8_4 = numpyro.infer.Predictive(
    model=model_8_4, num_samples=1_000
)(jrng, water=0, shade=0, blooms=None)
prior_predictive_samples_8_5 = numpyro.infer.Predictive(
    model=model_8_5, num_samples=1_000
)(jrng, water=0, shade=0, blooms=None)
```

## Easy
### 8E1
1. temperature
1. wealth
1. battery

### 8E2

Only (1): interaction between the heat and the amount of water

### 8E3

1. $C = \alpha + \Beta_H H + \Beta_W W + \Beta_{HW} HW$  (caramelization vs heat and water)
1. $S = \alpha + \Beta_C C + \Beta_F F$ (speed vs cylinders and fuel injector)
1. $C = \alpha + \Beta_P P + \Beta_F F$ (conservativsm vs parents' and friends' conservativsm)
1. $I = \alpha + \Beta_S S + \Beta_A A$ (intelligence vs social level and number of appendages)




## Medium
### 8M1

No amount of water or light can make a tulip bloom if the temperature is hot.

### 8M2

$ B = (\alpha + \Beta_W W + \Beta_S S + \Beta_{WS} W \cdot S) \mathbb{1}_{C}$

### 8M3

Not an interaction

$ R = \alpha + \Beta_{preys} P + \Beta_W W$


## Hard


### 8H1

```python
df = pd.read_csv("../data/tulips.csv", sep=";")
df["blooms_std"] = df["blooms"] / df["blooms"].max()
df["water_cent"] = df["water"] - 2
df["shade_cent"] = df["shade"] - 2
bed_labels = LabelEncoder().fit(df["bed"])
df["bed_idx"] = bed_labels.transform(df["bed"])
df.head()
```

```python
def model_8h1(bed, water, shade, blooms):
    with numpyro.plate(name="bed", size=len(bed_labels.classes_)):
        b_bed = numpyro.sample("b_bed", dist.Normal(0, 0.25))
    b_water = numpyro.sample("b_water", dist.Normal(0, 0.25))
    b_shade = numpyro.sample("b_shade", dist.Normal(0, 0.25))
    b_water_shade = numpyro.sample("b_water_shade", dist.Normal(0, 0.25))

    mu = numpyro.deterministic(
        "mu",
        b_bed[bed] + b_water * water + b_shade * shade + b_water_shade * water * shade,
    )
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    blooms = numpyro.sample("blooms", dist.Normal(mu, sigma), obs=blooms)
    return blooms


guide_8h1 = AutoLaplaceApproximation(model_8h1)
svi_8h1 = SVI(model=model_8h1, guide=guide_8h1, optim=optim, loss=loss).run(
    jrng,
    num_steps=2_500,
    bed=df["bed_idx"].values,
    water=df["water_cent"].values,
    shade=df["shade_cent"].values,
    blooms=df["blooms_std"].values,
)
posterior_samples_8h1 = guide_8h1.sample_posterior(
    jrng, params=svi_8h1.params, sample_shape=(1_000,)
)
log_likelihood_8h1 = numpyro.infer.log_likelihood(
    model=model_8h1,
    posterior_samples=posterior_samples_8h1,
    bed=df["bed_idx"].values,
    water=df["water_cent"].values,
    shade=df["shade_cent"].values,
    blooms=df["blooms_std"].values,
)
idata_8h1 = az.from_dict(
    posterior={k: v[None] for k, v in posterior_samples_8h1.items()},
    log_likelihood={k: v[None] for k, v in log_likelihood_8h1.items()},
)
numpyro.diagnostics.print_summary(
    prune_return_sites(posterior_samples_8h1, depth=2, exclude=["mu"]),
    prob=0.89,
    group_by_chain=False,
)
```

### 8H2

```python
comparison_8h2 = az.compare({"model_8_5": idata_8_5, "model_8h1": idata_8h1}, ic="loo")
az.plot_compare(comparison_8h2)
```

No discernible difference between the two models.


### 8H3

```python
_df = pd.read_csv("../data/rugged.csv", sep=";")
_df["log_gdp"] = jnp.log(_df["rgdppc_2000"].values)
df = _df.dropna(subset=["rgdppc_2000"]).copy()
df["log_gdp_std"] = df["log_gdp"] / df["log_gdp"].mean()
df["rugged_std"] = df["rugged"] / df["rugged"].max()
df["cid"] = jnp.logical_not(df["cont_africa"].values).astype(int)
df.head()
```

```python
numpyro.diagnostics.print_summary(
    prune_return_sites(posterior_samples_8_3, depth=2, exclude=["mu"]),
    group_by_chain=False,
)
pd.DataFrame(
    az.loo(idata_8_3, pointwise=True, scale="deviance")["pareto_k"],
    index=df["isocode"],
    columns=["pareto_k"],
).sort_values(["pareto_k"], ascending=False)
```

Seychelles indeed have outsized influence. To a lesser extent, so does Lesotho.

```python
def model_8h3(rugged, cid, log_gdp):
    with numpyro.plate(name="continent", size=2):
        a = numpyro.sample("a", dist.Normal(1, 0.1))
        b = numpyro.sample("b", dist.Normal(0, 0.3))
    mu = numpyro.deterministic("mu", a[cid] + b[cid] * (rugged - rugged_bar))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    log_gdp = numpyro.sample("log_gdp", dist.StudentT(2, mu, sigma), obs=log_gdp)
    return log_gdp


guide_8h3 = AutoLaplaceApproximation(model_8h3)
svi_8h3 = SVI(
    model=model_8h3,
    guide=guide_8h3,
    optim=optim,
    loss=loss,
    rugged=df["rugged_std"].values,
    cid=df["cid"].values,
    log_gdp=df["log_gdp_std"].values,
).run(jrng, 2000)
posterior_samples_8h3 = guide_8h3.sample_posterior(
    jrng,
    svi_8h3.params,
    sample_shape=(1_000,),
)
log_likelihood_8h3 = numpyro.infer.log_likelihood(
    model=model_8h3,
    posterior_samples=posterior_samples_8h3,
    rugged=df["rugged_std"].values,
    cid=df["cid"].values,
    log_gdp=df["log_gdp_std"].values,
)
idata_8h3 = az.from_dict(
    posterior={k: v[None] for k, v in posterior_samples_8h3.items()},
    log_likelihood={k: v[None] for k, v in log_likelihood_8h3.items()},
)
```

```python
numpyro.diagnostics.print_summary(
    prune_return_sites(posterior_samples_8h3, depth=2, exclude=["mu"]),
    group_by_chain=False,
)
pd.DataFrame(
    az.loo(idata_8h3, pointwise=True, scale="deviance")["pareto_k"],
    index=df["isocode"],
    columns=["pareto_k"],
).sort_values(["pareto_k"], ascending=False)
```

No more outsized influence of any one country. Also, the beta coefficient is not much stronger.


### 8H4

```python
df = pd.read_csv("../data/nettle.csv", sep=";")
df["lang.per.cap"] = df["num.lang"] / df["k.pop"]
df["log_langpc"] = jnp.log(df["lang.per.cap"].values)
df["log_area"] = jnp.log(df["area"].values)
print(df.shape)
df.head()
```

```python
def model_8h4a(mean_growing_season, log_langpc):
    # Prior: guessing 10 languages spoke in France (populaion = 60M)
    # log means taking the order of magnitude. 2 order magnitude in SD s/b v vague
    alpha = numpyro.sample(
        "alpha",
        dist.Normal(jnp.log(10 / 60_000), 2),
    )
    beta_mean_growing_season = numpyro.sample(
        "beta_mean_growing_season", dist.Normal(0, 2)
    )
    mu = numpyro.deterministic(
        "mu", alpha + beta_mean_growing_season * mean_growing_season
    )
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    log_langpc = numpyro.sample("log_langpc", dist.Normal(mu, sigma), obs=log_langpc)
    return log_langpc


guide_8h4a = AutoLaplaceApproximation(model_8h4a)
svi_8h4a = SVI(
    model=model_8h4a,
    guide=guide_8h4a,
    optim=optim,
    loss=loss,
    mean_growing_season=df["mean.growing.season"].values,
    log_langpc=df["log_langpc"].values,
).run(jrng, num_steps=2_500)
posterior_samples_8h4a = guide_8h4a.sample_posterior(
    jrng, params=svi_8h4a.params, sample_shape=(1_000,)
)
numpyro.diagnostics.print_summary(
    prune_return_sites(posterior_samples_8h4a), group_by_chain=False
)
```

Good evidence that the longer the mean growing season, the more language diversity

```python
def model_8h4b(sd_growing_season, log_langpc):
    alpha = numpyro.sample(
        "alpha",
        dist.Normal(jnp.log(10 / 60_000), 2),
    )
    beta_sd_growing_season = numpyro.sample("beta_sd_growing_season", dist.Normal(0, 2))
    mu = numpyro.deterministic(
        "mu",
        alpha + beta_sd_growing_season * sd_growing_season,
    )
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    log_langpc = numpyro.sample("log_langpc", dist.Normal(mu, sigma), obs=log_langpc)
    return log_langpc


guide_8h4b = AutoLaplaceApproximation(model_8h4b)
svi_8h4b = SVI(
    model=model_8h4b,
    guide=guide_8h4b,
    optim=optim,
    loss=loss,
    sd_growing_season=df["sd.growing.season"].values,
    log_langpc=df["log_langpc"].values,
).run(jrng, num_steps=2_500)
posterior_samples_8h4b = guide_8h4b.sample_posterior(
    jrng, params=svi_8h4b.params, sample_shape=(1_000,)
)
numpyro.diagnostics.print_summary(
    prune_return_sites(posterior_samples_8h4b), group_by_chain=False
)
```

Good evidence that the larger the uncertainty around the growing season, the less language diversity

```python
def model_8h4c(mean_growing_season, sd_growing_season, log_langpc):
    alpha = numpyro.sample(
        "alpha",
        dist.Normal(jnp.log(10 / 60_000), 2),
    )
    beta_mean_growing_season = numpyro.sample(
        "beta_mean_growing_season", dist.Normal(0, 10)
    )
    beta_sd_growing_season = numpyro.sample(
        "beta_sd_growing_season", dist.Normal(0, 10)
    )
    beta_mean_sd_growing_season = numpyro.sample(
        "beta_mean_sd_growing_season", dist.Normal(0, 10)
    )
    mu = numpyro.deterministic(
        "mu",
        alpha
        + beta_mean_growing_season * mean_growing_season
        + beta_sd_growing_season * sd_growing_season
        + beta_mean_sd_growing_season * mean_growing_season * sd_growing_season,
    )
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    log_langpc = numpyro.sample("log_langpc", dist.Normal(mu, sigma), obs=log_langpc)
    return log_langpc


guide_8h4c = AutoLaplaceApproximation(model_8h4c)
svi_8h4c = SVI(
    model=model_8h4c,
    guide=guide_8h4c,
    optim=optim,
    loss=loss,
    mean_growing_season=df["mean.growing.season"].values,
    sd_growing_season=df["sd.growing.season"].values,
    log_langpc=df["log_langpc"].values,
).run(jrng, num_steps=2_500)
posterior_samples_8h4c = guide_8h4c.sample_posterior(
    jrng, params=svi_8h4c.params, sample_shape=(1_000,)
)
numpyro.diagnostics.print_summary(
    prune_return_sites(posterior_samples_8h4c), group_by_chain=False
)
```

```python

```
