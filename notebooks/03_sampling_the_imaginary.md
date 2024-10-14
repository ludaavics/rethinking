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

# Chapter 3: Sampling the Imaginary

```python
%load_ext jupyter_black
```

```python
import random
from typing import Sequence

import jax
import arviz as az
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
import plotly
import plotly.graph_objects as go
import plotly.io as pio
from scipy import stats, optimize

pd.options.plotting.backend = "plotly"

seed = 84735
pio.templates.default = "plotly_white"
rng = np.random.default_rng(seed=seed)
jrng = jax.random.PRNGKey(seed)
```

## Code
### Code 3.1

```python
p_positive_vampire = 0.95
p_positive_mortal = 0.01
p_vampire = 0.001
p_positive = p_positive_vampire * p_vampire + p_positive_mortal * (1 - p_vampire)
p_vampire_positive = p_positive_vampire * p_vampire / p_positive
p_vampire_positive
```

### Code 3.2

```python
def calculate_posterior_numpyro(W: int, L: int, prior: Sequence[float], grid_size: int):
    grid = jnp.linspace(0, 1, grid_size)
    likelihood = jnp.exp(dist.Binomial(total_count=W + L, probs=grid).log_prob(W))
    raw_posterior = prior * likelihood
    posterior = raw_posterior / raw_posterior.sum()
    return posterior


def calculate_posterior(W: int, L: int, prior: Sequence[float], grid_size: int):
    p_grid = jnp.linspace(0, 1, grid_size)
    likelihood = stats.binom.pmf(k=W, n=W + L, p=p_grid)
    raw_posterior = prior * likelihood
    posterior = raw_posterior / raw_posterior.sum()
    return posterior


W = 6
L = 3
grid_size = 1_000
prior = jnp.full(grid_size, 1)
p_grid = jnp.linspace(0, 1, grid_size)
posterior = calculate_posterior(W, L, prior, grid_size)
```

### Code 3.3

```python
samples_numpyro = p_grid[dist.Categorical(probs=posterior).sample(jrng, (10_000,))]
```

```python
samples = rng.choice(p_grid, size=10_000, replace=True, p=posterior)
```

### Code 3.4

```python
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=jnp.arange(10_000),
        y=samples,
        mode="markers",
        line={"color": "rgba(0, 0, 255, 0.2)"},
    )
)
```

### Code 3.5

```python
az.plot_density({"": samples}, hdi_prob=1)
```

### Code 3.6

```python
posterior[p_grid < 0.5].sum()
```

### Code 3.7

```python
jnp.sum(samples < 0.5) / samples.shape[0]
```

### Code 3.8

```python
jnp.sum(jnp.logical_and(samples > 0.5, samples < 0.75)) / samples.shape[0]
```

### Code 3.9

```python
jnp.quantile(samples, 0.8)
```

### Code 3.10

```python
jnp.quantile(samples, jnp.array([0.1, 0.9]))
```

### Code 3.11

```python
posterior = calculate_posterior(W=3, L=0, prior=jnp.full(1_000, 1), grid_size=1_000)
samples_skewed = p_grid[dist.Categorical(probs=posterior).sample(jrng, (10_000,))]
```

### Code 3.12

```python
def percentile_interval(samples, prob):
    prob = min(prob, 1 - prob)
    return jnp.quantile(samples, jnp.array([prob / 2, 1 - prob / 2]))


percentile_interval(samples_skewed, 0.5)
```

### Code 3.13

```python
numpyro.diagnostics.hpdi(samples_skewed, prob=0.5)
```

```python
az.hdi(np.array(samples_skewed), hdi_prob=0.5)
```

### Code 3.14

```python
p_grid[jnp.argmax(posterior)]
```

### Code 3.15

```python
samples_skewed[
    jnp.argmax(stats.gaussian_kde(samples_skewed, bw_method=0.01)(samples_skewed))
]
```

### Code 3.16

```python
display(samples_skewed.mean())
jnp.median(samples_skewed)
```

### Code 3.17

```python
jnp.sum(jnp.abs(0.5 - p_grid) * posterior)
```

### Code 3.18

```python
loss = jax.vmap(lambda d: jnp.sum(jnp.abs(d - p_grid) * posterior))(p_grid)
fig = go.Figure(data=go.Scatter(x=p_grid, y=loss))
fig.update_layout(
    xaxis={"title": "parameter"},
    yaxis={"title": "expected loss"},
)
fig.show()
```

### Code 3.19



```python
p_grid[jnp.argmin(loss)]
```

### Code 3.20

```python
jnp.exp(dist.Binomial(total_count=2, probs=0.7).log_prob(jnp.arange(3)))
```

```python
stats.binom.pmf(k=jnp.arange(3), p=0.7, n=2)
```

### Code 3.21

```python
with numpyro.handlers.seed(rng_seed=seed):
    dummy_w = numpyro.sample("dummy_w", dist.Binomial(total_count=2, probs=0.7))
dummy_w
```

```python
stats.binom.rvs(n=2, p=0.7, random_state=rng)
```

### Code 3.22

```python
with numpyro.handlers.seed(rng_seed=seed):
    dummy_w = numpyro.sample(
        "dummy_w", dist.Binomial(total_count=2, probs=0.7), sample_shape=(10,)
    )
dummy_w
```

```python
stats.binom.rvs(n=2, p=0.7, size=(10,), random_state=rng)
```

### Code 3.23

```python
with numpyro.handlers.seed(rng_seed=seed):
    dummy_w = numpyro.sample(
        "dummy_w", dist.Binomial(total_count=2, probs=0.7), sample_shape=(100_000,)
    )
dummy_w = pd.DataFrame(dummy_w, columns=["dummy_w"])
dummy_w["freq"] = 1
dummy_w.groupby("dummy_w").sum() / 100_000
```

```python
dummy_w = stats.binom.rvs(n=2, p=0.7, size=(10_000,), random_state=rng)
dummy_w = pd.DataFrame(dummy_w, columns=["dummy_w"])
dummy_w["freq"] = 1
dummy_w.groupby("dummy_w").sum() / 100_000
```

### Code 3.24

```python
with numpyro.handlers.seed(rng_seed=seed):
    dummy_w = numpyro.sample(
        "dummy_w", dist.Binomial(total_count=9, probs=0.7), sample_shape=(100_000,)
    )
dummy_w = pd.DataFrame(dummy_w, columns=["dummy_w"])
dummy_w.plot(kind="hist")
```

```python
dummy_w = stats.binom.rvs(n=9, p=0.7, size=(100_000,), random_state=rng)
dummy_w = pd.DataFrame(dummy_w, columns=["dummy_w"])
dummy_w.plot(kind="hist")
```

### Code 3.25

```python
w = dist.Binomial(total_count=9, probs=0.6).sample(jrng, (10_000,))
pd.DataFrame(w).plot(kind="hist")
```

```python
w = stats.binom.rvs(n=9, p=0.6, size=(100_000,), random_state=rng)
pd.DataFrame(w).plot(kind="hist")
```

### Code 3.26

```python
w = dist.Binomial(total_count=9, probs=samples).sample(jrng)
pd.DataFrame(w).plot(kind="hist")
```

```python
w = stats.binom.rvs(n=9, p=samples, random_state=rng)
pd.DataFrame(w).plot(kind="hist")
```

## Easy

```python
def calculate_posterior(W: int, L: int, grid_size: int):
    p_grid = jnp.linspace(0, 1, grid_size)
    prior = jnp.array([1 / grid_size] * grid_size)
    likelihood = jnp.exp(dist.Binomial(total_count=W + L, probs=p_grid).log_prob(W))
    raw_posterior = prior * likelihood
    posterior = raw_posterior / raw_posterior.sum()
    samples = p_grid[dist.Categorical(probs=posterior).sample(jrng, (10_000,))]
    return {"pdf": posterior, "samples": samples}


grid_size = 1_000
posterior = calculate_posterior(W=6, L=3, grid_size=grid_size)
pdf = posterior["pdf"]
samples = posterior["samples"]
```

### 3E1

```python
jnp.sum(samples < 0.2) / samples.shape[0]
```

```python
p_grid = jnp.linspace(0, 1, grid_size)
pdf[p_grid < 0.2].sum()
```

### 3E2

```python
jnp.sum(samples > 0.8) / samples.shape[0]
```

```python
pdf[p_grid > 0.8].sum()
```

### 3E3

```python
jnp.sum(jnp.logical_and(0.2 < samples, samples < 0.8)) / samples.shape[0]
```

```python
pdf[np.logical_and(0.2 < p_grid, p_grid < 0.8)].sum()
```

### 3E4

```python
jnp.quantile(samples, 0.2)
```

```python
p_grid[jnp.searchsorted(pdf.cumsum(), 0.2)]
```

### 3E5

```python
jnp.quantile(samples, 0.8)
```

```python
p_grid[jnp.searchsorted(pdf.cumsum(), 0.8)]
```

### 3E6

```python
numpyro.diagnostics.hpdi(samples, prob=0.66)
```

### 3E7

```python
jnp.quantile(samples, jnp.array([(1 - 0.66) / 2, 1 - (1 - 0.66) / 2]))
```

## Medium


### 3M1

```python
def calculate_posterior(W: int, L: int, grid_size: int, n_samples):
    p_grid = jnp.linspace(0, 1, grid_size)
    prior = jnp.array([1 / grid_size] * grid_size)
    likelihood = jnp.exp(dist.Binomial(total_count=W + L, probs=p_grid).log_prob(W))
    raw_posterior = prior * likelihood
    posterior = raw_posterior / raw_posterior.sum()
    samples = p_grid[dist.Categorical(probs=posterior).sample(jrng, (n_samples,))]
    return {"pdf": posterior, "samples": samples}


grid_size = 1_000
n_samples = 10_000
posterior = calculate_posterior(W=8, L=7, grid_size=grid_size, n_samples=n_samples)
pdf = posterior["pdf"]
samples = posterior["samples"]
pd.DataFrame(
    pdf, index=pd.Index(jnp.linspace(0, 1, grid_size), name="posterior proba of W")
).plot()
```

### 3M2

```python
numpyro.diagnostics.hpdi(samples, prob=0.9)
```

### 3M3

```python
with numpyro.handlers.seed(rng_seed=seed):
    n_water = numpyro.sample("n_water", dist.Binomial(total_count=15, probs=samples))
p_8_w_7_l = jnp.sum(n_water == 8) / n_water.shape[0]
print(f"Probabilty 8/15 water is {p_8_w_7_l:.2%}")
pd.DataFrame(n_water).plot(kind="hist")
```

### 3M4

```python
with numpyro.handlers.seed(rng_seed=seed):
    n_water = numpyro.sample("n_water", dist.Binomial(total_count=9, probs=samples))
p_6_w_3_l = jnp.sum(n_water == 6) / n_water.shape[0]
print(f"Probabilty 6/9 water is {p_6_w_3_l:.2%}")
pd.DataFrame(n_water).plot(kind="hist")
```

### 3M5

```python
def calculate_posterior(W: int, L: int, grid_size: int, n_samples):
    p_grid = jnp.linspace(0, 1, grid_size)
    prior = jnp.array([0 if p < 0.5 else 1 for p in p_grid])
    prior /= prior.sum()
    likelihood = jnp.exp(dist.Binomial(total_count=W + L, probs=p_grid).log_prob(W))
    raw_posterior = prior * likelihood
    posterior = raw_posterior / raw_posterior.sum()
    samples = p_grid[dist.Categorical(probs=posterior).sample(jrng, (n_samples,))]
    return {"pdf": posterior, "samples": samples}


grid_size = 1_000
n_samples = 10_000
posterior = calculate_posterior(W=8, L=7, grid_size=grid_size, n_samples=n_samples)
pdf = posterior["pdf"]
samples = posterior["samples"]
pd.DataFrame(
    pdf, index=pd.Index(jnp.linspace(0, 1, grid_size), name="posterior proba of W")
).plot().show()
az.plot_dist(samples)
```

```python
n_water = stats.binom.rvs(n=15, p=samples, random_state=rng)
p_8_w_7_l = jnp.sum(n_water == 8) / n_water.shape[0]
print(f"Probabilty 8/15 water is {p_8_w_7_l:.2%}")
pd.DataFrame(n_water).plot(kind="hist")
```

### 3M6

```python
def pi_width(n_tosses, n_posterior_samples, n_experiments):
    p_grid = jnp.linspace(0, 1, grid_size)
    prior = jnp.array([1 / grid_size] * grid_size)

    width = [jnp.nan] * n_experiments

    n_water = stats.binom.rvs(
        n=n_tosses, p=0.7, random_state=rng, size=(n_experiments,)
    )
    likelihood = jnp.exp(
        jnp.array(
            [
                dist.Binomial(total_count=n_tosses, probs=p_grid).log_prob(_n_water)
                for _n_water in n_water
            ]
        )
    )
    raw_posterior = prior * likelihood
    posterior = raw_posterior / raw_posterior.sum()
    samples = jnp.array(
        [
            p_grid[
                dist.Categorical(probs=_posterior).sample(jrng, (n_posterior_samples,))
            ]
            for _posterior in posterior
        ]
    )
    quantiles = jnp.quantile(samples, jnp.array([0.005, 0.995]), axis=1)
    return float((quantiles[1] - quantiles[0]).mean())


def objective(n_tosses):
    return pi_width(int(n_tosses), n_posterior_samples=10_000, n_experiments=100) - 0.05
```

```python
n_tosses = list(range(500, 5_500, 500))
widths = [
    pi_width(_n_tosses, n_posterior_samples=10_000, n_experiments=100)
    for _n_tosses in n_tosses
]
```

```python
fig = go.Figure(data=go.Scatter(x=n_tosses, y=widths))
fig.add_hline(y=0.05, line={"color": "red", "dash": "dash"})
fig.update_layout(
    xaxis={"title": "number of tosses"},
    yaxis={"title": "width of 99% compatibility interval"},
)
fig
```

```python
result = optimize.root_scalar(objective, x0=2_000, bracket=[1_500, 2_500], xtol=1)
result
```

### 3H1

```python
# fmt: off
births_1 = [
    1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 
    0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 
    0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 
    0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1,
]
births_2 = [ 
    0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 
    1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
    1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 
    0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0,
]
births = jnp.array([births_1, births_2])
```

```python
grid_size = 1_000

p_grid = jnp.linspace(0, 1, grid_size)
prior = [1 / grid_size] * grid_size
likelhihood = jnp.exp(
    dist.Binomial(total_count=births.size, probs=p_grid).log_prob(births.sum())
)
raw_posterior = likelhihood * jnp.array(prior)
posterior = raw_posterior / raw_posterior.sum()
map_p = p_grid[jnp.argmax(posterior)]
print(f"p={map_p:.2%} maximizes the posterior probability.")
```

### 3H2

```python
posterior_samples = p_grid[dist.Categorical(probs=posterior).sample(jrng, (10_000,))]
print(f"50% HDPI: {numpyro.diagnostics.hpdi(posterior_samples, prob=0.5)}")
print(f"89% HDPI: {numpyro.diagnostics.hpdi(posterior_samples, prob=0.89)}")
print(f"97% HDPI: {numpyro.diagnostics.hpdi(posterior_samples, prob=0.97)}")
```

### 3H3

```python
posterior_predictive_samples = dist.Binomial(
    total_count=births.size, probs=posterior_samples
).sample(jrng)
print(
    "Posterior Predictive Check:\n"
    "Samples from posterior predictive distribution have, on average, "
    f"{posterior_predictive_samples.mean():.0f} boys "
    f"vs {births.sum()} in our training data"
)
fig = pd.DataFrame(posterior_predictive_samples, columns=["n_boys"]).plot(kind="hist")
fig.add_vline(x=births.sum(), line={"color": "red"})
```

### 3H4

```python
posterior_predictive_samples = dist.Binomial(
    total_count=births.shape[1], probs=posterior_samples
).sample(jrng)
print(
    f"Posterior predictive distribution of first born sons has mean {posterior_predictive_samples.mean():.0f} "
    f"vs obersvation of {births[0].sum()}; still reasonable but not as good as purely 'in-sample'."
)
fig = pd.DataFrame(posterior_predictive_samples, columns=["n_first_born_boys"]).plot(
    kind="hist"
)
fig.add_vline(x=births[0, :].sum(), line={"color": "red"})
```

### 3H5

```python
posterior_predictive_samples.shape
```

```python
posterior_predictive_samples = dist.Binomial(
    total_count=jnp.logical_not(births[0]).sum(), probs=posterior_samples
).sample(jrng)
is_big_sister = births[0, :] == 0
print(
    f"PoPD of boys with big sisters of {posterior_predictive_samples.mean():.0f} "
    f"is completely out of line with observations of {births[1, is_big_sister].sum()}: we didn't model "
    "the correlation between first and second birth that's present in our dataset."
)
fig = pd.DataFrame(
    posterior_predictive_samples, columns=["n_boys_with_big_sister"]
).plot(kind="hist")

fig.add_vline(x=births[1, is_big_sister].sum(), line={"color": "red"})
```
