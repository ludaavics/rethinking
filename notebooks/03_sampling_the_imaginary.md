---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: statistical-rethinking
    language: python
    name: statistical-rethinking
---

# Chapter 3: Sampling the Imaginary

```python
import random
from typing import Sequence

import arviz as az
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pandas as pd
import plotly
import plotly.graph_objects as go
import plotly.io as pio
from scipy.stats import gaussian_kde

pd.options.plotting.backend = "plotly"

seed = 84735
pio.templates.default = "plotly_white"
rng = jax.random.PRNGKey(seed)
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
def calculate_posterior(W: int, L: int, prior: Sequence[float], grid_size: int):
    grid = jnp.linspace(0, 1, grid_size)
    likelihood = jnp.exp(dist.Binomial(total_count=W + L, probs=grid).log_prob(W))
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
samples = p_grid[
    dist.Categorical(probs=posterior).sample(rng, (10_000,))
]

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
samples = p_grid[dist.Categorical(probs=posterior).sample(rng, (10_000,))]
```

### Code 3.12

```python
def percentile_interval(samples, prob):
    prob = min(prob, 1 - prob)
    return jnp.quantile(samples, jnp.array([prob / 2, 1 - prob / 2]))


percentile_interval(samples, 0.5)
```

### Code 3.13

```python
numpyro.diagnostics.hpdi(samples, prob=0.5)
```

### Code 3.14

```python
p_grid[jnp.argmax(posterior)]
```

### Code 3.15

```python
samples[jnp.argmax(gaussian_kde(samples, bw_method=0.01)(samples))]
```

### Code 3.16

```python
display(samples.mean())
jnp.median(samples)
```

### Code 3.17

```python
jnp.sum(jnp.abs(0.5 - p_grid) * posterior)
```

### Code 3.18

```python
loss = jax.vmap(lambda d: jnp.sum(jnp.abs(d - p_grid) * posterior))(p_grid)
display(pd.DataFrame(loss, index=p_grid).plot())
```

### Code 3.19



```python
p_grid[jnp.argmin(loss)]
```

### Code 3.20

```python
jnp.exp(dist.Binomial(total_count=2, probs=0.7).log_prob(jnp.arange(3)))
```

### Code 3.21

```python
with numpyro.handlers.seed(rng_seed=seed):
    dummy_w = numpyro.sample("dummy_w", dist.Binomial(total_count=2, probs=0.7))
dummy_w
```

### Code 3.22

```python
with numpyro.handlers.seed(rng_seed=seed):
    dummy_w = numpyro.sample(
        "dummy_w", dist.Binomial(total_count=2, probs=0.7), sample_shape=(10,)
    )
dummy_w
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

### Code 3.24

```python
with numpyro.handlers.seed(rng_seed=seed):
    dummy_w = numpyro.sample(
        "dummy_w", dist.Binomial(total_count=9, probs=0.7), sample_shape=(100_000,)
    )
dummy_w = pd.DataFrame(dummy_w, columns=["dummy_w"])
dummy_w.plot(kind="hist")
```

### Code 3.25

```python
w = dist.Binomial(total_count=9, probs=0.6).sample(jax.random.PRNGKey(seed), (10_000,))
pd.DataFrame(w).plot(kind="hist")
```

### Code 3.26

```python
w = dist.Binomial(total_count=9, probs=samples).sample(
    jax.random.PRNGKey(seed),
)
pd.DataFrame(w).plot(kind="hist")
```

## Hard


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
prior = [0.5] * grid_size
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
posterior_samples = p_grid[dist.Categorical(probs=posterior).sample(rng, (10_000,))]
print(f"50% HDPI: {numpyro.diagnostics.hpdi(posterior_samples, prob=0.5)}")
print(f"89% HDPI: {numpyro.diagnostics.hpdi(posterior_samples, prob=0.89)}")
print(f"97% HDPI: {numpyro.diagnostics.hpdi(posterior_samples, prob=0.97)}")
```

### 3H3

```python
posterior_predictive_samples = dist.Binomial(
    total_count=births.size, probs=posterior_samples
).sample(rng)
print(
    f"Posterior predictive distribution of number of boys has mean {posterior_predictive_samples.mean():.0f} "
    f"vs observation of {births.sum()}: we're evaluating model against training data"
)
pd.DataFrame(posterior_predictive_samples, columns=["n_boys"]).plot(kind="hist")
```

### 3H4

```python
posterior_predictive_samples = dist.Binomial(
    total_count=births.shape[1], probs=posterior_samples
).sample(rng)
print(
    f"Posterior predictive distribution of first born sons has mean {posterior_predictive_samples.mean():.0f} "
    f"vs obersvation of {births[0].sum()}; still reasonable but not as good as purely 'in-sample'."
)
pd.DataFrame(posterior_predictive_samples, columns=["n_first_born_boys"]).plot(
    kind="hist"
)
```

### 3H5

```python
posterior_predictive_samples = dist.Binomial(
    total_count=jnp.logical_not(births[0]).sum(), probs=posterior_samples
).sample(rng)
print(
    f"PPD of boys with big sisters of {posterior_predictive_samples.mean():.0f} "
    f"is completely out of line with observations of {births[1].sum()}: we didn't model "
    "the correlation between first and second birth that's present in our dataset."
)
pd.DataFrame(posterior_predictive_samples, columns=["n_boys_with_big_sister"]).plot(
    kind="hist"
)
```

```python

```
