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

# Chapter 2: Small Worlds And Large Worlds

```python
%load_ext jupyter_black
```

```python
from typing import Sequence

import arviz as az
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import numpyro.optim as optim
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import pymc as pm
from scipy import stats
from jax import random as jrandom
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoLaplaceApproximation

seed = 84735
pio.templates.default = "plotly_white"
```

## Code 
### Code 2.1

```python
ways = jnp.array([0, 3, 8, 9, 0])
ways / jnp.sum(ways)
```

### Code 2.2

```python
jnp.exp(dist.Binomial(total_count=9, probs=0.5).log_prob(6))
```

```python
stats.binom.pmf(k=6, n=9, p=[0.5])
```

### Code 2.3

```python
n_trials = 9
n_successes = 6
grid_size = 20
prior = jnp.full(grid_size, 1)
```

```python
def calculate_grid_approximation_posterior_numpyro(
    n_trials: int,
    n_successes: int,
    prior: Sequence[float],
    grid_size: int,
):
    grid = jnp.linspace(0, 1, grid_size)
    likelihood = jnp.exp(
        dist.Binomial(total_count=n_trials, probs=grid).log_prob(n_successes)
    )
    raw_posterior = prior * likelihood
    posterior = raw_posterior / raw_posterior.sum()
    return posterior


posterior_numpyro = calculate_grid_approximation_posterior_numpyro(
    n_trials, n_successes, prior, grid_size
)
posterior_numpyro
```

```python
def calculate_grid_approximation_posterior_pymc(
    n_trials: int,
    n_successes: int,
    prior: Sequence[float],
    grid_size: int,
):
    grid = jnp.linspace(0, 1, grid_size)
    likelihood = stats.binom.pmf(k=n_successes, n=n_trials, p=grid)
    raw_posterior = prior * likelihood
    posterior = raw_posterior / raw_posterior.sum()
    return posterior


posterior_pymc = calculate_grid_approximation_posterior_pymc(
    n_trials, n_successes, prior, grid_size
)
posterior_pymc
```

### Code 2.4

```python
def plot_grid_approximation(prior, posterior, *, title=None, grid_size=20):
    grid = jnp.linspace(0, 1, grid_size)
    title = title or f"Grid Approximation of Posterior Distribution"

    prior /= prior.sum()
    posterior /= posterior.sum()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=grid,
            y=prior,
            name="prior",
            mode="lines+markers",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=grid,
            y=posterior,
            name="posterior",
            mode="lines+markers",
        )
    )
    fig.update_layout(
        title=title,
        xaxis={"title": "p"},
        yaxis={"title": "posterior probability"},
    )
    fig.show()
    return fig


fig = plot_grid_approximation(prior, posterior_pymc)
```


### Code 2.5

```python
grid = jnp.linspace(0, 1, grid_size)
prior = jnp.where(grid < 0.5, 0, 1)
posterior = calculate_grid_approximation_posterior_numpyro(
    n_trials, n_successes, prior, grid_size
)
fig = plot_grid_approximation(prior, posterior)
```

```python
grid = jnp.linspace(0, 1, grid_size)
prior = jnp.exp(-5 * jnp.abs(grid - 0.5))
posterior = calculate_grid_approximation_posterior_pymc(
    n_trials, n_successes, prior, grid_size
)

fig = plot_grid_approximation(prior, posterior)
```

### Code 2.6

```python
W = 6
L = 3
n_steps = 1_000
n_samples = 1_000
```

```python
def calculate_quadratic_approximation_posterior_numpyro(
    W,
    L,
    n_steps,
    n_samples,
):
    def model(W, L):
        p = numpyro.sample("p", dist.Uniform(0, 1))
        numpyro.sample("W", dist.Binomial(total_count=W + L, probs=p), obs=W)

    guide = AutoLaplaceApproximation(model)
    loss = Trace_ELBO()
    learning_rate = 1
    optimizer = optim.Adam(learning_rate)
    rng_key_train, rng_key_sample = jrandom.split(jrandom.PRNGKey(seed))

    svi = SVI(model, guide, optimizer, loss, W=W, L=L)
    svi_result = svi.run(rng_key_train, num_steps=n_steps)

    samples = guide.sample_posterior(
        rng_key_sample, params=svi_result.params, sample_shape=(n_samples,)
    )

    return svi_result, samples


# display summary of quadratic approximation
svi_result, samples = calculate_quadratic_approximation_posterior_numpyro(
    W=W, L=L, n_steps=n_steps, n_samples=n_samples
)
numpyro.diagnostics.print_summary(samples, prob=0.89, group_by_chain=False)
```

```python
def calculate_quadratic_approximation_posterior_pymc(W, L, n_steps, n_samples):
    raise NotImplementedError
```

### Code 2.7

```python
# analtyical calculation
W = 6
L = 3

x = jnp.linspace(0, 1, 101)
analytical_posterior = jnp.exp(dist.Beta(W + 1, L + 1).log_prob(x))
quad_posterior = jnp.exp(
    dist.Normal(loc=samples["p"].mean(), scale=samples["p"].std()).log_prob(x)
)

fig = go.Figure(
    [
        go.Scatter(
            x=x,
            y=analytical_posterior,
            name="analytical",
        ),
        go.Scatter(
            x=x,
            y=quad_posterior,
            name="quadratic",
        ),
    ]
)

fig.show()
```

### Code 2.8

```python
n_samples = 1_000
p = [jnp.nan] * n_samples
p[0] = 0.5
W = 6
L = 3

with numpyro.handlers.seed(rng_seed=seed):
    for i in range(1, n_samples):
        p_new = numpyro.sample("p_new", dist.Normal(p[i - 1], 0.1))
        p_new = jnp.abs(p_new) if p_new < 0 else p_new
        p_new = 2 - p_new if p_new > 1 else p_new
        q0 = jnp.exp(dist.Binomial(total_count=W + L, probs=p[i - 1]).log_prob(W))
        q1 = jnp.exp(dist.Binomial(total_count=W + L, probs=p_new).log_prob(W))
        u = numpyro.sample("u", dist.Uniform(0, 1))
        p[i] = float(p_new if u < q1 / q0 else p[i - 1])
```

### Code 2.9

```python
ax = az.plot_density({"p": p}, hdi_prob=1)
plt.plot(x, analytical_posterior, "--")
```

## Medium
### 2M1

```python
def plot_grid_posterior(data, *, grid_size=100):
    prior = jnp.array([1 / grid_size] * grid_size)
    grid = jnp.linspace(0, 1, grid_size)
    n_success = jnp.sum(jnp.array([x == "W" for x in data]))
    likelihood = jnp.exp(
        dist.Binomial(total_count=len(data), probs=grid).log_prob(n_success)
    )
    posterior = prior * likelihood
    posterior /= jnp.sum(posterior)

    fig = go.Figure(
        [
            go.Scatter(
                x=grid,
                y=prior,
                name="prior",
            ),
            go.Scatter(
                x=grid,
                y=posterior,
                name="posteriror",
            ),
        ]
    )
    fig.update_layout(
        title=",".join(data),
        xaxis={"title": "p"},
        yaxis={"title": "probability"},
    )
    return {"posterior": posterior, "fig": fig}


for data in [
    ["W", "W", "W"],
    ["W", "W", "W", "L"],
    ["L", "W", "W", "L", "W", "W", "W"],
]:
    results = plot_grid_posterior(data)
    results["fig"].show()
```

### 2M2

```python
def plot_grid_posterior(data, *, grid_size=100):
    grid = jnp.linspace(0, 1, grid_size)
    prior = jnp.array([0 if p < 0.5 else 1 for p in grid])
    prior /= prior.sum()
    n_success = jnp.sum(jnp.array([x == "W" for x in data]))
    likelihood = jnp.exp(
        dist.Binomial(total_count=len(data), probs=grid).log_prob(n_success)
    )
    posterior = prior * likelihood
    posterior /= jnp.sum(posterior)

    fig = go.Figure(
        [
            go.Scatter(
                x=grid,
                y=prior,
                name="prior",
            ),
            go.Scatter(
                x=grid,
                y=posterior,
                name="posteriror",
            ),
        ]
    )
    fig.update_layout(
        title=",".join(data),
        xaxis={"title": "p"},
        yaxis={"title": "probability"},
    )
    return {"posterior": posterior, "fig": fig}


for data in [
    ["W", "W", "W"],
    ["W", "W", "W", "L"],
    ["L", "W", "W", "L", "W", "W", "W"],
]:
    results = plot_grid_posterior(data)
    results["fig"].show()
```

### 2M3
$$
\begin{aligned}
P(Earth|Land) & = \frac{P(Land|Earth) \cdot P(Earth)}{P(Land)} \\ 
    & = \frac{P(Land|Earth) \cdot P(Earth)}{P(Land|Earth) \cdot P(Earth) + P(Land|Mars) \cdot P(Mars)} \\ 
    & = \frac{0.3 \cdot 0.5}{0.3 \cdot 0.5 + 1 \cdot 0.5} \\ 
    & \approx 0.23
\end{aligned}
$$


### 2M4

 - card 1 has 2 ways of showing observed data
 - card 2 has 1 way of showing observed data
 - card 3 has 0 ways of showing observed data

Of the 3 possibles ways of showing observed data, only 2 ways to have the other side also black, hence 2/3 chances the other side is black.



### 2M5

Now 5 ways of seeing observed data and of those 5, 4 ways that the other side is also black. p=4/5


### 2M6

 - card 1 has 2 ways of yielding the observed data
 - card 2 has 2 ways of yielding the observed data
 - card 3 has 0 ways of yielding the observed data

Of the 4 possible ways of getting the observed data, only 2 ways ot have the other side also black. p=0.5


### 2M7

| cards | count |
| :---: | :---: |
|  1,2  |   2   |
|  1,3  |   4   |
|  2,1  |   0   |
|  2,3  |   2   |
|  3,1  |   0   |
|  3,2  |   0   |

p = 6 / 8


## Hard
### 2H1


Notation:
$$
\begin{aligned}
A & \sim \text{Species A} \\
B & \sim \text{Species B} \\
T & \sim \text{Twin} \\
\end{aligned}
$$

First we calculate the posterior probability of panda being of species A:

$$
\begin{aligned}
P(A|T) & = \frac{P(T|A) \cdot P(A)}{P(T)} \\ 
 & = \frac{P(T|A) \cdot P(A)}{P(T|A) \cdot P(A) + P(T|B) \cdot P(B)} \\
 & = \frac{0.1 \cdot 0.5}{0.1 \cdot 0.5 + 0.2 \cdot 0.5} \\
 & = \frac{1}{3}
\end{aligned}
$$

Armed with that, we calculate the probability of a second twin (assuming independance between births):

$$
\begin{aligned}
P(T, T | T) & = P(T | A) \cdot P(A | T) + P(T | B) \cdot P(B | T) \\
 & = 0.1 \cdot \frac{1}{3} + 0.2 \cdot \frac{2}{3} \\
 & = \frac{1}{6}
\end{aligned}
$$

```python
p_twins = [0.1, 0.2]
n_samples = 1_000

second_birth_is_twins = [jnp.nan] * n_samples

with numpyro.handlers.seed(rng_seed=seed):
    i = 0
    while i < n_samples:
        species = numpyro.sample("u", dist.Categorical(jnp.array([0.5, 0.5])))
        first_birth_is_twins = numpyro.sample(
            "first_birth_is_twins", dist.Bernoulli(probs=p_twins[species])
        )
        if not first_birth_is_twins:
            continue
        second_birth_is_twins[i] = numpyro.sample(
            "second_birth_is_twins", dist.Bernoulli(probs=p_twins[species])
        )
        i += 1
print(f"P(T, T| T) = {jnp.array(second_birth_is_twins).mean():.2f}.")
```

```python
p_twins = [0.1, 0.2]
n_samples = 5_000
rng = np.random.default_rng(seed=seed)
second_birth_is_twins = [jnp.nan] * n_samples
i = 0
while i < n_samples:
    species = rng.choice([0, 1])
    first_birth_is_twins = stats.bernoulli.rvs(p=p_twins[species], random_state=rng)
    if not first_birth_is_twins:
        continue
    second_birth_is_twins[i] = stats.bernoulli.rvs(p=p_twins[species], random_state=rng)
    i += 1
print(f"P(T, T| T) = {jnp.array(second_birth_is_twins).mean():.2f}.")
```

### 2H2


As per above $P(A) = 1/3$

```python
p_twins = [0.1, 0.2]
n_samples = 1_000

is_species_a = [jnp.nan] * n_samples
with numpyro.handlers.seed(rng_seed=seed):
    i = 0
    while i < n_samples:
        species = numpyro.sample("u", dist.Categorical(jnp.array([0.5, 0.5])))
        first_birth_is_twins = numpyro.sample(
            "first_birth_is_twins", dist.Bernoulli(probs=p_twins[species])
        )
        if not first_birth_is_twins:
            continue
        is_species_a[i] = 1 - species
        i += 1
print(f"P(A) = {jnp.array(is_species_a).mean():.2f}.")
```

```python
p_twins = [0.1, 0.2]
n_samples = 2_500
rng = np.random.default_rng(seed=seed)
is_species_a = [jnp.nan] * n_samples
i = 0
j = 0
while i < n_samples:
    j += 1
    species = rng.choice([0, 1])
    first_birth_is_twins = stats.bernoulli.rvs(p=p_twins[species], random_state=rng)
    if not first_birth_is_twins:
        continue
    is_species_a[i] = species == 0
    i += 1
print(f"P(A) = {jnp.array(is_species_a).mean():.2f}.")
```

### 2H3


$$
\begin{aligned}
P(A|T') & = \frac{P(T'|A) \cdot P(A)}{P(T')} \\ 
 & = \frac{P(T'|A) \cdot P(A)}{P(T'|A) \cdot P(A) + P(T'|B) \cdot P(B)} \\
 & = \frac{0.9 \cdot 1/3}{0.9 \cdot 1/3 + 0.8 \cdot 2/3} \\
 & = \frac{0.9}{2.5} \\
 & = 0.36
\end{aligned}
$$

```python
p_twins = [0.1, 0.2]
n_samples = 1_000

is_species_a = [jnp.nan] * n_samples
with numpyro.handlers.seed(rng_seed=seed):
    i = 0
    while i < n_samples:
        species = numpyro.sample("u", dist.Categorical(jnp.array([0.5, 0.5])))
        first_birth_is_twins = numpyro.sample(
            "first_birth_is_twins", dist.Bernoulli(probs=p_twins[species])
        )
        if not first_birth_is_twins:
            continue

        second_birth_is_twins = numpyro.sample(
            "second_birth_is_twins", dist.Bernoulli(probs=p_twins[species])
        )
        if second_birth_is_twins:
            continue

        is_species_a[i] = 1 - species
        i += 1
print(f"P(A | T') = {jnp.array(is_species_a).mean():.2f}")
```

```python
p_twins = [0.1, 0.2]
n_samples = 1_000
rng = np.random.default_rng(seed=seed)
is_species_a = [jnp.nan] * n_samples
i = 0
while i < n_samples:
    species = rng.choice([0, 1])
    first_birth_is_twins = stats.bernoulli.rvs(p=p_twins[species], random_state=rng)
    if not first_birth_is_twins:
        continue
    second_birth_is_twins = stats.bernoulli.rvs(p=p_twins[species], random_state=rng)
    if second_birth_is_twins:
        continue
    is_species_a[i] = species == 0
    i += 1
print(f"P(A | T') = {jnp.array(is_species_a).mean():.2f}")
```

### 2H4

<!-- #region -->
$$
\hat{A} \sim  \text{Test Predicts Species A}
$$

Ignoring the birth data:

$$
\begin{aligned}
P(A|\hat{A}) & = \frac{P(\hat{A}|A) \cdot P(A)}{P(\hat{A})} \\ 
 & = \frac{P(\hat{A}|A) \cdot P(A)}{P(\hat{A}|A) \cdot P(A) + P(\hat{A}|B) \cdot P(B)} \\
 & = \frac{0.8 \cdot 0.5}{0.8 \cdot 0.5 + (1 - 0.65) \cdot 0.5} \\
 & = \frac{0.4}{2.3} \\
 & = 0.6957
\end{aligned}
$$


With the birth data, our prior is now $P(A) = 0.36$:

$$
\begin{aligned}
P(A|\hat{A}) & = \frac{0.8 \cdot 0.36}{0.8 \cdot 0.36 + (1 - 0.65) \cdot 0.64} \\
 & = 0.5625
\end{aligned}
$$
<!-- #endregion -->

```python
no_births_analytical = (0.8 * 0.5) / (0.8 * 0.5 + (1 - 0.65) * 0.5)
with_births_analytical = (0.8 * 0.36) / (0.8 * 0.36 + (1 - 0.65) * 0.64)
```

```python
p_twins = [0.1, 0.2]
p_test_says_a = [0.8, 1 - 0.65]
n_samples = 1_000

is_species_a_no_births = []
is_species_a_with_births = []
with numpyro.handlers.seed(rng_seed=seed):
    i_no_births = 0
    i_with_births = 0
    while min(i_no_births, i_with_births) < n_samples:
        species = numpyro.sample("u", dist.Categorical(jnp.array([0.5, 0.5])))

        test_says_a = numpyro.sample(
            "test_says_a", dist.Bernoulli(probs=p_test_says_a[species])
        )
        if test_says_a:
            is_species_a_no_births.append(1 - species)
            i_no_births += 1
        else:
            continue

        total_twin_births = numpyro.sample(
            "total_twin_births", dist.Binomial(total_count=2, probs=p_twins[species])
        )
        if total_twin_births != 1:
            continue

        is_species_a_with_births.append(1 - species)
        i_with_births += 1
no_births_mean = jnp.array(is_species_a_no_births).mean()
no_births_z = jnp.array(is_species_a_no_births).std() * jnp.sqrt(n_samples)
no_births_t = abs(no_births_mean - no_births_analytical) / no_births_z

with_births_mean = jnp.array(is_species_a_with_births).mean()
with_births_z = jnp.array(is_species_a_with_births).std() * jnp.sqrt(n_samples)
with_births_t = abs(with_births_mean - with_births_analytical) / with_births_z
print(
    f"P(A|A_hat) = {no_births_mean:.4f} (t-stat: {no_births_t:.4f})\n"
    f"P(A|T, T', A_hat) =  {with_births_mean:.4f} (t-stat {with_births_t:.4f})"
)
```

```python
p_twins = [0.1, 0.2]
p_test_says_a = [0.8, 1 - 0.65]
n_samples = 2_000

rng = np.random.default_rng(seed=seed)
is_species_a_no_births = []
is_species_a_with_births = []

i_no_births = 0
i_with_births = 0
while min(i_no_births, i_with_births) < n_samples:
    species = rng.choice([0, 1])
    test_says_a = stats.bernoulli.rvs(p=p_test_says_a[species])
    if test_says_a:
        is_species_a_no_births.append(1 - species)
        i_no_births += 1
    else:
        continue

    first_birth_is_twins = stats.bernoulli.rvs(p=p_twins[species])
    if not first_birth_is_twins:
        continue
    second_birth_is_twins = stats.bernoulli.rvs(p=p_twins[species])
    if second_birth_is_twins:
        continue
    is_species_a_with_births.append(1 - species)
    i_with_births += 1

no_births_mean = jnp.array(is_species_a_no_births).mean()
no_births_z = jnp.array(is_species_a_no_births).std() * jnp.sqrt(n_samples)
no_births_t = abs(no_births_mean - no_births_analytical) / no_births_z

with_births_mean = jnp.array(is_species_a_with_births).mean()
with_births_z = jnp.array(is_species_a_with_births).std() * jnp.sqrt(n_samples)
with_births_t = abs(with_births_mean - with_births_analytical) / with_births_z
print(
    f"P(A|A_hat) = {no_births_mean:.4f} (t-stat: {no_births_t:.4f})\n"
    f"P(A|T, T', A_hat) =  {with_births_mean:.4f} (t-stat {with_births_t:.4f})"
)
```

```python

```
