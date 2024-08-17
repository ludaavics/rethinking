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

# Chapter 4: Geocentric Models

```python
%load_ext jupyter_black
```

```python
import jax
import arviz as az
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
from numpyro.infer import Predictive, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoLaplaceApproximation
import numpyro.distributions as dist
import pandas as pd
import plotly
import plotly.graph_objects as go
import plotly.io as pio
from scipy import stats, optimize
from scipy.interpolate import BSpline

pd.options.plotting.backend = "plotly"

seed = 84735
pio.templates.default = "plotly_white"
rng = np.random.default_rng(seed=seed)
jrng = jax.random.key(seed)
```

## Code
### Code 4.1

```python
steps = dist.Uniform(low=-1, high=1).sample(jrng, sample_shape=(1_000, 16))
```

```python
steps = -1 + 2 * stats.uniform.rvs(size=(1_000, 16))
```

```python
pd.DataFrame(steps.sum(axis=1)).plot(kind="hist")
```

### Code 4.2

```python
steps = dist.Uniform(low=0, high=0.1).sample(jrng, sample_shape=(12,))
jnp.prod(1 + steps)
```

### Code 4.3

```python
growth = jnp.prod(
    1 + dist.Uniform(low=0, high=0.1).sample(jrng, sample_shape=(10_000, 12)), axis=1
)
az.plot_density({"growth": growth}, hdi_prob=1)
```

### Code 4.4

```python
big = jnp.prod(
    1 + dist.Uniform(low=0, high=0.5).sample(jrng, sample_shape=(10_000, 12)), axis=1
)
small = jnp.prod(
    1 + dist.Uniform(low=0, high=0.1).sample(jrng, sample_shape=(10_000, 12)), axis=1
)
az.plot_density({"big": big, "small": small}, hdi_prob=1)
```

### Code 4.5

```python
log_big = jnp.log(
    jnp.prod(
        1 + dist.Uniform(low=0, high=0.5).sample(jrng, sample_shape=(10_000, 12)),
        axis=1,
    )
)
ax = az.plot_density({"log_big": log_big}, hdi_prob=1)
x = jnp.sort(log_big)
gaussian = jnp.exp(dist.Normal(jnp.mean(x), jnp.std(x)).log_prob(x))
ax[0][0].plot(x, gaussian, "--")
```

### Code 4.6

```python
w = 6
n = 9
p_grid = jnp.linspace(0, 1, 100)
posterior = jnp.exp(dist.Binomial(total_count=n, probs=p_grid).log_prob(w)) * jnp.exp(
    dist.Uniform(low=0, high=1).log_prob(p_grid)
)
posterior /= posterior.sum()
pd.DataFrame(posterior, index=p_grid).plot()
```

### Code 4.7

```python
df = pd.read_csv("../data/Howell1.csv", sep=";")
```

### Code 4.8

```python
df
```

### Code 4.9

```python
df.describe()
```

### Code 4.10

```python
df["height"]
```

### Code 4.11

```python
df2 = df[df["age"] >= 18]
```

### Code 4.12

```python
x = jnp.linspace(100, 250)
pd.DataFrame(stats.norm.pdf(x, loc=178, scale=20), index=x).plot()
```

### Code 4.13

```python
x = jnp.linspace(-10, 60)
pd.DataFrame(stats.uniform.pdf(x, loc=0, scale=50), index=x).plot()
```

### Code 4.14

```python
_, jrng = jax.random.split(jrng)
sample_mu = dist.Normal(loc=178, scale=20).sample(jrng, (10_000,))
_, jrng = jax.random.split(jrng)
sample_sigma = dist.Uniform(low=0, high=50).sample(jrng, (10_000,))
_, jrng = jax.random.split(jrng)
prior_predictive = dist.Normal(loc=sample_mu, scale=sample_sigma).sample(jrng)
az.plot_density({"Prior Predictive Distribution": prior_predictive}, hdi_prob=1)
```

```python
def adult_height_model(priors, heights):
    mu = numpyro.sample(
        "mu", dist.Normal(loc=priors["mu_mean"], scale=priors["mu_scale"])
    )
    sigma = numpyro.sample(
        "sigma", dist.Uniform(low=priors["sigma_low"], high=priors["sigma_high"])
    )
    numpyro.sample("height", dist.Normal(loc=mu, scale=sigma), obs=heights)


prior_samples = numpyro.infer.Predictive(adult_height_model, num_samples=10_000)(
    jrng,
    priors={"mu_mean": 178, "mu_scale": 20, "sigma_low": 0, "sigma_high": 50},
    heights=None,
)
az.plot_density(prior_samples, hdi_prob=1)
```

### Code 4.15

```python
prior_samples = numpyro.infer.Predictive(adult_height_model, num_samples=10_000)(
    jrng,
    priors={"mu_mean": 178, "mu_scale": 100, "sigma_low": 0, "sigma_high": 50},
    heights=None,
)
az.plot_density(prior_samples, hdi_prob=1)
```

### Code 4.16

```python
mu_list = jnp.linspace(start=150, stop=160, num=100)
sigma_list = jnp.linspace(start=7, stop=9, num=100)
mesh = jnp.meshgrid(mu_list, sigma_list)
posterior = {"mu": mesh[0].reshape(-1), "sigma": mesh[1].reshape(-1)}
posterior["LL"] = jax.vmap(
    lambda mu, sigma: jnp.sum(dist.Normal(mu, sigma).log_prob(df2.height.values))
)(posterior["mu"], posterior["sigma"])
logprob_mu = dist.Normal(178, 20).log_prob(posterior["mu"])
logprob_sigma = dist.Uniform(0, 50).log_prob(posterior["sigma"])
posterior["prob"] = posterior["LL"] + logprob_mu + logprob_sigma
posterior["prob"] = jnp.exp(posterior["prob"] - jnp.max(posterior["prob"]))
```

### Code 4.17

```python
plt.contour(
    posterior["mu"].reshape(100, 100),
    posterior["sigma"].reshape(100, 100),
    posterior["prob"].reshape(100, 100),
)
plt.show()
```

### Code 4.18

```python
plt.imshow(
    posterior["prob"].reshape(100, 100),
    origin="lower",
    extent=(150, 160, 7, 9),
    aspect="auto",
)
plt.show()
```

### Code 4.19

```python
prob = posterior["prob"] / jnp.sum(posterior["prob"])
sample_rows = dist.Categorical(probs=prob).sample(jrng, (int(1e4),))
sample_mu = posterior["mu"][sample_rows]
sample_sigma = posterior["sigma"][sample_rows]
```

```python
pd.DataFrame({"mu": sample_mu, "sigma": sample_sigma}).plot(
    kind="scatter", x="mu", y="sigma", backend="matplotlib", alpha=0.1
)
```

### Code 4.20

```python
az.plot_kde(sample_mu)
```

```python
az.plot_kde(sample_sigma)
```

### Code 4.22

```python
print(f"mu 89% HPDI: {numpyro.diagnostics.hpdi(sample_mu, prob=0.89)}")
print(f"sigma 89% HPDI: {numpyro.diagnostics.hpdi(sample_sigma, prob=0.89)}")
```

### Code 4.23

```python
df3 = df2["height"].sample(n=20, random_state=seed)
```

### Code 4.24

```python
mu_list = jnp.linspace(start=100, stop=170, num=200)
sigma_list = jnp.linspace(start=4, stop=20, num=200)
mesh = jnp.meshgrid(mu_list, sigma_list)
posterior2 = {"mu": mesh[0].reshape(-1), "sigma": mesh[1].reshape(-1)}
posterior2["LL"] = jax.vmap(
    lambda mu, sigma: jnp.sum(dist.Normal(mu, sigma).log_prob(df3.values))
)(posterior2["mu"], posterior2["sigma"])
logprob_mu = dist.Normal(178, 20).log_prob(posterior2["mu"])
logprob_sigma = dist.Uniform(0, 50).log_prob(posterior2["sigma"])
posterior2["prob"] = posterior2["LL"] + logprob_mu + logprob_sigma
posterior2["prob"] = jnp.exp(posterior2["prob"] - jnp.max(posterior2["prob"]))
prob = posterior2["prob"] / jnp.sum(posterior2["prob"])
sample2_rows = dist.Categorical(probs=prob).sample(jrng, (int(1e4),))
sample2_mu = posterior2["mu"][sample2_rows]
sample2_sigma = posterior2["sigma"][sample2_rows]
plt.scatter(sample2_mu, sample2_sigma, s=64, alpha=0.1, edgecolor="none")
plt.show()
```

### Code 4.25

```python
az.plot_kde(sample2_mu)
x = jnp.sort(sample2_mu)
plt.plot(x, jnp.exp(dist.Normal(jnp.mean(x), jnp.std(x)).log_prob(x)), "--")
plt.show()
```

```python
az.plot_kde(sample2_sigma)
x = jnp.sort(sample2_sigma)
plt.plot(x, jnp.exp(dist.Normal(jnp.mean(x), jnp.std(x)).log_prob(x)), "--")
plt.show()
```

### Code 4.26

```python
Howell1 = pd.read_csv("../data/Howell1.csv", sep=";")
df = Howell1
df2 = df[df["age"] >= 18]
```

### Code 4.27

```python
def adult_height_model(height, priors):
    mu = numpyro.sample(
        "mu", dist.Normal(loc=priors["mu_mean"], scale=priors["mu_scale"])
    )
    sigma = numpyro.sample(
        "sigma", dist.Uniform(low=priors["sigma_low"], high=priors["sigma_high"])
    )
    numpyro.sample("height", dist.Normal(loc=mu, scale=sigma), obs=height)
```

### Code 4.28

```python
adult_height_laplace_model = AutoLaplaceApproximation(adult_height_model)
adult_height_svi = SVI(
    model=adult_height_model,
    guide=adult_height_laplace_model,
    optim=numpyro.optim.Adam(step_size=0.1),
    loss=Trace_ELBO(),
    height=df2.height.values,
    priors={"mu_mean": 178, "mu_scale": 20, "sigma_low": 0, "sigma_high": 50},
).run(jrng, 5_000)
adult_height_svi.params
```

### Code 4.29

```python
_, _jrng = jax.random.split(jrng)
samples = adult_height_laplace_model.sample_posterior(
    _jrng, adult_height_svi.params, sample_shape=(1000,)
)
numpyro.diagnostics.print_summary(samples, 0.89, False)
```

### Code 4.30

```python
init_fn_values = {"mu": df2["height"].mean(), "sigma": df2["height"].std()}
adult_height_laplace_model = AutoLaplaceApproximation(
    adult_height_model, init_loc_fn=numpyro.infer.init_to_value(values=init_fn_values)
)
adult_height_svi = SVI(
    model=adult_height_model,
    guide=adult_height_laplace_model,
    optim=numpyro.optim.Adam(step_size=0.01),
    loss=Trace_ELBO(),
    height=df2.height.values,
    priors={"mu_mean": 178, "mu_scale": 20, "sigma_low": 0, "sigma_high": 50},
).run(jrng, 5000)
adult_height_svi.params
```

```python
_, _jrng = jax.random.split(_jrng)
samples = adult_height_laplace_model.sample_posterior(
    _jrng, adult_height_svi.params, sample_shape=(1000,)
)
numpyro.diagnostics.print_summary(samples, 0.89, False)
```

### Code 4.31

```python
adult_height_laplace_model_2 = AutoLaplaceApproximation(adult_height_model)
adult_height_svi_2 = SVI(
    model=adult_height_model,
    guide=adult_height_laplace_model_2,
    optim=numpyro.optim.Adam(1),
    loss=Trace_ELBO(),
    height=df2.height.values,
    priors={"mu_mean": 178, "mu_scale": 0.1, "sigma_low": 0, "sigma_high": 50},
).run(jrng, 2000)
adult_height_svi_2.params
```

```python
_, _jrng = jax.random.split(_jrng)
samples = adult_height_laplace_model_2.sample_posterior(
    _jrng, adult_height_svi_2.params, sample_shape=(1000,)
)
numpyro.diagnostics.print_summary(samples, 0.89, False)
```

### Code 4.32

```python
_, _jrng = jax.random.split(_jrng)
samples = adult_height_laplace_model.sample_posterior(
    _jrng, adult_height_svi.params, sample_shape=(1000,)
)
vcov = pd.DataFrame(jnp.stack(list(samples.values())), index=["mu", "sigma"]).T.cov()
vcov
```

### Code 4.33

```python
print(jnp.diagonal(vcov.values) ** 0.5)
print(
    vcov.values
    / jnp.sqrt(jnp.outer(jnp.diagonal(vcov.values), jnp.diagonal(vcov.values)))
)
```

### Code 4.34

```python
_, _jrng = jax.random.split(_jrng)
samples = pd.DataFrame(
    adult_height_laplace_model.sample_posterior(
        _jrng, adult_height_svi.params, sample_shape=(10_000,)
    )
)
samples.head()
```

### Code 4.35

```python
samples.describe()
```

### Code 4.36

```python
# don't know how to interpret adult_height_svi.params
# feels like it should be the MAP of sigma, but clearly not (it has wrong sign)
# below is kinda dumb workaround (get the samples from .sample_posterior in order to generate samples)
samples = pd.DataFrame(
    adult_height_laplace_model.sample_posterior(
        _jrng, adult_height_svi.params, sample_shape=(10_000,)
    )
)
vcov = samples.cov()
samples = pd.DataFrame(
    dist.MultivariateNormal(
        loc=samples.mean().values, covariance_matrix=vcov.values
    ).sample(_jrng, sample_shape=(10_000,)),
    columns=["mu", "sigma"],
)
samples
```

### Code 4.37

```python
df2.plot(kind="scatter", x="weight", y="height", backend="matplotlib")
```

### Code 4.38

```python
_, _jrng = jax.random.split(_jrng)
a = dist.Normal(loc=178, scale=20).sample(jrng, sample_shape=(100,))
_, _jrng = jax.random.split(_jrng)
b = dist.Normal(loc=0, scale=10).sample(_jrng, sample_shape=(100,))
```

```python
def adult_height_model(
    height, weight, *, average_weight, alpha_prior, beta_prior, sigma_prior
):
    alpha = numpyro.sample("alpha", dist.Normal(**alpha_prior))
    beta = numpyro.sample("beta", dist.Normal(**beta_prior))
    sigma = numpyro.sample("sigma", dist.Uniform(**sigma_prior))
    forecast = numpyro.deterministic(
        "forecast", alpha + beta * (weight - average_weight)
    )
    height = numpyro.sample(
        "height", dist.Normal(loc=forecast, scale=sigma), obs=height
    )
    return height
```

```python
prior_samples = numpyro.infer.Predictive(adult_height_model, num_samples=100)(
    jrng,
    height=None,
    weight=df2["weight"].values,
    average_weight=df2["weight"].mean(),
    alpha_prior={"loc": 178, "scale": 20},
    beta_prior={"loc": 0, "scale": 10},
    sigma_prior={"low": 0, "high": 50},
)
```

### Code 4.39

```python
def plot_prior_lines(a, b):
    plt.subplot(
        xlim=(df2.weight.min(), df2.weight.max()),
        ylim=(-100, 400),
        xlabel="weight",
        ylabel="height",
    )
    plt.axhline(y=0, c="k", ls="--")
    plt.axhline(y=272, c="k", ls="-", lw=0.5)
    plt.title("b ~ Normal(0, 10)")
    xbar = df2.weight.mean()
    x = jnp.linspace(df2.weight.min(), df2.weight.max(), 101)
    for i in range(100):
        plt.plot(
            x,
            a[i] + b[i] * (x - xbar),
            "k",
            alpha=0.2,
        )
    plt.show()


plot_prior_lines(a=prior_samples["alpha"], b=prior_samples["beta"])
```

### Code 4.40

```python
b = dist.LogNormal(loc=0, scale=1).sample(jrng, sample_shape=(10_000,))
az.plot_kde(b)
```

### Code 4.41

```python
def adult_height_model(
    height, weight, *, average_weight, alpha_prior, beta_prior, sigma_prior
):
    alpha = numpyro.sample("alpha", dist.Normal(**alpha_prior))
    beta = numpyro.sample("beta", dist.LogNormal(**beta_prior))
    sigma = numpyro.sample("sigma", dist.Uniform(**sigma_prior))
    forecast = numpyro.deterministic(
        "forecast", alpha + beta * (weight - average_weight)
    )
    height = numpyro.sample(
        "height", dist.Normal(loc=forecast, scale=sigma), obs=height
    )
    return height
```

```python
prior_samples = numpyro.infer.Predictive(adult_height_model, num_samples=100)(
    jrng,
    height=None,
    weight=df2["weight"].values,
    average_weight=df2["weight"].mean(),
    alpha_prior={"loc": 178, "scale": 20},
    beta_prior={"loc": 0, "scale": 1},
    sigma_prior={"low": 0, "high": 50},
)
plot_prior_lines(a=prior_samples["alpha"], b=prior_samples["beta"])
```

### Code 4.42

```python
Howell1 = pd.read_csv("../data/Howell1.csv", sep=";")
df = Howell1
df2 = df[df["age"] >= 18]
average_weight = df2["weight"].mean()


def adult_height_model(
    weight, *, average_weight, alpha_prior, beta_prior, sigma_prior, height=None
):
    alpha = numpyro.sample("alpha", dist.Normal(**alpha_prior))
    beta = numpyro.sample("beta", dist.LogNormal(**beta_prior))
    sigma = numpyro.sample("sigma", dist.Uniform(**sigma_prior))
    # mu = numpyro.deterministic("mu", alpha + beta * (weight - average_weight))
    mu = alpha + beta * (weight - average_weight)
    height = numpyro.sample("height", dist.Normal(loc=mu, scale=sigma), obs=height)
    return height


guide = AutoLaplaceApproximation(adult_height_model)
svi = SVI(
    model=adult_height_model,
    guide=guide,
    optim=numpyro.optim.Adam(step_size=0.1),
    loss=Trace_ELBO(),
    weight=df2["weight"].values,
    average_weight=average_weight,
    alpha_prior={"loc": 178, "scale": 20},
    beta_prior={"loc": 0, "scale": 1},
    sigma_prior={"low": 0, "high": 50},
    height=df2["height"].values,
).run(jrng, 5_000)

_, _jrng = jax.random.split(_jrng)
posterior_samples = guide.sample_posterior(_jrng, svi.params, sample_shape=(1000,))
numpyro.diagnostics.print_summary(posterior_samples, 0.89, False)
```

### Code 4.43

```python
def adult_height_model(
    weight, *, average_weight, alpha_prior, beta_prior, sigma_prior, height=None
):
    alpha = numpyro.sample("alpha", dist.Normal(**alpha_prior))
    beta = numpyro.sample("beta", dist.LogNormal(**beta_prior))
    sigma = numpyro.sample("sigma", dist.Uniform(**sigma_prior))
    # mu = numpyro.deterministic("mu", alpha + beta * (weight - average_weight))
    mu = alpha + beta * (weight - average_weight)
    height = numpyro.sample("height", dist.Normal(loc=mu, scale=sigma), obs=height)
    return height


guide = AutoLaplaceApproximation(adult_height_model)
svi = SVI(
    model=adult_height_model,
    guide=guide,
    optim=numpyro.optim.Adam(step_size=0.1),
    loss=Trace_ELBO(),
    weight=df2["weight"].values,
    average_weight=average_weight,
    alpha_prior={"loc": 178, "scale": 20},
    beta_prior={"loc": 0, "scale": 1},
    sigma_prior={"low": 0, "high": 50},
    height=df2["height"].values,
).run(jrng, 5_000)

_, _jrng = jax.random.split(_jrng)
posterior_samples = guide.sample_posterior(_jrng, svi.params, sample_shape=(1000,))
numpyro.diagnostics.print_summary(posterior_samples, 0.89, False)
```

### Code 4.44

```python
numpyro.diagnostics.print_summary(posterior_samples, 0.89, False)
```

### Code 4.45

```python
pd.DataFrame(posterior_samples).cov().round(3)
```

### Code 4.46

```python
fig = pd.DataFrame(df2[["weight", "height"]]).plot(
    kind="scatter",
    x="weight",
    y="height",
)
x = jnp.linspace(df2["weight"].min() * 0.95, df2["weight"].max() * 1.05)
y = posterior_samples["alpha"].mean() + posterior_samples["beta"].mean() * (
    x - average_weight
)
fig.add_trace(go.Scatter(x=x, y=y, name="posterior_mean"))
```

### Code 4.47

```python
pd.DataFrame(posterior_samples).head()
```

### Code 4.48

```python
def adult_height_model(
    weight, *, average_weight, alpha_prior, beta_prior, sigma_prior, height=None
):
    alpha = numpyro.sample("alpha", dist.Normal(**alpha_prior))
    beta = numpyro.sample("beta", dist.LogNormal(**beta_prior))
    sigma = numpyro.sample("sigma", dist.Uniform(**sigma_prior))
    # mu = numpyro.deterministic("mu", alpha + beta * (weight - average_weight))
    mu = alpha + beta * (weight - average_weight)
    height = numpyro.sample("height", dist.Normal(loc=mu, scale=sigma), obs=height)
    return height


guide = AutoLaplaceApproximation(adult_height_model)
svi = SVI(
    model=adult_height_model,
    guide=guide,
    optim=numpyro.optim.Adam(step_size=0.1),
    loss=Trace_ELBO(),
    weight=df2["weight"].values[:10],
    average_weight=average_weight,
    alpha_prior={"loc": 178, "scale": 20},
    beta_prior={"loc": 0, "scale": 1},
    sigma_prior={"low": 0, "high": 50},
    height=df2["height"].values[:10],
).run(jrng, 5_000)
```

### Code 4.49

```python
_, _jrng = jax.random.split(_jrng)
posterior_samples = guide.sample_posterior(_jrng, svi.params, sample_shape=(20,))
numpyro.diagnostics.print_summary(posterior_samples, 0.89, False)
fig = pd.DataFrame(df2[["weight", "height"]].iloc[:10]).plot(
    kind="scatter",
    x="weight",
    y="height",
)
x = jnp.linspace(df2["weight"].min() * 0.95, df2["weight"].max() * 1.05)
for i in range(20):
    y = posterior_samples["alpha"][i] + posterior_samples["beta"][i] * (
        x - average_weight
    )
    fig.add_trace(
        go.Scatter(x=x, y=y, line={"color": "black"}, opacity=0.3, showlegend=False)
    )
fig
```

### Code 4.50

```python
def adult_height_model(
    weight, *, average_weight, alpha_prior, beta_prior, sigma_prior, height=None
):
    alpha = numpyro.sample("alpha", dist.Normal(**alpha_prior))
    beta = numpyro.sample("beta", dist.LogNormal(**beta_prior))
    sigma = numpyro.sample("sigma", dist.Uniform(**sigma_prior))
    mu = numpyro.deterministic("mu", alpha + beta * (weight - average_weight))
    height = numpyro.sample("height", dist.Normal(loc=mu, scale=sigma), obs=height)
    return height


guide = AutoLaplaceApproximation(adult_height_model)
svi = SVI(
    model=adult_height_model,
    guide=guide,
    optim=numpyro.optim.Adam(step_size=0.1),
    loss=Trace_ELBO(),
    weight=df2["weight"].values,
    average_weight=average_weight,
    alpha_prior={"loc": 178, "scale": 20},
    beta_prior={"loc": 0, "scale": 1},
    sigma_prior={"low": 0, "high": 50},
    height=df2["height"].values,
).run(jrng, 5_000)

_, _jrng = jax.random.split(_jrng)
posterior_samples = guide.sample_posterior(_jrng, svi.params, sample_shape=(1000,))
# numpyro.diagnostics.print_summary(posterior_samples, 0.89, False)
```

```python
mu_at_50 = posterior_samples["alpha"] + posterior_samples["beta"] * (
    50 - average_weight
)
```

### Code 4.51

```python
az.plot_kde(mu_at_50, label="mu|weight=50")
```

### Code 4.52

```python
numpyro.diagnostics.hpdi(mu_at_50, prob=0.89)
```

### Code 4.53

```python
mu = pd.DataFrame(posterior_samples["mu"])
mu.columns.name = "training sample"
mu.index.name = "posterior predictive sample"
mu
```

### Code 4.54

```python
_, _jrng = jax.random.split(_jrng)
weight_seq = jnp.arange(start=25, stop=71, step=1)
mu = numpyro.infer.Predictive(guide.model, posterior_samples, return_sites=["mu"])(
    _jrng,
    weight=weight_seq,
    average_weight=average_weight,
    alpha_prior={"loc": 178, "scale": 20},
    beta_prior={"loc": 0, "scale": 1},
    sigma_prior={"low": 0, "high": 50},
    height=None,
)["mu"]
mu = pd.DataFrame(mu, columns=pd.Index(weight_seq, name="weight"))
mu.index.name = "posterior predictive sample"
assert (
    posterior_samples["alpha"][0] + posterior_samples["beta"][0] * (25 - average_weight)
    == mu.iat[0, 0]
)
mu
```

### Code 4.55

```python
df2[["weight", "height"]].plot(kind="scatter", x="weight", y="height", opacity=0)
for i in range(100):
    plt.plot(weight_seq, mu.values[i], "o", c="royalblue", alpha=0.1)
```

### Code 4.56

```python
mu_mean = mu.mean().to_frame().T
mu_mean
```

```python
mu_hpdi = mu.apply(lambda x: numpyro.diagnostics.hpdi(x, prob=0.89))
mu_hpdi
```

### Code 4.57

```python
ax = df2[["weight", "height"]].plot(
    kind="scatter", x="weight", y="height", backend="matplotlib", alpha=0.5
)
plt.plot(weight_seq, mu_mean.T, "k-")
plt.fill_between(weight_seq, mu_hpdi.iloc[0], mu_hpdi.iloc[1], color="k", alpha=0.4)
```

### Code 4.58

```python
posterior_samples = guide.sample_posterior(_jrng, svi.params, sample_shape=(1000,))
posterior_samples.pop("mu")
posterior_samples = pd.DataFrame(posterior_samples)


def mu_link(weight):
    return posterior_samples["alpha"] + posterior_samples["beta"] * (
        weight - average_weight
    )


mu = pd.concat([mu_link(_weight) for _weight in weight_seq], axis=1)
mu.columns = pd.Index(weight_seq, name="weight")
mu.index.name = "posterior predictive sample"
mu
```

### Code 4.59

```python
_, _jrng = jax.random.split(_jrng)
posterior_samples = guide.sample_posterior(_jrng, svi.params, sample_shape=(1000,))
weight_seq = jnp.arange(start=25, stop=71, step=1)
height = numpyro.infer.Predictive(
    guide.model, posterior_samples, return_sites=["height"]
)(
    _jrng,
    weight=weight_seq,
    average_weight=average_weight,
    alpha_prior={"loc": 178, "scale": 20},
    beta_prior={"loc": 0, "scale": 1},
    sigma_prior={"low": 0, "high": 50},
    height=None,
)[
    "height"
]
height = pd.DataFrame(height, columns=pd.Index(weight_seq, name="weight"))
height.index.name = "posterior predictive height sample"
height
```

### Code 4.60

```python
height_hpdi = height.apply(lambda x: numpyro.diagnostics.hpdi(x, prob=0.89))
height_hpdi
```

### Code 4.61

```python
ax = df2[["weight", "height"]].plot(
    kind="scatter", x="weight", y="height", backend="matplotlib", alpha=0.5
)
plt.plot(weight_seq, mu_mean.T, "k-")
plt.fill_between(weight_seq, mu_hpdi.iloc[0], mu_hpdi.iloc[1], color="k", alpha=0.4)
plt.fill_between(
    weight_seq, height_hpdi.iloc[0], height_hpdi.iloc[1], color="k", alpha=0.2
)
```

### Code 4.62

```python
_, _jrng = jax.random.split(_jrng)
posterior_samples = guide.sample_posterior(_jrng, svi.params, sample_shape=(10_000,))
weight_seq = jnp.arange(start=25, stop=71, step=1)
height = numpyro.infer.Predictive(
    guide.model, posterior_samples, return_sites=["height"]
)(
    _jrng,
    weight=weight_seq,
    average_weight=average_weight,
    alpha_prior={"loc": 178, "scale": 20},
    beta_prior={"loc": 0, "scale": 1},
    sigma_prior={"low": 0, "high": 50},
    height=None,
)[
    "height"
]
height = pd.DataFrame(height, columns=pd.Index(weight_seq, name="weight"))
height.index.name = "posterior predictive height sample"
height_hpdi = height.apply(lambda x: numpyro.diagnostics.hpdi(x, prob=0.89))
display(height_hpdi)
ax = df2[["weight", "height"]].plot(
    kind="scatter", x="weight", y="height", backend="matplotlib", alpha=0.5
)
plt.plot(weight_seq, mu_mean.T, "k-")
plt.fill_between(weight_seq, mu_hpdi.iloc[0], mu_hpdi.iloc[1], color="k", alpha=0.4)
plt.fill_between(
    weight_seq, height_hpdi.iloc[0], height_hpdi.iloc[1], color="k", alpha=0.2
)
```

### Code 4.63

```python
posterior_samples = guide.sample_posterior(_jrng, svi.params, sample_shape=(1000,))
posterior_samples.pop("mu")
posterior_samples = pd.DataFrame(posterior_samples)


def sim_height(weight, jrng):
    mu = posterior_samples["alpha"] + posterior_samples["beta"] * (
        weight - average_weight
    )

    return dist.Normal(loc=mu, scale=posterior_samples["sigma"]).sample(
        jrng,
    )


height = []
for _weight in weight_seq:
    _, jrng = jax.random.split(jrng)
    height.append(sim_height(_weight, _jrng))
height = pd.concat(height, axis=1)
height.columns = pd.Index(weight_seq, name="weight")
height.index.name = "posterior predictive sample"
height_hpdi = height.apply(lambda x: numpyro.diagnostics.hpdi(x, prob=0.89))
display(height_hpdi)
ax = df2[["weight", "height"]].plot(
    kind="scatter", x="weight", y="height", backend="matplotlib", alpha=0.5
)
plt.plot(weight_seq, mu_mean.T, "k-")
plt.fill_between(weight_seq, mu_hpdi.iloc[0], mu_hpdi.iloc[1], color="k", alpha=0.4)
plt.fill_between(
    weight_seq, height_hpdi.iloc[0], height_hpdi.iloc[1], color="k", alpha=0.2
)
```

### Code 4.64

```python
df = pd.read_csv("../data/Howell1.csv", sep=";")
df
```

```python
df[["weight", "height"]].plot(
    kind="scatter", x="weight", y="height", backend="matplotlib"
)
```

### Code 4.65

```python
df["weight_s"] = (df["weight"] - df["weight"].mean()) / df["weight"].std()
df["weight_s2"] = df["weight_s"] ** 2


def m4_5(
    weight_s,
    weight_s2,
    *,
    alpha_prior={"loc": 178, "scale": 20},
    beta1_prior={"loc": 0, "scale": 1},
    beta2_prior={"loc": 0, "scale": 1},
    sigma_prior={"low": 0, "high": 50},
    height=None,
):
    alpha = numpyro.sample("alpha", dist.Normal(**alpha_prior))
    beta1 = numpyro.sample("beta1", dist.LogNormal(**beta1_prior))
    beta2 = numpyro.sample("beta2", dist.Normal(**beta2_prior))
    sigma = numpyro.sample("sigma", dist.Uniform(**sigma_prior))
    mu = numpyro.deterministic("mu", alpha + beta1 * weight_s + beta2 * weight_s2)
    height = numpyro.sample("height", dist.Normal(loc=mu, scale=sigma), obs=height)
    return height


guide = AutoLaplaceApproximation(m4_5)
_, _jrng = jax.random.split(_jrng)
svi = SVI(
    model=m4_5,
    guide=guide,
    optim=numpyro.optim.Adam(step_size=0.1),
    loss=Trace_ELBO(),
    weight_s=df["weight_s"].values,
    weight_s2=df["weight_s2"].values,
    height=df["height"].values,
).run(jrng, 5_000)
```

### Code 4.66

```python
_, _jrng = jax.random.split(_jrng)
posterior_samples = guide.sample_posterior(_jrng, svi.params, sample_shape=(1000,))
_posterior_samples = {
    k: posterior_samples[k] for k in posterior_samples if k not in "mu"
}
numpyro.diagnostics.print_summary(_posterior_samples, 0.89, False)
```

### Code 4.67

```python
_, _jrng = jax.random.split(_jrng)
posterior_samples = guide.sample_posterior(_jrng, svi.params, sample_shape=(10_000,))
weight_seq = jnp.linspace(start=-2.2, stop=2, num=50)
weight_seq2 = weight_seq**2
posterior_predictive = numpyro.infer.Predictive(
    guide.model, posterior_samples, return_sites=["mu", "height"]
)(
    _jrng,
    weight_s=weight_seq,
    weight_s2=weight_seq2,
    height=None,
)
mu_posterior_predictive = pd.DataFrame(posterior_predictive["mu"], columns=weight_seq)
height_posterior_predictive = pd.DataFrame(
    posterior_predictive["height"], columns=weight_seq
)
```

```python
mu_mean = mu_posterior_predictive.mean(axis=0).to_frame().T
display(mu_mean)

mu_hpdi = pd.DataFrame(
    numpyro.diagnostics.hpdi(mu_posterior_predictive, prob=0.89), columns=weight_seq
)
display(mu_hpdi)

height_mean = height_posterior_predictive.mean(axis=0).to_frame().T
display(height_mean)

height_hpdi = pd.DataFrame(
    numpyro.diagnostics.hpdi(height_posterior_predictive, prob=0.89), columns=weight_seq
)
display(height_hpdi)
```

### Code 4.68

```python
df[["weight_s", "height"]].plot(
    kind="scatter", x="weight_s", y="height", backend="matplotlib"
)
plt.plot(weight_seq, mu_mean.loc[0, :], "k")
plt.fill_between(weight_seq, mu_hpdi.loc[0, :], mu_hpdi.loc[1, :], color="k", alpha=0.5)
plt.fill_between(
    weight_seq, height_hpdi.loc[0, :], height_hpdi.loc[1, :], color="k", alpha=0.2
)
plt.show()
```

### Code 4.69

```python
df["weight_s3"] = df["weight_s"] ** 3


def m4_6(
    weight_s,
    weight_s2,
    weight_s3,
    *,
    alpha_prior={"loc": 178, "scale": 20},
    beta1_prior={"loc": 0, "scale": 1},
    beta2_prior={"loc": 0, "scale": 10},
    beta3_prior={"loc": 0, "scale": 10},
    sigma_prior={"low": 0, "high": 50},
    height=None,
):
    alpha = numpyro.sample("alpha", dist.Normal(**alpha_prior))
    beta1 = numpyro.sample("beta1", dist.LogNormal(**beta1_prior))
    beta2 = numpyro.sample("beta2", dist.Normal(**beta2_prior))
    beta3 = numpyro.sample("beta3", dist.Normal(**beta3_prior))
    sigma = numpyro.sample("sigma", dist.Uniform(**sigma_prior))
    mu = numpyro.deterministic(
        "mu", alpha + beta1 * weight_s + beta2 * weight_s2 + beta3 * weight_s3
    )
    height = numpyro.sample("height", dist.Normal(loc=mu, scale=sigma), obs=height)
    return height


guide = AutoLaplaceApproximation(m4_6)
_, _jrng = jax.random.split(_jrng)
svi = SVI(
    model=m4_6,
    guide=guide,
    optim=numpyro.optim.Adam(step_size=0.1),
    loss=Trace_ELBO(),
    weight_s=df["weight_s"].values,
    weight_s2=df["weight_s2"].values,
    weight_s3=df["weight_s3"].values,
    height=df["height"].values,
).run(jrng, 5_000)
```

```python
_, _jrng = jax.random.split(_jrng)
posterior_samples = guide.sample_posterior(_jrng, svi.params, sample_shape=(10_000,))
weight_seq = jnp.linspace(start=-2.2, stop=2, num=50)
weight_seq2 = weight_seq**2
weight_seq3 = weight_seq**3
posterior_predictive = numpyro.infer.Predictive(
    guide.model, posterior_samples, return_sites=["mu", "height"]
)(
    _jrng,
    weight_s=weight_seq,
    weight_s2=weight_seq2,
    weight_s3=weight_seq3,
    height=None,
)
mu_posterior_predictive = pd.DataFrame(posterior_predictive["mu"], columns=weight_seq)
height_posterior_predictive = pd.DataFrame(
    posterior_predictive["height"], columns=weight_seq
)

mu_mean = mu_posterior_predictive.mean(axis=0).to_frame().T
mu_hpdi = pd.DataFrame(
    numpyro.diagnostics.hpdi(mu_posterior_predictive, prob=0.89), columns=weight_seq
)
height_mean = height_posterior_predictive.mean(axis=0).to_frame().T
height_hpdi = pd.DataFrame(
    numpyro.diagnostics.hpdi(height_posterior_predictive, prob=0.89), columns=weight_seq
)
df[["weight_s", "height"]].plot(
    kind="scatter", x="weight_s", y="height", backend="matplotlib"
)
plt.plot(weight_seq, mu_mean.loc[0, :], "k")
plt.fill_between(weight_seq, mu_hpdi.loc[0, :], mu_hpdi.loc[1, :], color="k", alpha=0.5)
plt.fill_between(
    weight_seq, height_hpdi.loc[0, :], height_hpdi.loc[1, :], color="k", alpha=0.2
)
plt.show()
```

### Code 4.70

```python
df.plot(kind="scatter", x="weight_s", y="height", backend="matplotlib", xticks=[])
```

### Code 4.71

```python
df.plot(kind="scatter", x="weight", y="height", backend="matplotlib")
```

### Code 4.72

```python
df = pd.read_csv("../data/cherry_blossoms.csv", sep=";")
df.describe()
```

```python
df["temp"].plot(backend="matplotlib")
```

### Code 4.73

```python
df2 = df.dropna(subset=["temp"])
num_knots = 15
knots_list = jnp.quantile(
    df["year"].values, jnp.linspace(start=0, stop=1, num=num_knots)
)
```

### Code 4.74

```python
degree = 3
knots = jnp.pad(knots_list, (degree, degree), mode="edge")
B = BSpline(knots, jnp.identity(num_knots + 2), k=degree)(df2.year.values)
```

### Code 4.75

```python
plt.subplot(
    xlim=(df2.year.min(), df2.year.max()),
    ylim=(0, 1),
    xlabel="year",
    ylabel="basis value",
)
for i in range(B.shape[1]):
    plt.plot(df2.year, B[:, i], "k", alpha=0.5)
```

### Code 4.76

```python
def m4_7(
    B,
    *,
    alpha_prior={"loc": 6, "scale": 10},
    weight_prior={"loc": 0, "scale": 1},
    sigma_prior={"rate": 1},
    temp=None,
):
    alpha = numpyro.sample("alpha", dist.Normal(**alpha_prior))
    weight = numpyro.sample(
        "weight", dist.Normal(**weight_prior), sample_shape=B.shape[1:]
    )
    mu = numpyro.deterministic("mu", alpha + jnp.dot(B, weight))
    sigma = numpyro.sample("sigma", dist.Exponential(**sigma_prior))
    temp = numpyro.sample("temp", dist.Normal(loc=mu, scale=sigma), obs=temp)
    return temp
```

```python
guide = AutoLaplaceApproximation(
    m4_7, init_loc_fn=numpyro.infer.init_to_value(values={"w": jnp.zeros(B.shape[1])})
)
_, _jrng = jax.random.split(_jrng)
svi = SVI(
    model=m4_7,
    guide=guide,
    optim=numpyro.optim.Adam(step_size=0.5),
    loss=Trace_ELBO(),
    B=B,
    temp=df2["temp"].values,
).run(jrng, 10_000)
```

```python
_, _jrng = jax.random.split(_jrng)
posterior_samples = guide.sample_posterior(_jrng, svi.params, sample_shape=(10_000,))
_posterior_samples = {k: v for k, v in posterior_samples.items() if k != "mu"}
numpyro.diagnostics.print_summary(_posterior_samples, prob=0.89, group_by_chain=False)
```

### Code 4.77

```python
weight_mean = posterior_samples["weight"].mean(axis=0)
plt.subplot(
    xlim=(df2.year.min(), df2.year.max()),
    ylim=(-2, 2),
    xlabel="year",
    ylabel="basis * weight",
)
for i in range(B.shape[1]):
    plt.plot(df2["year"], weight_mean[i] * B[:, i], "k", alpha=0.2)
```

### Code 4.78

```python
mu_hpdi = pd.DataFrame(numpyro.diagnostics.hpdi(posterior_samples["mu"], prob=0.89))
df2.plot(kind="scatter", x="year", y="temp", backend="matplotlib", figsize=(10, 6))
plt.fill_between(
    df2["year"], mu_hpdi.loc[0, :], mu_hpdi.loc[1, :], color="k", alpha=0.5
)
```

### Code 4.79

```python
def m4_7_alt(
    B,
    *,
    alpha_prior={"loc": 6, "scale": 10},
    weight_prior={"loc": 0, "scale": 1},
    sigma_prior={"rate": 1},
    temp=None,
):
    alpha = numpyro.sample("alpha", dist.Normal(**alpha_prior))
    weight = numpyro.sample(
        "weight", dist.Normal(**weight_prior), sample_shape=B.shape[1:]
    )
    mu = numpyro.deterministic("mu", alpha + jnp.sum(B * weight, axis=-1))
    sigma = numpyro.sample("sigma", dist.Exponential(**sigma_prior))
    temp = numpyro.sample("temp", dist.Normal(loc=mu, scale=sigma), obs=temp)
    return temp


guide = AutoLaplaceApproximation(
    m4_7_alt,
    init_loc_fn=numpyro.infer.init_to_value(values={"w": jnp.zeros(B.shape[1])}),
)
_, _jrng = jax.random.split(_jrng)
svi = SVI(
    model=m4_7_alt,
    guide=guide,
    optim=numpyro.optim.Adam(step_size=0.5),
    loss=Trace_ELBO(),
    B=B,
    temp=df2["temp"].values,
).run(jrng, 10_000)

_, _jrng = jax.random.split(_jrng)
posterior_samples = guide.sample_posterior(_jrng, svi.params, sample_shape=(10_000,))
_posterior_samples = {k: v for k, v in posterior_samples.items() if k != "mu"}
numpyro.diagnostics.print_summary(_posterior_samples, prob=0.89, group_by_chain=False)
```

## Easy


### 4E1

The first line is the likelihood.


### 4E2

2 parameters, mu and sigma


### 4E3

$$
P(\mu, \sigma | y) = \frac{P(y | \mu, \sigma) \cdot P(\mu) \cdot P(\sigma)}{\int \int P(y | \mu, \sigma) \cdot P(\mu) \cdot P(\sigma) d \mu d \sigma}
$$



### 4E4

The second line is the linear model


### 4E5

Three parameters, $\alpha$, $\beta$ and $\sigma$


## Medium


### 4M1

```python
def m1(y):
    mu = numpyro.sample("mu", dist.Normal(0, 10))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    return numpyro.sample("y", dist.Normal(mu, sigma), obs=y)


prior_predictive_samples = pd.DataFrame(
    numpyro.infer.Predictive(m1, num_samples=10_000)(jrng, y=None)
)
prior_predictive_samples["y"].plot(kind="kde", backend="matplotlib")
```

### 4M2

```python
def m1(y):
    mu = numpyro.sample("mu", dist.Normal(0, 10))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    return numpyro.sample("y", dist.Normal(mu, sigma), obs=y)
```

### 4M3

$$
\begin{split}
y & \sim Normal(\mu, \sigma) \\ 
\mu & = a + b \cdot x \\
\alpha & \sim Normal(0, 10) \\
\beta & \sim Uniform(0, 1) \\
\sigma & \sim Exponential(1)
\end{split}
$$


### 4M4

$$
\begin{split}
d_height & \sim Normal(\mu, \sigma) \\ 
\mu & = a + b \cdot (year - start year) \\
\alpha & \sim Normal(170, 29) \\
\beta & \sim Normal(0, 1) \\
\sigma & \sim Exponential(1)
\end{split}
$$


### 4M5

$$
\begin{split}
d_height & \sim Normal(\mu, \sigma) \\ 
\mu & = a + b \cdot (year - start year) \\
\alpha & \sim Normal(170, 29) \\
\beta & \sim LogNormal(0, 1) \\
\sigma & \sim Exponential(1)
\end{split}
$$


### 4M6

We don't want to change prior by peeking at data.


## Hard


### 4H1


```python
df = pd.read_csv("../data/Howell1.csv", sep=";")
df["weight_z_score"] = (df["weight"] - df["weight"].mean()) / df["weight"].std()


def h1(
    weight_z_score,
    *,
    alpha_prior={"loc": 170, "scale": 20},
    beta1_prior={"loc": 0, "scale": 1},
    beta2_prior={"loc": 0, "scale": 1},
    sigma_prior={"rate": 1 / 10},
    height=None,
):
    alpha = numpyro.sample("alpha", dist.Normal(**alpha_prior))
    beta1 = numpyro.sample("beta1", dist.Normal(**beta1_prior))
    beta2 = numpyro.sample("beta2", dist.Normal(**beta2_prior))
    mu = numpyro.deterministic(
        "mu", alpha + beta1 * weight_z_score + beta2 * weight_z_score**2
    )
    sigma = numpyro.sample("sigma", dist.Exponential(**sigma_prior))
    height_forecast = numpyro.sample("height", dist.Normal(mu, sigma), obs=height)
    return height_forecast
```

```python
# plot 100 random samples from the prior predictive distribution
weights = jnp.linspace(df["weight"].min(), df["weight"].max(), 50)
weight_z_score = (weights - df["weight"].mean()) / df["weight"].std()
prior_predictive_samples = numpyro.infer.Predictive(h1, num_samples=10_000)(
    jrng, weight_z_score=weight_z_score, height=None
)
height_prior_predictive_samples = pd.DataFrame(prior_predictive_samples["height"])
for _, sample in height_prior_predictive_samples.sample(
    n=100, random_state=seed
).iterrows():
    plt.plot(
        weights,
        sample,
        "k",
        alpha=0.2,
    )
plt.axhline(y=0, c="k", ls="--")
plt.axhline(y=272, c="k", ls="-", lw=0.5)
plt.title("Prior Predictive Height Samples")
```

```python
# plot mean and hpdi of prior predictive distribution
mu_mean = pd.DataFrame(prior_predictive_samples["mu"].mean(axis=0), index=weights)
mu_hpdi = pd.DataFrame(
    numpyro.diagnostics.hpdi(prior_predictive_samples["mu"], prob=0.89),
    columns=weights,
    index=["low", "high"],
)
height_hpdi = pd.DataFrame(
    numpyro.diagnostics.hpdi(prior_predictive_samples["height"], prob=0.89),
    columns=weights,
    index=["low", "high"],
)

# df.plot(kind="scatter", x="weight", y="height", backend="matplotlib")
plt.plot(weights, mu_mean, "k")
plt.fill_between(weights, mu_hpdi.iloc[0, :], mu_hpdi.iloc[1, :], color="k", alpha=0.5)
plt.fill_between(
    weights, height_hpdi.iloc[0, :], height_hpdi.iloc[1, :], color="k", alpha=0.2
)
# plt.show()
```

```python
guide = AutoLaplaceApproximation(h1)
_, _jrng = jax.random.split(_jrng)
svi = SVI(
    model=h1,
    guide=guide,
    optim=numpyro.optim.Adam(step_size=0.5),
    loss=Trace_ELBO(),
    weight_z_score=df["weight_z_score"].values,
    height=df["height"].values,
).run(jrng, 10_000)
```

```python
_, _jrng = jax.random.split(_jrng)
posterior_samples = guide.sample_posterior(_jrng, svi.params, sample_shape=(10_000,))
weights = jnp.linspace(df["weight"].min(), df["weight"].max(), 50)
weight_z_score = (weights - df["weight"].mean()) / df["weight"].std()
posterior_predictive = numpyro.infer.Predictive(
    guide.model, posterior_samples, return_sites=["mu", "height"]
)
posterior_predictive_samples = posterior_predictive(
    _jrng,
    weight_z_score=weight_z_score,
    height=None,
)

mu_posterior_predictive = pd.DataFrame(
    posterior_predictive_samples["mu"], columns=weights
)
height_posterior_predictive = pd.DataFrame(
    posterior_predictive_samples["height"], columns=weights
)

mu_mean = mu_posterior_predictive.mean(axis=0).to_frame().T
mu_hpdi = pd.DataFrame(
    numpyro.diagnostics.hpdi(mu_posterior_predictive, prob=0.89), columns=weight_seq
)
height_mean = height_posterior_predictive.mean(axis=0).to_frame().T
height_hpdi = pd.DataFrame(
    numpyro.diagnostics.hpdi(height_posterior_predictive, prob=0.89), columns=weight_seq
)
df[["weight", "height"]].plot(
    kind="scatter", x="weight", y="height", backend="matplotlib"
)
plt.plot(weights, mu_mean.loc[0, :], "k")
plt.fill_between(weights, mu_hpdi.loc[0, :], mu_hpdi.loc[1, :], color="k", alpha=0.5)
plt.fill_between(
    weights, height_hpdi.loc[0, :], height_hpdi.loc[1, :], color="k", alpha=0.2
)
plt.show()
```

```python
weights = jnp.array([46.95, 43.72, 64.68, 32.59, 54.63])
weight_z_score = (weights - df["weight"].mean()) / df["weight"].std()
posterior_predictive_samples = posterior_predictive(
    _jrng,
    weight_z_score=weight_z_score,
    height=None,
)

mu_posterior_predictive_samples = pd.DataFrame(posterior_predictive_samples["mu"])
height_posterior_predictive_samples = pd.DataFrame(
    posterior_predictive_samples["height"]
)

mu_mean = mu_posterior_predictive_samples.mean(axis=0).to_frame().T
mu_hpdi = pd.DataFrame(
    numpyro.diagnostics.hpdi(mu_posterior_predictive_samples, prob=0.89),
)
height_mean = height_posterior_predictive_samples.mean(axis=0).to_frame().T
height_hpdi = pd.DataFrame(
    numpyro.diagnostics.hpdi(height_posterior_predictive_samples, prob=0.89),
)

pd.concat(
    [
        pd.DataFrame(weights, columns=["weight"]),
        height_mean.T.rename(columns={0: "expected height"}),
        height_hpdi.T.rename(columns={0: "89% hpdi low", 1: "89% hpdi high"}),
    ],
    axis=1,
)
```

### 4H2

```python
df = pd.read_csv("../data/Howell1.csv", sep=";")
df = df.loc[df["age"] < 18, :]
df
```

```python
def h2(
    weight,
    *,
    alpha_prior={"loc": 100, "scale": 20},
    beta_prior={"loc": 0, "scale": 1},
    sigma_prior={"rate": 1 / 10},
    height=None,
):
    alpha = numpyro.sample("alpha", dist.Normal(**alpha_prior))
    beta = numpyro.sample("beta", dist.LogNormal(**beta_prior))
    mu = numpyro.deterministic("mu", alpha + beta * weight)
    sigma = numpyro.sample("sigma", dist.Exponential(**sigma_prior))
    height = numpyro.sample("height", dist.Normal(loc=mu, scale=sigma), obs=height)
```

```python
# plot 100 random samples from the prior predictive distribution
weight = jnp.linspace(df["weight"].min(), df["weight"].max(), 50)
prior_predictive_samples = numpyro.infer.Predictive(h2, num_samples=10_000)(
    jrng, weight=weight, height=None
)
height_prior_predictive_samples = pd.DataFrame(prior_predictive_samples["height"])
for _, sample in height_prior_predictive_samples.sample(
    n=100, random_state=seed
).iterrows():
    plt.plot(
        weight,
        sample,
        "k",
        alpha=0.2,
    )
plt.axhline(y=0, c="k", ls="--")
plt.axhline(y=272, c="k", ls="-", lw=0.5)
plt.title("Prior Predictive Height Samples")
```

```python
guide = AutoLaplaceApproximation(h2)
_, _jrng = jax.random.split(_jrng)
svi = SVI(
    model=h2,
    guide=guide,
    optim=numpyro.optim.Adam(step_size=0.5),
    loss=Trace_ELBO(),
    weight=df["weight"].values,
    height=df["height"].values,
).run(jrng, 10_000)
```

```python
_, _jrng = jax.random.split(_jrng)
posterior_samples = guide.sample_posterior(_jrng, svi.params, sample_shape=(10_000,))
_posterior_samples = {k: v for k, v in posterior_samples.items() if k != "mu"}
numpyro.diagnostics.print_summary(_posterior_samples, prob=0.89, group_by_chain=False)
```

For every 10 kg increase in weight, we expect 28cm increase in height

```python
weight = jnp.linspace(df["weight"].min(), df["weight"].max(), 50)
posterior_predictive = numpyro.infer.Predictive(
    guide.model, posterior_samples, return_sites=["mu", "height"]
)
posterior_predictive_samples = posterior_predictive(
    _jrng,
    weight=weight,
    height=None,
)

mu_posterior_predictive = pd.DataFrame(
    posterior_predictive_samples["mu"], columns=weight
)
height_posterior_predictive = pd.DataFrame(
    posterior_predictive_samples["height"], columns=weight
)

mu_mean = mu_posterior_predictive.mean(axis=0).to_frame().T
mu_hpdi = pd.DataFrame(
    numpyro.diagnostics.hpdi(mu_posterior_predictive, prob=0.89), columns=weight
)
height_mean = height_posterior_predictive.mean(axis=0).to_frame().T
height_hpdi = pd.DataFrame(
    numpyro.diagnostics.hpdi(height_posterior_predictive, prob=0.89), columns=weight
)
df[["weight", "height"]].plot(
    kind="scatter", x="weight", y="height", backend="matplotlib"
)
plt.plot(weight, mu_mean.loc[0, :], "k")
plt.fill_between(weight, mu_hpdi.loc[0, :], mu_hpdi.loc[1, :], color="k", alpha=0.5)
plt.fill_between(
    weight, height_hpdi.loc[0, :], height_hpdi.loc[1, :], color="k", alpha=0.2
)
plt.show()
```

Data appears to have curvature that's not captured by linear model.


### 4H3

```python
df = pd.read_csv("../data/Howell1.csv", sep=";")


def h3(
    weight,
    *,
    alpha_prior={"loc": 178, "scale": 20},
    beta_prior={"loc": 0, "scale": 1},
    sigma_prior={"rate": 1 / 10},
    height=None,
):
    alpha = numpyro.sample("alpha", dist.Normal(**alpha_prior))
    beta = numpyro.sample("beta", dist.LogNormal(**beta_prior))
    mu = numpyro.deterministic("mu", alpha + beta * jnp.log(weight))
    # mu = numpyro.deterministic("mu", alpha + beta * weight)
    sigma = numpyro.sample("sigma", dist.Exponential(**sigma_prior))
    height = numpyro.sample("height", dist.Normal(loc=mu, scale=sigma), obs=height)
    return height
```

```python
# plot 100 random samples from the prior predictive distribution
weight = jnp.linspace(df["weight"].min(), df["weight"].max(), 50)
prior_predictive_samples = numpyro.infer.Predictive(h3, num_samples=10_000)(
    jrng, weight=weight, height=None
)
mu_prior_predictive_samples = pd.DataFrame(prior_predictive_samples["mu"])
height_prior_predictive_samples = pd.DataFrame(prior_predictive_samples["height"])
for _, sample in mu_prior_predictive_samples.sample(
    n=100, random_state=seed
).iterrows():
    plt.plot(
        weight,
        sample,
        "k",
        alpha=0.2,
    )
plt.axhline(y=0, c="k", ls="--")
plt.axhline(y=272, c="k", ls="-", lw=0.5)
plt.title("Prior Predictive Expected Height Samples")
```

```python
guide = AutoLaplaceApproximation(h3)

_, _jrng = jax.random.split(_jrng)
svi = SVI(
    model=h3,
    guide=guide,
    optim=numpyro.optim.Adam(step_size=0.5),
    loss=Trace_ELBO(),
    weight=df["weight"].values,
    height=df["height"].values,
).run(jrng, 10_000)
```

```python
_, _jrng = jax.random.split(_jrng)
posterior_samples = guide.sample_posterior(_jrng, svi.params, sample_shape=(10_000,))
_posterior_samples = {k: v for k, v in posterior_samples.items() if k != "mu"}
numpyro.diagnostics.print_summary(_posterior_samples, prob=0.89, group_by_chain=False)
```

```python
weight = jnp.linspace(df["weight"].min(), df["weight"].max(), 50)
posterior_predictive = numpyro.infer.Predictive(
    guide.model, posterior_samples, return_sites=["mu", "height"]
)
posterior_predictive_samples = posterior_predictive(
    _jrng,
    weight=weight,
    height=None,
)

mu_posterior_predictive = pd.DataFrame(
    posterior_predictive_samples["mu"], columns=weight
)
height_posterior_predictive = pd.DataFrame(
    posterior_predictive_samples["height"], columns=weight
)

mu_mean = mu_posterior_predictive.mean(axis=0).to_frame().T
mu_hpdi = pd.DataFrame(
    numpyro.diagnostics.hpdi(mu_posterior_predictive, prob=0.89), columns=weight
)
height_mean = height_posterior_predictive.mean(axis=0).to_frame().T
height_hpdi = pd.DataFrame(
    numpyro.diagnostics.hpdi(height_posterior_predictive, prob=0.89), columns=weight
)
df[["weight", "height"]].plot(
    kind="scatter", x="weight", y="height", backend="matplotlib"
)
plt.plot(weight, mu_mean.loc[0, :], "k")
plt.fill_between(weight, mu_hpdi.loc[0, :], mu_hpdi.loc[1, :], color="k", alpha=0.5)
plt.fill_between(
    weight, height_hpdi.loc[0, :], height_hpdi.loc[1, :], color="k", alpha=0.2
)
plt.show()
```

### 4H4

```python
df = pd.read_csv("../data/Howell1.csv", sep=";")
df["weight_z_score"] = (df["weight"] - df["weight"].mean()) / df["weight"].std()


def h4(
    weight_z_score,
    *,
    alpha_prior={"loc": 170, "scale": 20},
    beta1_prior={"loc": 0, "scale": 1},
    beta2_prior={"loc": 0, "scale": 1},
    sigma_prior={"rate": 1 / 10},
    height=None,
):
    alpha = numpyro.sample("alpha", dist.Normal(**alpha_prior))
    beta1 = numpyro.sample("beta1", dist.Normal(**beta1_prior))
    beta2 = numpyro.sample("beta2", dist.Normal(**beta2_prior))
    mu = numpyro.deterministic(
        "mu", alpha + beta1 * weight_z_score + beta2 * weight_z_score**2
    )
    sigma = numpyro.sample("sigma", dist.Exponential(**sigma_prior))
    height_forecast = numpyro.sample("height", dist.Normal(mu, sigma), obs=height)
    return height_forecast
```

```python
# plot 100 random samples from the prior predictive distribution
weights = jnp.linspace(df["weight"].min(), df["weight"].max(), 50)
weight_z_score = (weights - df["weight"].mean()) / df["weight"].std()
prior_predictive_samples = numpyro.infer.Predictive(h4, num_samples=10_000)(
    jrng, weight_z_score=weight_z_score, height=None
)
height_prior_predictive_samples = pd.DataFrame(prior_predictive_samples["height"])
for _, sample in height_prior_predictive_samples.sample(
    n=100, random_state=seed
).iterrows():
    plt.plot(
        weights,
        sample,
        "k",
        alpha=0.2,
    )
plt.axhline(y=0, c="k", ls="--")
plt.axhline(y=272, c="k", ls="-", lw=0.5)
plt.title("Prior Predictive Height Samples")
```

```python

```
