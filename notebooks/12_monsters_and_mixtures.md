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

# Chapter 12: Monsters and Mixtures

```python
%load_ext jupyter_black
import os

import arviz as az
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
import pandas as pd

numpyro.enable_x64()
numpyro.set_host_device_count(os.cpu_count())

seed = 84735
jrng = jax.random.key(seed)
_, *jrngs = jax.random.split(jrng, 5)
plt.rcParams["figure.figsize"] = [10, 6]
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
### Code 12.1

```python
pbar = 0.5
theta = 5
x = jnp.linspace(0, 1, 101)
pdf = jnp.exp(dist.Beta(pbar * theta, (1 - pbar) * theta).log_prob(x))
plt.plot(x, pdf)
plt.gca().set(xlabel="x", ylabel="pdf", title="Beta (0.5, 5)")
```

### Code 12.2

```python
df = pd.read_csv("../data/UCBadmit.csv", sep=";")
df["gender_id"] = (df["applicant.gender"] == "female").astype(int)
df
```

```python
def model_12_1(gender_id, admit, applications):
    with numpyro.plate(name="gender", size=2):
        alpha = numpyro.sample("alpha", dist.Normal(0, 1.5))
        _pbar = numpyro.deterministic("pbar", jax.scipy.special.expit(alpha))
    pbar = _pbar[gender_id]
    theta = numpyro.deterministic(
        "theta", numpyro.sample("phi", dist.Exponential(1)) + 2
    )
    admit = numpyro.sample(
        "admit",
        dist.BetaBinomial(pbar * theta, (1 - pbar) * theta, applications),
        obs=admit,
    )
    numpyro.deterministic("admit_rate", admit / applications)
    return admit


prior_predictive_12_1 = Predictive(model_12_1, num_samples=100)
prior_predictive_samples_12_1 = prior_predictive_12_1(
    jrng, gender_id=0, admit=None, applications=100
)
```

```python
plt.scatter(
    prior_predictive_samples_12_1["pbar"][:, 0], prior_predictive_samples_12_1["theta"]
)
plt.gca().set(xlabel="pbar", ylabel="theta", title="Prior predictive samples")
```

```python
for i in range(len(prior_predictive_samples_12_1["admit"])):
    plt.plot(
        x,
        jnp.exp(
            dist.Beta(
                prior_predictive_samples_12_1["pbar"][i, 0]
                * prior_predictive_samples_12_1["theta"][i],
                (1 - prior_predictive_samples_12_1["pbar"][i, 0])
                * prior_predictive_samples_12_1["theta"][i],
            ).log_prob(x)
        ),
        color="k",
        alpha=0.2,
    )
    plt.gca().set(
        xlabel="x", ylabel="pdf", title="Prior predictive samples of Beta Distributions"
    )
```

```python
train_data_12_1 = {
    "gender_id": df["gender_id"].values,
    "admit": df["admit"].values,
    "applications": df["applications"].values,
}
mcmc_12_1 = MCMC(NUTS(model_12_1), num_warmup=500, num_samples=500, num_chains=4)
mcmc_12_1.run(jrngs[0], **train_data_12_1)
mcmc_12_1.print_summary(exclude_deterministic=False)
posterior_samples_12_1 = mcmc_12_1.get_samples()
idata_12_1 = az.from_numpyro(mcmc_12_1)
az.plot_trace(
    idata_12_1, var_names=["alpha", "pbar", "theta"], kind="rank_bars", compact=False
)
plt.tight_layout()
```

### Code 12.3

```python
da = posterior_samples_12_1["alpha"][:, 0] - posterior_samples_12_1["alpha"][:, 1]
numpyro.diagnostics.print_summary(
    {
        "da": da,
        **prune_return_sites(posterior_samples_12_1, depth=2, exclude=["admit_rate"]),
    },
    0.89,
    False,
)
```

### Code 12.4

```python
gender_id = 1
x = jnp.linspace(0, 1, 101)
plt.subplot(title="Posterior Distribution of Female Admission Rate", ylim=(0, 3))

# draw posterior mean beta distribution
pbar_mu = posterior_samples_12_1["pbar"][:, gender_id].mean()
theta_mu = posterior_samples_12_1["theta"].mean()
plt.plot(
    x,
    jnp.exp(dist.Beta(pbar_mu * theta_mu, (1 - pbar_mu) * theta_mu).log_prob(x)),
    color="blue",
    linewidth=3,
    label="mean",
)

# draw posterior samples of beta distribution
for i in range(50):
    _pbar = posterior_samples_12_1["pbar"][i, gender_id]
    _theta = posterior_samples_12_1["theta"][i]
    plt.plot(
        x,
        jnp.exp(dist.Beta(_pbar * _theta, (1 - _pbar) * _theta).log_prob(x)),
        color="k",
        alpha=0.2,
    )
```

### Code 12.5

```python
plt.subplot(title="Posterior Validation Check", xlabel="Case", ylabel="Admissions Rate")
plt.plot(df.index, df["admit"] / df["applications"], "o", color="blue", alpha=0.7)
posterior_predictive_12_1 = Predictive(model_12_1, posterior_samples_12_1)
posterior_predictive_samples_12_1 = posterior_predictive_12_1(
    jrng,
    gender_id=df["gender_id"].values,
    applications=df["applications"].values,
    admit=None,
)
posterior_predictive_samples_admit_rate_12_1 = posterior_predictive_samples_12_1[
    "admit_rate"
][:, df["gender_id"].values]
expected_pbar_12_1 = posterior_predictive_samples_admit_rate_12_1.mean(0)
stddev_pbar_12_1 = posterior_predictive_samples_admit_rate_12_1.std(0)
hpdi_pbar_12_1 = numpyro.diagnostics.hpdi(
    posterior_predictive_samples_admit_rate_12_1, 0.89
)
plt.errorbar(
    x=df.index,
    y=expected_pbar_12_1,
    yerr=stddev_pbar_12_1,
    fmt="o",
    color="k",
    mfc="none",
)
plt.scatter(x=df.index, y=hpdi_pbar_12_1[0, :], marker="+", color="k", alpha=0.7)
plt.scatter(x=df.index, y=hpdi_pbar_12_1[1, :], marker="+", color="k", alpha=0.7)
```

### Code 12.6

```python
df = pd.read_csv("../data/Kline.csv", sep=";")
df["contact_id"] = (df["contact"] == "high").astype(int)


def model_12_2(contact_id, P, tools):
    g = numpyro.sample("g", dist.Exponential(1))
    with numpyro.plate(name="contact", size=2):
        a = numpyro.sample("a", dist.Normal(1, 1))
        b = numpyro.sample("b", dist.Exponential(1))
    lambda_ = numpyro.deterministic(
        "lambda",
        jnp.exp(a[contact_id]) * jnp.power(P, b[contact_id]) / g,
    )
    phi = numpyro.sample("phi", dist.Exponential(1))
    tools = numpyro.sample(
        "tools",
        dist.GammaPoisson(lambda_ / phi, 1 / phi),
        obs=tools,
    )
    return tools


training_data_12_2 = {
    "contact_id": df["contact_id"].values,
    "P": df["population"].values,
    "tools": df["total_tools"].values,
}
mcmc_12_2 = MCMC(NUTS(model_12_2), num_warmup=500, num_samples=500, num_chains=4)
mcmc_12_2.run(jrngs[1], **training_data_12_2)
mcmc_12_2.print_summary(exclude_deterministic=False)
posterior_samples_12_2 = mcmc_12_2.get_samples()
idata_12_2 = az.from_numpyro(mcmc_12_2)
```

### Code 12.7

```python
# define parameters
prob_drink = 0.2
rate_work = 1  # average 1 manuscript per day

# sample one year of production
N = 365
# simulate days monks drink
drink = dist.Bernoulli(prob_drink).sample(jrng, (N,))

# simulate manuscript completed
y = (1 - drink) * dist.Poisson(rate_work).sample(jrng, (N,))
```

### Code 12.8

```python
plt.hist(y, color="k", bins=jnp.arange(-0.5, 6), rwidth=0.1)
plt.gca().set(xlabel="manuscripts completed")
zeros_drink = jnp.sum(drink)
zeros_work = jnp.sum((y == 0) & (drink == 0))
zeros_total = jnp.sum(y == 0)
plt.plot([0, 0], [zeros_work, zeros_total], "royalblue", lw=8)
plt.show()
```

### Code 12.9

```python
def model_12_3(y):
    a_p = numpyro.sample("a_p", dist.Normal(-1.5, 1))
    p = numpyro.deterministic("p", jax.scipy.special.expit(a_p))
    a_lambda = numpyro.sample("a_lambda", dist.Normal(1, 0.5))
    lambda_ = numpyro.deterministic("lambda", jnp.exp(a_lambda))
    y = numpyro.sample("y", dist.ZeroInflatedPoisson(p, lambda_), obs=y)
    return y


mcmc_12_3 = MCMC(NUTS(model_12_3), num_warmup=500, num_samples=1000, num_chains=4)
mcmc_12_3.run(jrng, y=y)
mcmc_12_3.print_summary(exclude_deterministic=False)
idata_12_3 = az.from_numpyro(mcmc_12_3)
az.plot_trace(idata_12_3, kind="rank_bars", compact=False)
plt.tight_layout()
```

### Code 12.10

```python
print(mcmc_12_3.get_samples()["p"].mean())
print(mcmc_12_3.get_samples()["lambda"].mean())
```

### Code 12.11

```python
def model_12_3_alt(y):
    a_p = numpyro.sample("a_p", dist.Normal(-1.5, 1))
    p = numpyro.deterministic("p", jax.scipy.special.expit(a_p))
    a_lambda = numpyro.sample("a_lambda", dist.Normal(1, 0.5))
    lambda_ = numpyro.deterministic("lambda", jnp.exp(a_lambda))

    with numpyro.handlers.mask(mask=y > 0):
        numpyro.factor(
            name="y|y>0",
            log_factor=jnp.log1p(-p) + dist.Poisson(lambda_).log_prob(y),
        )
    with numpyro.handlers.mask(mask=y == 0):
        numpyro.factor(
            name="y|y=0",
            log_factor=jnp.log(
                p + (1 - p) * jnp.exp(dist.Poisson(lambda_).log_prob(0))
            ),
        )

    return y


mcmc_12_3_alt = MCMC(
    NUTS(model_12_3_alt), num_warmup=500, num_samples=1000, num_chains=4
)
mcmc_12_3_alt.run(jrng, y=y)
mcmc_12_3_alt.print_summary(exclude_deterministic=False)
idata_12_3_alt = az.from_numpyro(mcmc_12_3_alt)
az.plot_trace(idata_12_3_alt, kind="rank_bars", compact=False)
plt.tight_layout()
```

### Code 12.12

```python
df = pd.read_csv("../data/Trolley.csv", sep=";")
df
```

### Code 12.13

```python
plt.hist(df["response"], label="response", bins=jnp.arange(0.5, 8), rwidth=0.1)
```

### Code 12.14

```python
# discrete proportion of each response value
pr_k = jnp.bincount(df["response"].values) / df.shape[0]

# cumsum converts to cumulative proportions
cum_pr_k = jnp.cumsum(pr_k)[1:]

# plot
plt.step(jnp.arange(1, 8), cum_pr_k, where="mid", label="response")
plt.gca().set(xlabel="response", ylabel="cumulative proportion")
```

### Code 12.15

```python
jnp.round(jax.scipy.special.logit(cum_pr_k), 2)
```

### Code 12.16

```python
def model_12_4(response):
    cutpoints = numpyro.sample(
        "cutpoints",
        dist.TransformedDistribution(
            dist.Normal(0, 1.5).expand([6]), dist.transforms.OrderedTransform()
        ),
    )
    response = numpyro.sample(
        "response", dist.OrderedLogistic(0, cutpoints), obs=response
    )
    return response


mcmc_12_4 = MCMC(NUTS(model_12_4), num_warmup=500, num_samples=1000, num_chains=4)
mcmc_12_4.run(jrng, response=df["response"].values - 1)
mcmc_12_4.print_summary(exclude_deterministic=False)
idata_12_4 = az.from_numpyro(mcmc_12_4)
az.plot_trace(idata_12_4, kind="rank_bars", compact=False)
plt.tight_layout()
```

### Code 12.17

pass


### Code 12.18

```python
mcmc_12_4.print_summary(exclude_deterministic=False)
```

### Code 12.19

```python
jnp.round(jax.scipy.special.expit(mcmc_12_4.get_samples()["cutpoints"].mean(0)), 3)
```

### Code 12.20

```python
coef = jnp.mean(mcmc_12_4.get_samples()["cutpoints"], 0)
pk = jnp.exp(dist.OrderedLogistic(0, coef).log_prob(jnp.arange(7)))
pk
```

### Code 12.21

```python
jnp.sum(pk * jnp.arange(1, 8))
```

### Code 12.22

```python
coef = jnp.mean(mcmc_12_4.get_samples()["cutpoints"], 0) - 0.5
pk = jnp.exp(dist.OrderedLogistic(0, coef).log_prob(jnp.arange(7)))
pk
```

### Code 12.23

```python
jnp.sum(pk * jnp.arange(1, 8))
```

### Code 12.24

```python
def model_12_5(action, contact, intention, response):
    cutpoints = numpyro.sample(
        "cutpoints",
        dist.TransformedDistribution(
            dist.Normal(0, 1.5).expand([6]), dist.transforms.OrderedTransform()
        ),
    )
    b_action = numpyro.sample("b_action", dist.Normal(0, 0.5))
    b_contact = numpyro.sample("b_contact", dist.Normal(0, 0.5))
    b_intention = numpyro.sample("b_intention", dist.Normal(0, 0.5))
    b_action_intention = numpyro.sample("b_action_intention", dist.Normal(0, 0.5))
    b_contact_intention = numpyro.sample("b_contact_intention", dist.Normal(0, 0.5))
    phi = numpyro.deterministic(
        "phi",
        b_action * action
        + b_contact * contact
        + b_intention * intention
        + b_action_intention * action * intention
        + b_contact_intention * contact * intention,
    )
    response = numpyro.sample(
        "response", dist.OrderedLogistic(phi, cutpoints), obs=response
    )
    return response


training_data_12_5 = {
    "action": df["action"].values,
    "contact": df["contact"].values,
    "intention": df["intention"].values,
    "response": df["response"].values - 1,
}
mcmc_12_5 = MCMC(NUTS(model_12_5), num_warmup=500, num_samples=1000, num_chains=4)
mcmc_12_5.run(jrng, **training_data_12_5)
mcmc_12_5.print_summary()
idata_12_5 = az.from_numpyro(mcmc_12_5)
az.plot_trace(
    idata_12_5,
    var_names=[
        "b_action",
        "b_contact",
        "b_intention",
        "b_action_intention",
        "b_contact_intention",
    ],
    kind="rank_bars",
    compact=False,
)
plt.tight_layout()
```

### Code 12.25

```python
az.plot_forest(
    mcmc_12_5.get_samples(group_by_chain=True),
    var_names=[
        "b_action",
        "b_contact",
        "b_intention",
        "b_action_intention",
        "b_contact_intention",
    ],
    combined=True,
    hdi_prob=0.89,
)
plt.gca().set(xlim=(-1.42, 0.02))
plt.show()
```

### Code 12.26

```python
ax = plt.subplot(xlabel="intention", ylabel="probability", xlim=(0, 1), ylim=(0, 1))
fig = plt.gcf()
```

### Code 12.27

```python
posterior_samples_12_5 = mcmc_12_5.get_samples()
posterior_predictive_12_5 = Predictive(model_12_5, posterior_samples_12_5)

actions = 0
contacts = 0
intentions = jnp.arange(2)
posterior_predictive_samples_12_5 = posterior_predictive_12_5(
    jrng, action=actions, contact=contacts, intention=intentions, response=None
)
```

### Code 12.28

```python
for s in range(50):
    pk = jax.scipy.special.expit(
        posterior_samples_12_5["cutpoints"][s]
        - posterior_predictive_samples_12_5["phi"][s][..., None]
    )
    for i in range(6):
        ax.plot(intentions, pk[:, i], color="k", alpha=0.2)
fig
```

### Code 12.29

```python
plt.hist(
    posterior_predictive_samples_12_5["response"] + 1,
    bins=jnp.arange(0.5, 8),
    rwidth=0.5,
)
plt.gca().set(
    xlabel="response",
    ylabel="count",
    title="Posterior Distribution",
)
```

### Code 12.30

```python
df = pd.read_csv("../data/Trolley.csv", sep=";")
display(df["edu"].unique())
df
```

### Code 12.31

```python
edu_levels = [
    "Elementary School",
    "Middle School",
    "Some High School",
    "High School Graduate",
    "Some College",
    "Bachelor's Degree",
    "Master's Degree",
    "Graduate Degree",
]
df["edu_new"] = df["edu"].map(dict(zip(edu_levels, range(8))))
df
```

### Code 12.32

```python
delta = dist.Dirichlet(2 * jnp.ones(7)).sample(jrng, sample_shape=(10,))
display(delta.shape)
delta
```

### Code 12.33

```python
h = 3
plt.subplot(xlim=(0, 7), ylim=(0, 0.4), xlabel="education level", ylabel="probability")
for i in range(10):
    plt.plot(range(7), delta[i], color="k", alpha=1 if i == h else 0.3)
```

### Code 12.34

```python
def model_12_6(action, contact, intention, education, response):

    cutpoints = numpyro.sample(
        "cutpoints",
        dist.TransformedDistribution(
            dist.Normal(0, 1.5).expand([6]), dist.transforms.OrderedTransform()
        ),
    )
    b_action = numpyro.sample("b_action", dist.Normal(0, 1))
    b_contact = numpyro.sample("b_contact", dist.Normal(0, 1))
    b_intention = numpyro.sample("b_intention", dist.Normal(0, 1))
    b_education = numpyro.sample("b_education", dist.Normal(0, 1))
    delta_education = numpyro.sample(
        "delta_education", dist.Dirichlet(jnp.repeat(2, 7))
    )
    total_education = jnp.sum(
        jnp.where(
            jnp.arange(8) <= education[..., None], jnp.pad(delta_education, (1, 0)), 0
        ),
        axis=1,
    )
    phi = numpyro.deterministic(
        "phi",
        b_action * action
        + b_contact * contact
        + b_intention * intention
        + b_education * total_education,
    )
    response = numpyro.sample(
        "response", dist.OrderedLogistic(phi, cutpoints), obs=response
    )
    return response


training_data = {
    "action": df["action"].values,
    "contact": df["contact"].values,
    "intention": df["intention"].values,
    "education": df["edu_new"].values,
    "response": df["response"].values - 1,
}
mcmc_12_6 = MCMC(NUTS(model_12_6), num_warmup=500, num_samples=1000, num_chains=4)
mcmc_12_6.run(jrng, **training_data)
idata_12_6 = az.from_numpyro(mcmc_12_6)
az.plot_trace(
    idata_12_6,
    var_names=[
        "b_action",
        "b_contact",
        "b_intention",
        "b_education",
        "delta_education",
    ],
    kind="rank_bars",
    compact=False,
)
plt.tight_layout()
```

### Code 12.35

```python
mcmc_12_6.print_summary()
```

### Code 12.36

```python
a12_6 = az.from_numpyro(
    mcmc_12_6, coords={"labels": edu_levels[:7]}, dims={"delta_education": ["labels"]}
)
az.plot_pair(a12_6, var_names="delta_education")
```

### Code 12.37

```python
def model_12_7(action, contact, intention, education, response):

    cutpoints = numpyro.sample(
        "cutpoints",
        dist.TransformedDistribution(
            dist.Normal(0, 1.5).expand([6]), dist.transforms.OrderedTransform()
        ),
    )
    b_action = numpyro.sample("b_action", dist.Normal(0, 1))
    b_contact = numpyro.sample("b_contact", dist.Normal(0, 1))
    b_intention = numpyro.sample("b_intention", dist.Normal(0, 1))
    b_education = numpyro.sample("b_education", dist.Normal(0, 1))
    phi = numpyro.deterministic(
        "phi",
        b_action * action
        + b_contact * contact
        + b_intention * intention
        + b_education * education,
    )
    response = numpyro.sample(
        "response", dist.OrderedLogistic(phi, cutpoints), obs=response
    )
    return response


training_data = {
    "action": df["action"].values,
    "contact": df["contact"].values,
    "intention": df["intention"].values,
    "education": scale(df["edu_new"].values),
    "response": df["response"].values - 1,
}
mcmc_12_7 = MCMC(NUTS(model_12_7), num_warmup=500, num_samples=1000, num_chains=4)
mcmc_12_7.run(jrng, **training_data)
mcmc_12_7.print_summary()
idata_12_7 = az.from_numpyro(mcmc_12_7)
az.plot_trace(
    idata_12_7,
    var_names=["b_action", "b_contact", "b_intention", "b_education"],
    kind="rank_bars",
    compact=False,
)
plt.tight_layout()
```

<!-- #region -->
## Easy

### 12E1

Ordered categorical, some categories are better than others (but no clear measure of distance between categories). For instance, karate belt colors.
Unorder example: favorite ice cream flavor.

### 12E2

Cumulative logit link function. Same as logit, but the probability is that of all the categories below the current category.


### 12E3

Will biais parameter estimates lower.


### 12E4

Number of children by women in new york state. rural vs city and within city, diff cultural norms: mixture of rates will lead to over-dispertion.

Count of the latest fashionable sneakers sold will be under dispersed (sold out).
<!-- #endregion -->

## Medium

### 12M1

```python
count = jnp.array([12, 36, 7, 41])
p = count / count.sum()
cum_proba = jnp.cumsum(p)
cum_odds = cum_proba / (1 - cum_proba)
log_cum_odds = jnp.log(cum_odds)
log_cum_odds
```

### 12M2

```python
plt.subplot(xlabel="response", ylabel="cumulative proportion")
x = jnp.arange(1, 5)
plt.plot(x, cum_proba, label="cumulative proportion")
plt.bar(
    x,
    jnp.pad(cum_proba[:-1], (1, 0)),
    label="proportion",
    color="k",
    width=0.1,
    alpha=0.5,
)
plt.bar(
    x,
    p,
    label="proportion",
    color="b",
    width=0.1,
    alpha=0.5,
    bottom=jnp.pad(cum_proba[:-1], (1, 0)),
)
```

### 12M3

$P(Y=y|y=0, p_0, p_b) = p_0 + (1-p_0) \cdot Binom(Y=0|p_b)$

$P(Y=y|y>0, p_0, p_b) = Binom(Y=y|p_b)$

```python
def zero_inflated_binomial(jrng, p_binom, p_zero, sample_shape):
    is_zero = dist.Bernoulli(p_zero).sample(jrng, sample_shape=sample_shape)
    return jnp.where(
        is_zero,
        jnp.zeros(sample_shape),
        dist.Binomial(1, p_binom).sample(jrng, sample_shape=sample_shape),
    )
```

## Hard
### 12H1

```python
df = pd.read_csv("../data/Hurricanes.csv", sep=";")
df["femininity"] = scale(df["femininity"])
df
```

```python
def model_12h0(deaths):
    a = numpyro.sample("a", dist.Normal(0, 10))
    log_lambda = numpyro.deterministic("log_lambda", a)
    lambda_ = numpyro.deterministic("lambda", jnp.exp(log_lambda))
    numpyro.sample("deaths", dist.Poisson(lambda_), obs=deaths)


training_data = {"deaths": df["deaths"].values}
mcmc_12h0 = MCMC(NUTS(model_12h0), num_warmup=500, num_samples=1000, num_chains=4)
mcmc_12h0.run(jrng, **training_data)
mcmc_12h0.print_summary()
idata_12h0 = az.from_numpyro(mcmc_12h0)
az.plot_trace(
    idata_12h0,
    kind="rank_bars",
    var_names=["a", "log_lambda", "lambda"],
    compact=False,
)
plt.tight_layout()
```

```python
def model_12h1(femininity, deaths):
    a = numpyro.sample("a", dist.Normal(0, 10))
    b = numpyro.sample("b", dist.Normal(0, 1))
    log_lambda = numpyro.deterministic("log_lambda", a + b * femininity)
    lambda_ = numpyro.deterministic("lambda", jnp.exp(log_lambda))
    numpyro.sample("deaths", dist.Poisson(lambda_), obs=deaths)


training_data = {"femininity": df["femininity"].values, "deaths": df["deaths"].values}
mcmc_12h1 = MCMC(NUTS(model_12h1), num_warmup=500, num_samples=1000, num_chains=4)
mcmc_12h1.run(jrng, **training_data)
mcmc_12h1.print_summary()
idata_12h1 = az.from_numpyro(mcmc_12h1)
az.plot_trace(
    idata_12h1,
    kind="rank_bars",
    var_names=["a", "b"],
    compact=False,
)
plt.tight_layout()
```

```python
compare_12h1 = az.compare(
    {"mcmc_12h0": idata_12h0, "mcmc_12h1": idata_12h1}, scale="deviance", ic="waic"
)
display(compare_12h1)
az.plot_compare(compare_12h1)
```

```python
femininities = jnp.linspace(-2, 2, 30)
posterior_predictive_12h1 = Predictive(model_12h1, mcmc_12h1.get_samples())
posterior_predictive_samples_12h1 = posterior_predictive_12h1(
    jrng, femininity=femininities, deaths=None
)
deaths_mu = posterior_predictive_samples_12h1["deaths"].mean(axis=0)
deaths_hpdi = numpyro.diagnostics.hpdi(
    posterior_predictive_samples_12h1["deaths"], 0.89
)
plt.plot(
    femininities,
    deaths_mu,
    label="mean",
    color="black",
)
plt.fill_between(femininities, deaths_hpdi[0], deaths_hpdi[1], color="black", alpha=0.2)
plt.scatter(
    df["femininity"],
    df["deaths"],
)
```

Although the model appears to be quite convinced femininity matters (beta is positive), we can see that it is unable to retrodict the training set. Femininity does not appear to explain much of the variation in the data.


### 12H2

```python
df = pd.read_csv("../data/Hurricanes.csv", sep=";")
df["femininity"] = scale(df["femininity"])
df
```

```python
def model_12h2(femininity, deaths):
    a = numpyro.sample("a", dist.Normal(0, 10))
    b = numpyro.sample("b", dist.Normal(0, 1))
    log_lambda = numpyro.deterministic("log_lambda", a + b * femininity)
    lambda_ = numpyro.deterministic("lambda", jnp.exp(log_lambda))
    phi = numpyro.sample("phi", dist.Exponential(1))
    numpyro.sample(
        "deaths",
        dist.GammaPoisson(concentration=lambda_ / phi, rate=1 / phi),
        obs=deaths,
    )


training_data = {"femininity": df["femininity"].values, "deaths": df["deaths"].values}
mcmc_12h2 = MCMC(NUTS(model_12h2), num_warmup=500, num_samples=1000, num_chains=4)
mcmc_12h2.run(jrng, **training_data)
mcmc_12h2.print_summary()
idata_12h2 = az.from_numpyro(mcmc_12h2)
az.plot_trace(
    idata_12h2,
    kind="rank_bars",
    var_names=["a", "b", "phi"],
    compact=False,
)
plt.tight_layout()
```

```python
femininities = jnp.linspace(-2, 2, 30)
posterior_predictive_12h2 = Predictive(model_12h2, mcmc_12h2.get_samples())
posterior_predictive_samples_12h2 = posterior_predictive_12h2(
    jrng, femininity=femininities, deaths=None
)
deaths_mu = posterior_predictive_samples_12h2["deaths"].mean(axis=0)
deaths_hpdi = numpyro.diagnostics.hpdi(
    posterior_predictive_samples_12h2["deaths"], 0.89
)
plt.plot(
    femininities,
    deaths_mu,
    label="mean",
    color="black",
)
plt.fill_between(femininities, deaths_hpdi[0], deaths_hpdi[1], color="black", alpha=0.2)
plt.scatter(
    df["femininity"],
    df["deaths"],
)
```

```python
compare_12h2 = az.compare(
    {"mcmc_12h0": idata_12h0, "mcmc_12h1": idata_12h1, "mcmc_12h2": idata_12h2},
    scale="deviance",
    ic="waic",
)
display(compare_12h2)
az.plot_compare(compare_12h2)
```

The model no longer thinks that femininity is associated with higher death count: now that we allow each hurricane to have its own death rate, the variability goes into that, rather than being squeezed into femininity.


### 12H3

```python
df = pd.read_csv("../data/Hurricanes.csv", sep=";")
df["femininity"] = scale(df["femininity"])
df["damage_norm"] = scale(df["damage_norm"])
df["min_pressure"] = scale(df["min_pressure"])
df
```

```python
def model_12h3a(damage, femininity, deaths):
    a = numpyro.sample("a", dist.Normal(2, 1))
    b_feminity = numpyro.sample("b_feminity", dist.Normal(0, 0.5))
    b_damage = numpyro.sample("b_damage", dist.Normal(0, 0.5))
    b_damage_feminity = numpyro.sample("b_damage_feminity", dist.Normal(0, 0.25))
    log_lambda = numpyro.deterministic(
        "log_lambda",
        a
        + b_feminity * femininity
        + b_damage * damage
        + b_damage_feminity * damage * femininity,
    )
    lambda_ = numpyro.deterministic("lambda", jnp.exp(log_lambda))
    phi = numpyro.sample("phi", dist.Exponential(1))
    numpyro.sample(
        "deaths",
        dist.GammaPoisson(concentration=lambda_ / phi, rate=1 / phi),
        obs=deaths,
    )


num_samples = 100
damages = jnp.tile(jnp.linspace(-2, 2, 30), 4)
femininities = jnp.repeat(jnp.linspace(-2, 2, 4), 30)
prior_predictive_12h3a = Predictive(model_12h3a, num_samples=num_samples)
prior_predictive_samples_12h3a = prior_predictive_12h3a(
    jrng,
    damage=damages,
    femininity=femininities,
    deaths=None,
)
fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
fig.suptitle("Prior Predictive Checks")
for i, femininity in enumerate(jnp.unique(femininities)):
    ax = axes[i // 2, i % 2]
    ax.set(
        title=f"femininity={femininity:.2f}",
        xlabel="damage",
        ylabel="deaths",
        ylim=(0, 250),
    )
    is_current_femininity = femininities == femininity

    deaths_mu = prior_predictive_samples_12h3a["deaths"][:, is_current_femininity].mean(
        axis=0
    )
    deaths_hpdi = numpyro.diagnostics.hpdi(
        prior_predictive_samples_12h3a["deaths"][:, is_current_femininity],
        0.89,
    )
    for sample in range(num_samples):
        ax.plot(
            damages[is_current_femininity],
            prior_predictive_samples_12h3a["deaths"][sample, is_current_femininity],
            color="k",
            alpha=0.2,
        )
    ax.plot(
        damages[is_current_femininity],
        deaths_mu,
        color="black",
        label=f"femininity={femininity:.2f}",
    )
    ax.fill_between(
        damages[is_current_femininity],
        deaths_hpdi[0],
        deaths_hpdi[1],
        color="black",
        alpha=0.2,
    )
    plt.tight_layout()
```

```python
training_data_12h3a = {
    "damage": df["damage_norm"].values,
    "femininity": df["femininity"].values,
    "deaths": df["deaths"].values,
}
mcmc_12h3a = MCMC(NUTS(model_12h3a), num_warmup=500, num_samples=1000, num_chains=4)
mcmc_12h3a.run(jrng, **training_data_12h3a)
mcmc_12h3a.print_summary()
idata_12h3a = az.from_numpyro(mcmc_12h3a)
az.plot_trace(
    idata_12h3a,
    kind="rank_bars",
    var_names=["a", "b_feminity", "b_damage", "b_damage_feminity", "phi"],
    compact=False,
)
plt.tight_layout()
```

```python
def model_12h3b(pressure, femininity, deaths):
    a = numpyro.sample("a", dist.Normal(2, 1))
    b_feminity = numpyro.sample("b_feminity", dist.Normal(0, 0.5))
    b_pressure = numpyro.sample("b_pressure", dist.Normal(0, 0.5))
    b_pressure_feminity = numpyro.sample("b_pressure_feminity", dist.Normal(0, 0.25))
    log_lambda = numpyro.deterministic(
        "log_lambda",
        a
        + b_feminity * femininity
        + b_pressure * pressure
        + b_pressure_feminity * pressure * femininity,
    )
    lambda_ = numpyro.deterministic("lambda", jnp.exp(log_lambda))
    phi = numpyro.sample("phi", dist.Exponential(1))
    numpyro.sample(
        "deaths",
        dist.GammaPoisson(concentration=lambda_ / phi, rate=1 / phi),
        obs=deaths,
    )


num_samples = 100
pressures = jnp.tile(jnp.linspace(-2, 2, 30), 4)
femininities = jnp.repeat(jnp.linspace(-2, 2, 4), 30)
prior_predictive_12h3b = Predictive(model_12h3b, num_samples=num_samples)
prior_predictive_samples_12h3b = prior_predictive_12h3b(
    jrng,
    pressure=pressures,
    femininity=femininities,
    deaths=None,
)
fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
fig.suptitle("Prior Predictive Checks")
for i, femininity in enumerate(jnp.unique(femininities)):
    ax = axes[i // 2, i % 2]
    ax.set(
        title=f"femininity={femininity:.2f}",
        xlabel="pressure",
        ylabel="deaths",
        ylim=(0, 250),
    )
    is_current_femininity = femininities == femininity

    deaths_mu = prior_predictive_samples_12h3b["deaths"][:, is_current_femininity].mean(
        axis=0
    )
    deaths_hpdi = numpyro.diagnostics.hpdi(
        prior_predictive_samples_12h3b["deaths"][:, is_current_femininity],
        0.89,
    )
    for sample in range(num_samples):
        ax.plot(
            pressures[is_current_femininity],
            prior_predictive_samples_12h3b["deaths"][sample, is_current_femininity],
            color="k",
            alpha=0.2,
        )
    ax.plot(
        pressures[is_current_femininity],
        deaths_mu,
        color="black",
        label=f"femininity={femininity:.2f}",
    )
    ax.fill_between(
        pressures[is_current_femininity],
        deaths_hpdi[0],
        deaths_hpdi[1],
        color="black",
        alpha=0.2,
    )
plt.tight_layout()
```

```python
training_data_12h3b = {
    "pressure": df["min_pressure"].values,
    "femininity": df["femininity"].values,
    "deaths": df["deaths"].values,
}
mcmc_12h3b = MCMC(NUTS(model_12h3b), num_warmup=500, num_samples=1000, num_chains=4)
mcmc_12h3b.run(jrng, **training_data_12h3b)
mcmc_12h3b.print_summary()
idata_12h3b = az.from_numpyro(mcmc_12h3b)
az.plot_trace(
    idata_12h3b,
    kind="rank_bars",
    var_names=["a", "b_feminity", "b_pressure", "b_pressure_feminity", "phi"],
    compact=False,
)
plt.tight_layout()
```

```python
def model_12h3c(damage, pressure, femininity, deaths):
    a = numpyro.sample("a", dist.Normal(2, 1))
    b_feminity = numpyro.sample("b_feminity", dist.Normal(0, 0.5))
    b_damage = numpyro.sample("b_damage", dist.Normal(0, 0.5))
    b_pressure = numpyro.sample("b_pressure", dist.Normal(0, 0.5))
    b_damage_feminity = numpyro.sample("b_damage_femininity", dist.Normal(0, 0.25))
    b_pressure_feminity = numpyro.sample("b_pressure_feminity", dist.Normal(0, 0.25))
    log_lambda = numpyro.deterministic(
        "log_lambda",
        a
        + b_feminity * femininity
        + b_damage * damage
        + b_pressure * pressure
        + b_damage_feminity * damage * femininity
        + b_pressure_feminity * pressure * femininity,
    )
    lambda_ = numpyro.deterministic("lambda", jnp.exp(log_lambda))
    phi = numpyro.sample("phi", dist.Exponential(1))
    numpyro.sample(
        "deaths",
        dist.GammaPoisson(concentration=lambda_ / phi, rate=1 / phi),
        obs=deaths,
    )


training_data_12h3c = {
    "damage": df["damage_norm"].values,
    "pressure": df["min_pressure"].values,
    "femininity": df["femininity"].values,
    "deaths": df["deaths"].values,
}
mcmc_12h3c = MCMC(NUTS(model_12h3c), num_warmup=500, num_samples=1000, num_chains=4)
mcmc_12h3c.run(jrng, **training_data_12h3c)
mcmc_12h3c.print_summary()
idata_12h3c = az.from_numpyro(mcmc_12h3c)
az.plot_trace(
    idata_12h3c,
    kind="rank_bars",
    var_names=[
        "a",
        "b_feminity",
        "b_damage",
        "b_pressure",
        "b_damage_femininity",
        "b_pressure_feminity",
        "phi",
    ],
    compact=False,
)
plt.tight_layout()
```

```python
compare_12h3 = az.compare(
    {
        "mcmc_12h2": idata_12h2,
        "mcmc_12h3a": idata_12h3a,
        "mcmc_12h3b": idata_12h3b,
        "mcmc_12h3c": idata_12h3c,
    },
    scale="deviance",
    ic="waic",
)
display(compare_12h3)
az.plot_compare(compare_12h3)
```

- In all models, femininitiy appears to play no stand-alone role number of deaths (beta is 0).
- The interaction models provide mild support for a modest
- however, interaction models appear to do better than the models without interaction
- deviance is still very large so even interaction model doesn't do that good of a job


### 12H4

```python
df = pd.read_csv("../data/Hurricanes.csv", sep=";")
df["femininity"] = scale(df["femininity"])
df["log_dammage"] = scale(jnp.log(df["damage_norm"].values))
```

```python
def model_12h4(log_damage, femininity, deaths):
    a = numpyro.sample("a", dist.Normal(2, 1))
    b_feminity = numpyro.sample("b_feminity", dist.Normal(0, 0.5))
    b_log_damage = numpyro.sample("b_log_damage", dist.Normal(0, 0.5))
    b_log_damage_feminity = numpyro.sample("b_log_damage_feminity", dist.Normal(0, 0.5))
    log_lambda = numpyro.deterministic(
        "log_lambda",
        a
        + b_feminity * femininity
        + b_log_damage * log_damage
        + b_log_damage_feminity * log_damage * femininity,
    )
    lambda_ = numpyro.deterministic("lambda", jnp.exp(log_lambda))
    phi = numpyro.sample("phi", dist.Exponential(1))
    numpyro.sample(
        "deaths",
        dist.GammaPoisson(concentration=lambda_ / phi, rate=1 / phi),
        obs=deaths,
    )


training_data_12h4 = {
    "log_damage": df["log_dammage"].values,
    "femininity": df["femininity"].values,
    "deaths": df["deaths"].values,
}
mcmc_12h4 = MCMC(NUTS(model_12h4), num_warmup=500, num_samples=1000, num_chains=4)
mcmc_12h4.run(jrng, **training_data_12h4)
mcmc_12h4.print_summary()
idata_12h4 = az.from_numpyro(mcmc_12h4)
az.plot_trace(
    idata_12h4,
    kind="rank_bars",
    var_names=["a", "b_feminity", "b_log_damage", "b_log_damage_feminity"],
    compact=False,
)
plt.tight_layout()
```

```python
compare_12h4 = az.compare(
    {"mcmc_12h3a": idata_12h3c, "mcmc_12h4": idata_12h4}, scale="deviance", ic="waic"
)
display(compare_12h4)
```

```python
is_female = df["female"].astype(bool).values
plt.scatter(df.loc[is_female, "log_dammage"], df.loc[is_female, "deaths"], color="r")
plt.scatter(df.loc[~is_female, "log_dammage"], df.loc[~is_female, "deaths"], color="b")

log_damages = jnp.linspace(-2, 2, 30)
posterior_predictive_12h4 = Predictive(model_12h4, mcmc_12h4.get_samples())
posterior_predictive_female_samples_12h4 = posterior_predictive_12h4(
    jrng, log_damage=log_damages, femininity=1, deaths=None
)
deaths_female_mu_12h4 = posterior_predictive_female_samples_12h4["deaths"].mean(axis=0)
deaths_female_hpdi_12h4 = numpyro.diagnostics.hpdi(
    posterior_predictive_female_samples_12h4["deaths"], 0.89
)
plt.plot(log_damages, deaths_female_mu_12h4, color="red")
plt.fill_between(
    log_damages,
    deaths_female_hpdi_12h4[0],
    deaths_female_hpdi_12h4[1],
    color="red",
    alpha=0.2,
)

posterior_predictive_male_samples_12h4 = posterior_predictive_12h4(
    jrng, log_damage=log_damages, femininity=-1, deaths=None
)
deaths_male_mu_12h4 = posterior_predictive_male_samples_12h4["deaths"].mean(axis=0)
deaths_male_hpdi_12h4 = numpyro.diagnostics.hpdi(
    posterior_predictive_male_samples_12h4["deaths"], 0.89
)
deaths_male_mu_12h4 = posterior_predictive_male_samples_12h4["deaths"].mean(axis=0)
deaths_male_hpdi_12h4 = numpyro.diagnostics.hpdi(
    posterior_predictive_male_samples_12h4["deaths"], 0.89
)
plt.plot(log_damages, deaths_male_mu_12h4, color="blue")
plt.fill_between(
    log_damages,
    deaths_male_hpdi_12h4[0],
    deaths_male_hpdi_12h4[1],
    color="blue",
    alpha=0.2,
)
```

- we have reduced the influence of large storms
- log model appears to do better
- femininity inconsequental on its own, but honestly looks pretty compelling as an interaction

```python
def model_12h4b(log_damage, feminine, deaths):
    a = numpyro.sample("a", dist.Normal(0, 1))
    with numpyro.plate("feminine", size=2):
        b_log_damage = numpyro.sample("b_log_damage", dist.Normal(0, 0.5))
    log_lambda = numpyro.deterministic(
        "log_lambda", a + b_log_damage[feminine] * b_log_damage
    )
    lambda_ = numpyro.deterministic("lambda", jnp.exp(log_lambda))
    numpyro.sample(
        "deaths",
        dist.Poisson(rate=lambda_),
        obs=deaths,
    )


training_data_12h4 = {
    "log_damage": df["log_dammage"].values,
    "femininity": df["femininity"].values,
    "deaths": df["deaths"].values,
}
mcmc_12h4 = MCMC(NUTS(model_12h4), num_warmup=2_000, num_samples=5000, num_chains=4)
mcmc_12h4.run(jrng, **training_data_12h4)
mcmc_12h4.print_summary()
idata_12h4 = az.from_numpyro(mcmc_12h4)
az.plot_trace(
    idata_12h4,
    kind="rank_bars",
    var_names=["a", "b_feminity", "b_log_damage", "b_log_damage_feminity"],
    compact=False,
)
plt.tight_layout()
```

### 12H5

```python
df = pd.read_csv("../data/Trolley.csv", sep=";")
df
```

```python
def model_12h5(gender, action, contact, intention, response):

    cutpoints = numpyro.sample(
        "cutpoints",
        dist.TransformedDistribution(
            dist.Normal(0, 1.5).expand([6]), dist.transforms.OrderedTransform()
        ),
    )
    b_action = numpyro.sample("b_action", dist.Normal(0, 1))
    b_intention = numpyro.sample("b_intention", dist.Normal(0, 1))
    with numpyro.plate(name="gender", size=2):
        b_contact = numpyro.sample("b_contact", dist.Normal(0, 1))
    numpyro.deterministic("diff", b_contact[1] - b_contact[0])
    phi = numpyro.deterministic(
        "phi", b_action * action + b_intention * intention + b_contact[gender] * contact
    )
    response = numpyro.sample(
        "response", dist.OrderedLogistic(phi, cutpoints), obs=response
    )
    return response


training_data = {
    "gender": df["male"].values,
    "action": df["action"].values,
    "contact": df["contact"].values,
    "intention": df["intention"].values,
    "response": df["response"].values - 1,
}
mcmc_12h5 = MCMC(NUTS(model_12h5), num_warmup=500, num_samples=1000, num_chains=4)
mcmc_12h5.run(jrng, **training_data)
idata_12h5 = az.from_numpyro(mcmc_12h5)
az.plot_trace(
    idata_12h5,
    var_names=["b_action", "b_intention", "b_contact", "diff"],
    kind="rank_bars",
    compact=False,
)
plt.tight_layout()
```

Women do appear to be more bothered by contact than men.


### 12H6

```python
df = pd.read_csv("../data/Fish.csv", sep=";")
df
```

```python
def model_12h6(live_bait, camper, persons, children, hours, fish_caught):

    al = numpyro.sample("al", dist.Normal(0, 1))
    bl_live_bait = numpyro.sample("bl_live_bait", dist.Normal(0, 1))
    bl_camper = numpyro.sample("bl_camper", dist.Normal(0, 1))
    bl_persons = numpyro.sample("bl_persons", dist.Normal(0, 1))
    bl_children = numpyro.sample("bl_children", dist.Normal(0, 1))
    log_lambda = numpyro.deterministic(
        "log_lambda",
        al
        + bl_live_bait * live_bait
        + bl_camper * camper
        + bl_persons * persons
        + bl_children * children
        + jnp.log(hours),
    )
    lambda_ = numpyro.deterministic("lambda", jnp.exp(log_lambda))

    ap = numpyro.sample("ap", dist.Normal(0, 1))
    bp_live_bait = numpyro.sample("bp_live_bait", dist.Normal(0, 1))
    bp_camper = numpyro.sample("bp_camper", dist.Normal(0, 1))
    bp_children = numpyro.sample("bp_children", dist.Normal(0, 1))
    logit_p_fishing = (
        ap + bp_live_bait * live_bait + bp_camper * camper + bp_children * children
    )
    p_fishing = numpyro.deterministic(
        "p_fishing", jax.scipy.special.expit(logit_p_fishing)
    )
    numpyro.sample(
        "fish_caught",
        dist.ZeroInflatedPoisson(gate=1 - p_fishing, rate=lambda_),
        obs=fish_caught,
    )


prior_predictive_12h6 = Predictive(model_12h6, num_samples=1_000)
prior_predictive_samples_12h6 = prior_predictive_12h6(
    jrng,
    live_bait=0,
    camper=0,
    persons=1,
    children=0,
    hours=1,
    fish_caught=None,
)
az.plot_kde(prior_predictive_samples_12h6["fish_caught"], hdi_probs=[0.89])
```

```python
mcmc_12h6 = MCMC(NUTS(model_12h6), num_warmup=500, num_samples=1000, num_chains=4)
training_data = {
    "live_bait": df["livebait"].values,
    "camper": df["camper"].values,
    "persons": df["persons"].values,
    "children": df["child"].values,
    "hours": df["hours"].values,
    "fish_caught": df["fish_caught"].values,
}
mcmc_12h6.run(jrng, **training_data)
mcmc_12h6.print_summary()
idata_12h6 = az.from_numpyro(mcmc_12h6)
az.plot_trace(
    idata_12h6,
    var_names=[
        "al",
        "bl_live_bait",
        "bl_camper",
        "bl_persons",
        "bl_children",
        "ap",
        "bp_live_bait",
        "bp_camper",
        "bp_children",
    ],
    kind="rank_bars",
    compact=False,
)
plt.tight_layout()
```

```python

```
