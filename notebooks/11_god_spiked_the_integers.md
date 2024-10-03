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

# Chapter 11: God Spiked the Integers

```python
%load_ext jupyter_black

import inspect
import os
import warnings

import arviz as az
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpyro
import numpyro.distributions as dist
import pandas as pd


from numpyro.infer.autoguide import AutoLaplaceApproximation
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, Predictive
from sklearn.preprocessing import StandardScaler

numpyro.set_host_device_count(os.cpu_count())
optim = numpyro.optim.Adam(step_size=0.1)
loss = Trace_ELBO()

seed = 84735
jrng = jax.random.key(seed)
_, *jrngs = jax.random.split(jrng, 5)
plt.rcParams["figure.figsize"] = [10, 6]

warnings.formatwarning = lambda message, category, *args, **kwargs: "{}: {}\n".format(
    category.__name__, message
)
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
### Code 11.1

```python
df = pd.read_csv("../data/chimpanzees.csv", sep=";")
df
```

### Code. 11.2

```python
df["treatment"] = df["prosoc_left"] + 2 * df["condition"]
```

### Code 11.3

```python
df.reset_index().groupby(["condition", "prosoc_left", "treatment"]).count()["index"]
```

### Code 11.4

```python
def model_11_1(pulled_left):
    a = numpyro.sample("a", dist.Normal(0, 10))
    logit_p = a
    p = numpyro.deterministic("p", 1 / (1 + jnp.exp(-logit_p)))
    pulled_left = numpyro.sample("pulled_left", dist.Bernoulli(p), obs=pulled_left)
    return pulled_left
```

### Code 11.5

```python
prior_predictive_11_1 = Predictive(model=model_11_1, num_samples=10_000)
prior_predictive_samples_11_1 = prior_predictive_11_1(jrng, pulled_left=None)
```

### Code 11.6

```python
az.plot_kde(prior_predictive_samples_11_1["p"])
```

### Code 11.7

```python
def model_11_2(treatment, pulled_left):
    a = numpyro.sample("a", dist.Normal(0, 1.5))
    with numpyro.plate(name="treatment", size=4):
        b = numpyro.sample("b", dist.Normal(0, 10))
    logit_p = a + b[treatment]
    p = numpyro.deterministic("p", 1 / (1 + jnp.exp(-logit_p)))
    pulled_left = numpyro.sample("pulled_left", dist.Bernoulli(p), obs=pulled_left)
    return pulled_left


prior_predictive_11_2 = Predictive(model=model_11_2, num_samples=10_000)
prior_predictive_samples_11_2 = prior_predictive_11_2(
    jrng, pulled_left=None, treatment=None
)
az.plot_kde(
    jnp.abs(
        prior_predictive_samples_11_2["p"][:, 0, 0]
        - prior_predictive_samples_11_2["p"][:, 0, 1]
    )
)
```

### Code 11.9

```python
def model_11_3(treatment, pulled_left):
    a = numpyro.sample("a", dist.Normal(0, 1.5))
    with numpyro.plate(name="treatment", size=4):
        b = numpyro.sample("b", dist.Normal(0, 0.5))
    logit_p = a + b[treatment]
    p = numpyro.deterministic("p", 1 / (1 + jnp.exp(-logit_p)))
    pulled_left = numpyro.sample("pulled_left", dist.Bernoulli(p), obs=pulled_left)
    return pulled_left


prior_predictive_11_3 = Predictive(model=model_11_3, num_samples=10_000)
prior_predictive_samples_11_3 = prior_predictive_11_3(
    jrng, pulled_left=None, treatment=None
)
jnp.abs(
    prior_predictive_samples_11_3["p"][:, 0, 0]
    - prior_predictive_samples_11_3["p"][:, 0, 1]
).mean()
```

### Code 11.10

```python
# prior trimmed data list
training_data = {
    "pulled_left": df["pulled_left"].values,
    "actor": df["actor"].values - 1,  # NB: 0-indexing
    "treatment": df["treatment"].values,
}
```

### Code 11.11

```python
def model_11_4(actor, treatment, pulled_left):
    with numpyro.plate(name="actor", size=7):
        a = numpyro.sample("a", dist.Normal(0, 1.5))
    with numpyro.plate(name="treatment", size=4):
        b = numpyro.sample("b", dist.Normal(0, 0.5))
    logit_p = a[actor] + b[treatment]
    numpyro.deterministic("p_left_handed", 1 / (1 + jnp.exp(-a)))
    numpyro.deterministic("p_treatment", 1 / (1 + jnp.exp(-b)))
    p = numpyro.deterministic("p", 1 / (1 + jnp.exp(-logit_p)))
    pulled_left = numpyro.sample("pulled_left", dist.Bernoulli(p), obs=pulled_left)
    return pulled_left


mcmc_11_4 = MCMC(NUTS(model_11_4), num_warmup=500, num_samples=500, num_chains=4)
mcmc_11_4.run(jrng, **training_data)
mcmc_11_4.print_summary(prob=0.89)
idata_11_4 = az.from_numpyro(mcmc_11_4)
```

```python
az.plot_trace(idata_11_4, var_names=["a", "b"], kind="rank_bars", compact=False)
plt.tight_layout()
```

### Code 11.12

```python
az.plot_forest(idata_11_4.posterior["p_left_handed"], combined=True, hdi_prob=0.89)
```

### Code 11.13

```python
labels = ["R/N", "L/N", "R/P", "L/P"]
az.plot_forest(
    idata_11_4.posterior["b"],
    combined=True,
    hdi_prob=0.89,
)
plt.gca().set_yticklabels(labels[::-1])
```

### Code 11.14

```python
(idata_11_4.posterior["b"][:, :, 0] - idata_11_4.posterior["b"][:, :, 2]).shape
```

```python
contrasts_11_4 = {
    "db31: R/P - R/N": idata_11_4.posterior["b"][:, :, 2]
    - idata_11_4.posterior["b"][:, :, 0],
    "db42: L/P - L/N": idata_11_4.posterior["b"][:, :, 3]
    - idata_11_4.posterior["b"][:, :, 1],
}
az.plot_forest(contrasts_11_4, combined=True, hdi_prob=0.89)
```

### Code 11.15

```python
pl = df.groupby(["actor", "treatment"])["pulled_left"].mean().unstack()
pl
```

### Code 11.16

```python
ax = plt.subplot(
    xlim=(0.5, 28.5),
    ylim=(0, 1.05),
    xlabel="",
    ylabel="proportion left lever",
    xticks=[],
)
plt.yticks(ticks=[0, 0.5, 1], labels=[0, 0.5, 1])
ax.axhline(0.5, c="k", lw=1, ls="--")
for j in range(1, 8):
    ax.axvline((j - 1) * 4 + 4.5, c="k", lw=0.5)
for j in range(1, 8):
    ax.annotate(
        "actor {}".format(j),
        ((j - 1) * 4 + 2.5, 1.1),
        ha="center",
        va="center",
        annotation_clip=False,
    )
for j in [1] + list(range(3, 8)):
    ax.plot((j - 1) * 4 + jnp.array([1, 3]), pl.loc[j, [0, 2]], "b")
    ax.plot((j - 1) * 4 + jnp.array([2, 4]), pl.loc[j, [1, 3]], "b")
x = jnp.arange(1, 29).reshape(7, 4)
ax.scatter(
    x[:, [0, 1]].reshape(-1),
    pl.values[:, [0, 1]].reshape(-1),
    edgecolor="b",
    facecolor="w",
    zorder=3,
)
ax.scatter(
    x[:, [2, 3]].reshape(-1), pl.values[:, [2, 3]].reshape(-1), marker=".", c="b", s=80
)
yoff = 0.01
ax.annotate("R/N", (1, pl.loc[1, 0] - yoff), ha="center", va="top")
ax.annotate("L/N", (2, pl.loc[1, 1] + yoff), ha="center", va="bottom")
ax.annotate("R/P", (3, pl.loc[1, 2] - yoff), ha="center", va="top")
ax.annotate("L/P", (4, pl.loc[1, 3] + yoff), ha="center", va="bottom")
ax.set_title("observed proportions\n")
plt.show()
```

### Code 11.17

```python
actors = jnp.repeat(jnp.arange(7), 4)
treatments = jnp.tile(jnp.arange(4), 7)
posterior_predictive_11_4 = Predictive(model_11_4, mcmc_11_4.get_samples())
posterior_predictive_samples_11_4 = posterior_predictive_11_4(
    jrng,
    actor=actors,
    treatment=treatments,
    pulled_left=None,
)
p_mu = posterior_predictive_samples_11_4["p"].mean(axis=0)
p_ci = numpyro.diagnostics.hpdi(posterior_predictive_samples_11_4["p"], 0.89)
```

### Code 11.18

```python
df["side"] = df["prosoc_left"]  # right 0, left 1
df["cond"] = df["condition"]  # no partner 0, partner 1
```

### Code 11.19

```python
def model_11_5(actor, side, cond, pulled_left):
    with numpyro.plate(name="actor", size=7):
        a = numpyro.sample("a", dist.Normal(0, 1.5))
    with numpyro.plate(name="side", size=2):
        b_side = numpyro.sample("b_side", dist.Normal(0, 0.5))
    with numpyro.plate(name="cond", size=2):
        b_cond = numpyro.sample("b_cond", dist.Normal(0, 0.5))
    logit_p = a[actor] + b_side[side] + b_cond[cond]
    p = numpyro.deterministic("p", 1 / (1 + jnp.exp(-logit_p)))
    pulled_left = numpyro.sample("pulled_left", dist.Bernoulli(p), obs=pulled_left)
    return pulled_left


training_data = {
    "actor": df["actor"].values - 1,
    "side": df["side"].values,
    "cond": df["cond"].values,
    "pulled_left": df["pulled_left"].values,
}
mcmc_11_5 = MCMC(NUTS(model_11_5), num_warmup=500, num_samples=1_000, num_chains=4)
mcmc_11_5.run(jrng, **training_data)
idata_11_5 = az.from_numpyro(mcmc_11_5)
mcmc_11_5.print_summary(prob=0.89)
```

```python
az.plot_trace(
    mcmc_11_5, var_names=["a", "b_side", "b_cond"], compact=False, kind="rank_bars"
)
plt.tight_layout()
```

### Code 11.20

```python
compare = az.compare(
    {"m11.4": idata_11_4, "m11.5": idata_11_5}, ic="loo", scale="deviance"
)
display(compare)
az.plot_compare(compare)
```

### Code 11.21


### Code 11.22

```python
training_data = {
    "actor": df["actor"].values - 1,
    "treatment": df["treatment"].values,
    "pulled_left": df["pulled_left"].values,
}


def model_11_4_pe(params, data=training_data, log_likelihood=False):
    a_logprob = jnp.sum(dist.Normal(0, 1.5).log_prob(params["a"]))
    b_logprob = jnp.sum(dist.Normal(0, 0.5).log_prob(params["b"]))
    logit_p = params["a"][data["actor"]] + params["b"][data["treatment"]]
    pulled_left_logprob = dist.Bernoulli(logits=logit_p).log_prob(data["pulled_left"])
    if log_likelihood:
        return pulled_left_logprob
    potential_energy = -(a_logprob + b_logprob + jnp.sum(pulled_left_logprob))
    return potential_energy


init_params = {"a": jnp.zeros((4, 7)), "b": jnp.zeros((4, 7))}
mcmc_11_4_pe = MCMC(
    NUTS(potential_fn=model_11_4_pe), num_warmup=1_000, num_samples=1_000, num_chains=4
)

# draw posterior samples according to the potential energy function
# corresponding to model 11.4
mcmc_11_4_pe.run(jrng, init_params=init_params)

# now compute the log likelihood of the observed data for each posterior sample
log_likelihood = jax.vmap(lambda p: model_11_4_pe(params=p, log_likelihood=True))(
    mcmc_11_4_pe.get_samples()
)
```

```python
idata_11_4_pe = az.from_numpyro(mcmc_11_4_pe)
idata_11_4_pe.sample_stats["log_likelihood"] = (
    ("chain", "draw", "pulled_left_dim_0"),
    jnp.reshape(log_likelihood, (4, 1_000, -1)),
)
compare = az.compare(
    {"m11.4": idata_11_4, "m11.4_pe": idata_11_4_pe}, ic="waic", scale="deviance"
)
compare
```

### Code 11.23

```python
posterior_samples_11_4 = mcmc_11_4.get_samples()
jnp.exp(posterior_samples_11_4["b"][:, 3] - posterior_samples_11_4["b"][:, 1]).mean()
```

### Code 11.24

```python
df = pd.read_csv("../data/chimpanzees.csv", sep=";")
df["treatment"] = df["prosoc_left"] + 2 * df["condition"]
df["side"] = df["prosoc_left"]  # right 0, left 1
df["cond"] = df["condition"]  # no partner 0, partner 1
df["actor"] = df["actor"] - 1  # 0-indexing
df_aggregated = (
    df.groupby(["actor", "treatment", "side", "cond"])["pulled_left"]
    .sum()
    .reset_index()
    .rename(columns={"pulled_left": "left_pulls"})
)
df_aggregated
```

### Code 11.25

```python
def model_11_6(actor, treatment, left_pulls):
    with numpyro.plate(name="actor", size=7):
        a = numpyro.sample("a", dist.Normal(0, 1.5))
    with numpyro.plate(name="treatment", size=4):
        b = numpyro.sample("b", dist.Normal(0, 0.5))
    logit_p = a[actor] + b[treatment]
    p = numpyro.deterministic("p", 1 / (1 + jnp.exp(-logit_p)))
    left_pulls = numpyro.sample("left_pulls", dist.Binomial(18, p), obs=left_pulls)
    return left_pulls


training_data = {
    "actor": df_aggregated["actor"].values,
    "treatment": df_aggregated["treatment"].values,
    "left_pulls": df_aggregated["left_pulls"].values,
}
mcmc_11_6 = MCMC(NUTS(model_11_6), num_warmup=500, num_samples=1_000, num_chains=4)
mcmc_11_6.run(jrng, **training_data)
mcmc_11_6.print_summary(prob=0.89)
idata_11_6 = az.from_numpyro(mcmc_11_6)
az.plot_trace(idata_11_6, var_names=["a", "b"], kind="rank_bars", compact=False)
```

### Code 11.26

```python
try:
    az.waic(idata_11_4)
    az.waic(idata_11_6)
    az.compare({"m11.4": idata_11_4, "m11.6": idata_11_6}, ic="loo", scale="deviance")
except Exception as e:
    warnings.warn(f"\n{type(e).__name__}: {e}")
```

### Code 11.27

```python
# deviance of aggregated 6-in-9
display(-2 * dist.Binomial(9, 0.2).log_prob(6))
# deviance of disaggregated
-2 * dist.Bernoulli(0.2).log_prob(jnp.array([1] * 6 + [0] * 3)).sum()
```

### Code 11.28

```python
df = pd.read_csv("../data/UCBadmit.csv", sep=";")
df
```

### Code 11.29

```python
df["gender"] = (df["applicant.gender"] == "female").astype(int)


def model_11_7(gender, applications, admit):
    with numpyro.plate(name="gender", size=2):
        a = numpyro.sample("a", dist.Normal(0, 1.5))
    logit_p = a[gender]
    p = numpyro.deterministic("p", 1 / (1 + jnp.exp(-logit_p)))
    admit = numpyro.sample("admit", dist.Binomial(applications, p), obs=admit)
    return admit


prior_predictive_11_7 = Predictive(model_11_7, num_samples=10_000)
prior_predictive_samples_11_7 = prior_predictive_11_7(
    jrng, gender=None, applications=500, admit=None
)
ax = az.plot_kde(prior_predictive_samples_11_7["p"])
ax.set_title("prior predictive distribution of p")
ax.set_xlabel("p")
```

```python
training_data = {
    "gender": df["gender"].values,
    "applications": df["applications"].values,
    "admit": df["admit"].values,
}
mcmc_11_7 = MCMC(NUTS(model_11_7), num_warmup=500, num_samples=500, num_chains=4)
mcmc_11_7.run(jrng, **training_data)
mcmc_11_7.print_summary(prob=0.89)
```

### Code 11.30

```python
posterior_samples_11_7 = mcmc_11_7.get_samples()
diff_a = posterior_samples_11_7["a"][:, 0] - posterior_samples_11_7["a"][:, 1]
diff_p = posterior_samples_11_7["p"][:, 0] - posterior_samples_11_7["p"][:, 1]
numpyro.diagnostics.print_summary(
    {"diff_a": diff_a, "diff_p": diff_p}, prob=0.89, group_by_chain=False
)
```

### Code 11.31

```python
# compute and plot in-sample predictions
posterior_predictive_11_7 = Predictive(model_11_7, posterior_samples_11_7)
posterior_predictive_samples_11_7 = posterior_predictive_11_7(
    jrngs[0],
    gender=training_data["gender"],
    applications=training_data["applications"],
    admit=None,
)
admit_rate = posterior_predictive_samples_11_7["admit"] / training_data["applications"]
plt.errorbar(
    x=range(1, 13),
    y=admit_rate.mean(axis=0),
    yerr=admit_rate.std(axis=0) / 2,
    fmt="o",
    c="k",
    mfc="none",
    ms=7,
    elinewidth=1,
)
hpdi_11_7 = numpyro.diagnostics.hpdi(admit_rate, 0.89)
plt.plot(range(1, 13), hpdi_11_7[0, :], "k+")
plt.plot(range(1, 13), hpdi_11_7[1, :], "k+")

# plot actuals
for i in range(1, 7):
    x = 1 + 2 * (i - 1)
    y1 = df["admit"].iloc[x - 1] / df["applications"].iloc[x - 1]
    y2 = df["admit"].iloc[x] / df["applications"].iloc[x]
    plt.plot((x, x + 1), (y1, y2), "bo-")
    plt.annotate(
        df["dept"].iloc[x],
        (x + 0.5, (y1 + y2) / 2 + 0.05),
        ha="center",
        color="royalblue",
    )
```

### Code 11.32

```python
df["dept_id"] = df["dept"].astype("category").cat.codes


def model_11_8(gender, dept_id, applications, admit):
    with numpyro.plate(name="gender", size=2):
        a = numpyro.sample("a", dist.Normal(0, 1.5))
    with numpyro.plate(name="dept_id", size=6):
        b = numpyro.sample("b", dist.Normal(0, 1.5))
    logit_p = a[gender] + b[dept_id]
    p = numpyro.deterministic("p", 1 / (1 + jnp.exp(-logit_p)))
    admit = numpyro.sample("admit", dist.Binomial(applications, p), obs=admit)
    return admit


prior_predictive_11_8 = Predictive(model_11_8, num_samples=10_000)
prior_predictive_samples_11_8 = prior_predictive_11_8(
    jrng,
    gender=0,
    dept_id=0,
    applications=500,
    admit=None,
)
az.plot_kde(prior_predictive_samples_11_8["p"])
print(
    "We would probably be better off using a slightly more informative prior e.g. N(0, 1)."
)
```

```python
training_data = {
    "gender": df["gender"].values,
    "dept_id": df["dept_id"].values,
    "applications": df["applications"].values,
    "admit": df["admit"].values,
}

mcmc_11_8 = MCMC(NUTS(model_11_8), num_warmup=1_000, num_samples=3_000, num_chains=4)
mcmc_11_8.run(jrng, **training_data)
mcmc_11_8.print_summary(prob=0.89)
```

### Code 11.33

```python
posterior_samples_11_8 = mcmc_11_8.get_samples()
diff_a = posterior_samples_11_8["a"][:, 0] - posterior_samples_11_8["a"][:, 1]
diff_p = posterior_samples_11_8["p"][:, 0] - posterior_samples_11_8["p"][:, 1]
numpyro.diagnostics.print_summary(
    {"diff_a": diff_a, "diff_p": diff_p}, prob=0.89, group_by_chain=False
)
```

### Code 11.34

```python vscode={"languageId": "ruby"}
total_applicants = df.groupby("dept")["applications"].sum()
gender_applicants = (
    df.groupby(["dept", "applicant.gender"])["applications"].sum().unstack()
)
gender_percentage = (gender_applicants.div(total_applicants, axis=0)).round(2).T

gender_percentage
```

### Code 11.35

```python
y = dist.Binomial(total_count=1_000, probs=1 / 1000).sample(
    jrng, sample_shape=(10_000,)
)
y.mean(), y.std()
```

### Code 11.36

```python
df = pd.read_csv("../data/Kline.csv", sep=";")
df
```

### Code 11.37

```python
df["log_population"] = jnp.log(df["population"].values)
log_populataion_scaler = StandardScaler().fit(df[["log_population"]])
df["P"] = log_populataion_scaler.transform(df[["log_population"]])
df["contact_id"] = df["contact"].astype("category").cat.codes
df
```

### Code 11.38

```python
x = jnp.linspace(0, 100, 200)
plt.plot(x, jnp.exp(dist.LogNormal(0, 10).log_prob(x)))
```

### Code 11.39

```python
a = dist.Normal(0, 10).sample(jrng, sample_shape=(10_000,))
lambda_ = jnp.exp(a)
lambda_.mean()
```

### Code 11.40

```python
x = jnp.linspace(0, 100, 200)
plt.plot(x, jnp.exp(dist.LogNormal(3, 0.5).log_prob(x)))
```

### Code 11.41

```python
N = 100
a = dist.Normal(3, 0.5).sample(jrng, sample_shape=(N,))
b = dist.Normal(0, 10).sample(jrng, sample_shape=(N,))
plt.subplot(xlim=(-2, 2), ylim=(0, 100))
x = jnp.linspace(-2, 2, 100)
for i in range(N):
    plt.plot(x, jnp.exp(a[i] + b[i] * x), "k", alpha=0.1)
```

### Code 11.42

```python
N = 100
a = dist.Normal(3, 0.5).sample(jrng, sample_shape=(N,))
b = dist.Normal(0, 0.2).sample(jrng, sample_shape=(N,))
plt.subplot(xlim=(-2, 2), ylim=(0, 100))
x = jnp.linspace(-2, 2, 100)
for i in range(N):
    plt.plot(x, jnp.exp(a[i] + b[i] * x), "k", alpha=0.1)
```

### Code 11.43

```python
n_plot_points = 100
x_seq = jnp.linspace(jnp.log(100), jnp.log(200_000), n_plot_points)
lambda_ = jax.vmap(lambda x: jnp.exp(a + b * x), out_axes=1)(x_seq)
plt.subplot(
    xlim=(x_seq[0], x_seq[-1]),
    ylim=(0, 500),
    xlabel="log population",
    ylabel="total tools",
)
for i in range(N):
    plt.plot(x_seq, lambda_[i], "k", alpha=0.5)
```

### Code 11.44

```python
plt.subplot(
    xlim=(jnp.exp(x_seq[0]), jnp.exp(x_seq[-1])),
    ylim=(0, 500),
    xlabel="population",
    ylabel="total tools",
)
for i in range(N):
    plt.plot(jnp.exp(x_seq), lambda_[i], "k", alpha=0.5)
```

### Code 11.45

```python
training_data = {
    "tools": df["total_tools"].values,
}


def model_11_9(tools):
    a = numpyro.sample("a", dist.Normal(3, 0.5))
    lambda_ = numpyro.deterministic("lambda", jnp.exp(a))
    tools = numpyro.sample("tools", dist.Poisson(rate=lambda_), obs=tools)
    return tools


mcmc_11_9 = MCMC(NUTS(model_11_9), num_warmup=500, num_samples=500, num_chains=4)
mcmc_11_9.run(jrng, **training_data)
mcmc_11_9.print_summary(prob=0.89)
idata_11_9 = az.from_numpyro(mcmc_11_9)
az.plot_trace(idata_11_9, compact=False, var_names=["a"], kind="rank_bars")
plt.tight_layout()
```

```python
training_data = {
    "tools": df["total_tools"].values,
    "P": df["P"].values,
    "contact_id": df["contact_id"].values,
}


def model_11_10(contact_id, P, tools):
    with numpyro.plate(name="contact", size=2):
        a = numpyro.sample("a", dist.Normal(3, 0.5))
        b = numpyro.sample("b", dist.Normal(0, 0.2))
    lambda_ = numpyro.deterministic(
        "lambda", jnp.exp(a[contact_id] + b[contact_id] * P)
    )
    tools = numpyro.sample("tools", dist.Poisson(rate=lambda_), obs=tools)
    return tools


mcmc_11_10 = MCMC(NUTS(model_11_10), num_warmup=500, num_samples=500, num_chains=4)
mcmc_11_10.run(jrng, **training_data)
mcmc_11_10.print_summary(prob=0.89)
idata_11_10 = az.from_numpyro(mcmc_11_10)
az.plot_trace(idata_11_10, compact=False, var_names=["a", "b"], kind="rank_bars")

plt.tight_layout()
```

### Code 11.46

```python
compare_11_1 = az.compare(
    {"m11.9": idata_11_9, "m11.10": idata_11_10}, ic="loo", scale="deviance"
)
compare_11_1
```

### Code 11.47

```python
k = az.loo(idata_11_10, pointwise=True)["pareto_k"].values
plt.subplot(xlabel="log population", ylabel="total tools", ylim=(0, 75))
cex = 1 + (k - jnp.min(k)) / (jnp.max(k) - jnp.min(k))
plt.scatter(
    df["P"],
    df["total_tools"],
    s=40 * cex,
    edgecolors=["b" if x else "none" for x in df["contact_id"]],
    facecolors=["none" if x else "b" for x in df["contact_id"]],
)

# set up the horizontal axis values to compute predictions at
ns = 100
P_seq = jnp.linspace(-1.4, 3, ns)

# predictions for cid=1 (low contact)
posterior_predictive_11_10 = Predictive(model_11_10, mcmc_11_10.get_samples())
posterior_predictive_samples_11_10 = posterior_predictive_11_10(
    jrng, contact_id=1, P=P_seq, tools=None
)
lambda_mu_11_10_1 = posterior_predictive_samples_11_10["lambda"].mean(axis=0)
lambda_hpdi_11_10_1 = numpyro.diagnostics.hpdi(
    posterior_predictive_samples_11_10["lambda"], 0.89
)
plt.plot(P_seq, lambda_mu_11_10_1, "k--")
plt.fill_between(
    P_seq, lambda_hpdi_11_10_1[0], lambda_hpdi_11_10_1[1], color="k", alpha=0.3
)

# predictions for cid=0 (high contact)
posterior_predictive_samples_11_10 = posterior_predictive_11_10(
    jrng, contact_id=0, P=P_seq, tools=None
)
lambda_mu_11_10_2 = posterior_predictive_samples_11_10["lambda"].mean(axis=0)
lambda_hpdi_11_10_2 = numpyro.diagnostics.hpdi(
    posterior_predictive_samples_11_10["lambda"], 0.89
)
plt.plot(P_seq, lambda_mu_11_10_2, "k", lw=1.5)
plt.fill_between(
    P_seq, lambda_hpdi_11_10_2[0], lambda_hpdi_11_10_2[1], color="k", alpha=0.3
)
```

### Code 11.48

```python
plt.subplot(xlabel="population", ylabel="total tools", xlim=(0, 300_000), ylim=(0, 75))
plt.scatter(
    df["population"],
    df["total_tools"],
    s=40 * cex,
    edgecolors=["b" if x else "none" for x in df["contact_id"]],
    facecolors=["none" if x else "b" for x in df["contact_id"]],
)

# set up the horizontal axis values to compute predictions at
ns = 100
P_seq = jnp.linspace(-5, 3, ns)
population_seq = jnp.exp(
    log_populataion_scaler.inverse_transform(P_seq[:, None]).flatten()
)

# predictions for cid=1 (low contact)
posterior_predictive_samples_11_10 = posterior_predictive_11_10(
    jrngs[0], contact_id=1, P=P_seq, tools=None
)
lambda_mu_11_10_1 = posterior_predictive_samples_11_10["lambda"].mean(axis=0)
lambda_hpdi_11_10_1 = numpyro.diagnostics.hpdi(
    posterior_predictive_samples_11_10["lambda"], 0.89
)
plt.plot(population_seq, lambda_mu_11_10_1, "k--")
plt.fill_between(
    population_seq, lambda_hpdi_11_10_1[0], lambda_hpdi_11_10_1[1], color="k", alpha=0.3
)

# predictions for cid=0 (high contact)
posterior_predictive_samples_11_10 = posterior_predictive_11_10(
    jrngs[1], contact_id=0, P=P_seq, tools=None
)
lambda_mu_11_10_2 = posterior_predictive_samples_11_10["lambda"].mean(axis=0)
lambda_hpdi_11_10_2 = numpyro.diagnostics.hpdi(
    posterior_predictive_samples_11_10["lambda"], 0.89
)
plt.plot(population_seq, lambda_mu_11_10_2, "k", lw=1.5)
plt.fill_between(
    population_seq, lambda_hpdi_11_10_2[0], lambda_hpdi_11_10_2[1], color="k", alpha=0.3
)
```

### Code 11.49

```python
def model_11_11(contact_id, population, tools):
    with numpyro.plate(name="contact", size=2):
        a = numpyro.sample("a", dist.LogNormal(1, 1))
        b = numpyro.sample("b", dist.Exponential(1))
    g = numpyro.sample("g", dist.Exponential(1))
    lambda_ = numpyro.deterministic(
        "lambda", a[contact_id] * jnp.power(population, b[contact_id]) / g
    )
    tools = numpyro.sample("tools", dist.Poisson(rate=lambda_), obs=tools)
    return tools


prior_predictive_11_11 = Predictive(model_11_11, num_samples=100)
populations = jnp.linspace(0, 300_000, 200)
prior_predictive_samples_11_11 = prior_predictive_11_11(
    jrng,
    contact_id=0,  # doesn't matter which, since they have the same prior
    population=populations,
    tools=None,
)
plt.subplot(
    title="Prior Predictive Samples",
    xlim=(0, 300_000),
    # ylim=(0, 100),
    xlabel="population",
    ylabel="total tools",
)
for i in range(100):
    plt.plot(populations, prior_predictive_samples_11_11["tools"][i], "k", alpha=0.3)
```

```python
training_data = {
    "tools": df["total_tools"].values,
    "population": df["population"].values,
    "contact_id": df["contact_id"].values,
}

mcmc_11_11 = MCMC(NUTS(model_11_11), num_warmup=500, num_samples=1_000, num_chains=4)
mcmc_11_11.run(jrng, **training_data)
mcmc_11_11.print_summary(prob=0.89)
idata_11_11 = az.from_numpyro(mcmc_11_11)
az.plot_trace(
    idata_11_11,
    var_names=["a", "b", "g"],
    kind="rank_bars",
    compact=False,
    divergences=True,
)
plt.tight_layout()
```

```python
plt.subplot(xlabel="population", ylabel="total tools", xlim=(0, 300_000), ylim=(0, 75))
plt.scatter(
    df["population"],
    df["total_tools"],
    s=40 * cex,
    edgecolors=["b" if x else "none" for x in df["contact_id"]],
    facecolors=["none" if x else "b" for x in df["contact_id"]],
)

# set up the horizontal axis values to compute predictions at
ns = 100
population_seq = jnp.linspace(0, 300_000, ns)

# predictions for cid=1 (low contact)
posterior_predictive_11_11 = Predictive(model_11_11, mcmc_11_11.get_samples())
posterior_predictive_samples_11_11 = posterior_predictive_11_11(
    jrngs[0], contact_id=1, population=population_seq, tools=None
)
lambda_mu_11_11_1 = posterior_predictive_samples_11_11["lambda"].mean(axis=0)
lambda_hpdi_11_11_1 = numpyro.diagnostics.hpdi(
    posterior_predictive_samples_11_11["lambda"], 0.89
)
plt.plot(population_seq, lambda_mu_11_11_1, "k--")
plt.fill_between(
    population_seq, lambda_hpdi_11_11_1[0], lambda_hpdi_11_11_1[1], color="k", alpha=0.3
)

# predictions for cid=0 (high contact)
posterior_predictive_samples_11_11 = posterior_predictive_11_11(
    jrngs[1], contact_id=0, population=population_seq, tools=None
)
lambda_mu_11_11_2 = posterior_predictive_samples_11_11["lambda"].mean(axis=0)
lambda_hpdi_11_11_2 = numpyro.diagnostics.hpdi(
    posterior_predictive_samples_11_11["lambda"], 0.89
)
plt.plot(population_seq, lambda_mu_11_11_2, "k", lw=1.5)
plt.fill_between(
    population_seq, lambda_hpdi_11_11_2[0], lambda_hpdi_11_11_2[1], color="k", alpha=0.3
)
```

### Code 11.50

```python
num_days = 30
y = dist.Poisson(1.5).sample(jrng, sample_shape=(num_days,))
y
```

### Code 11.51

```python
num_weeks = 4
y_new = dist.Poisson(0.5 * 7).sample(jrng, sample_shape=(num_weeks,))
y_new
```

### Code 11.52

```python
df = pd.DataFrame(
    {
        "y": jnp.concatenate([y, y_new]),
        "days": [1] * num_days + [7] * num_weeks,
        "monastery": [0] * num_days + [1] * num_weeks,
    }
)
df
```

### Code 11.53

```python
df["log_days"] = jnp.log(df["days"].values)


def model_11_12(monastery, log_days, y):
    with numpyro.plate(name="monastery", size=2):
        a = numpyro.sample("a", dist.Normal(0, 1))
    lambda_ = numpyro.deterministic("lambda", jnp.exp(log_days + a[monastery]))
    y = numpyro.sample("y", dist.Poisson(lambda_), obs=y)
    return y


prior_predictive_11_12 = Predictive(model_11_12, num_samples=1_000)
prior_predictive_samples_11_12 = prior_predictive_11_12(
    jrng, monastery=1, log_days=1, y=None
)
plt.subplot(xlim=(0, 25), ylim=(0, 1))
az.plot_kde(prior_predictive_samples_11_12["y"])
```

### Code 11.54

```python
train_data = {
    "monastery": df["monastery"].values,
    "log_days": df["log_days"].values,
    "y": df["y"].values,
}
mcmc_11_12 = MCMC(NUTS(model_11_12), num_warmup=500, num_samples=500, num_chains=4)
mcmc_11_12.run(jrng, **train_data)
mcmc_11_12.print_summary(prob=0.89)
numpyro.diagnostics.print_summary(
    {"lambda": jnp.exp(mcmc_11_12.get_samples()["a"])},
    prob=0.89,
    group_by_chain=False,
)
idata_11_12 = az.from_numpyro(mcmc_11_12)
az.plot_trace(
    idata_11_12,
    var_names=["a"],
    kind="rank_bars",
    compact=False,
    divergences=True,
)
plt.tight_layout()
```

### Code 11.55

```python
# simulat career choices among 500 individuals
N = 500  # number of individuals
career_income = jnp.array([1, 2, 5])  # expected income for each career
score = 0.5 * career_income  # score for each career
# next line convers score to probabilities
p = jax.nn.softmax(score)

# now simulate career choices
# outcome career holds event type values, not counts
career = dist.Categorical(probs=p).sample(jrng, sample_shape=(N,))
```

### Code 11.56

```python
def model_11_13(K, career_income, career):
    with numpyro.plate(name="career_income", size=K - 1):
        a = numpyro.sample("a", dist.Normal(0, 1))

    # association of income with choice
    b = numpyro.sample("b", dist.Normal(0.5))
    score = jnp.concat([a, jnp.zeros(1)]) + b * career_income
    p = numpyro.deterministic("p", jax.nn.softmax(score))
    career = numpyro.sample("career", dist.Categorical(probs=p), obs=career)
    return career
```

### Code 11.57

```python
train_data = {"K": 3, "career_income": career_income, "career": career}
mcmc_11_13 = MCMC(NUTS(model_11_13), num_warmup=1_000, num_samples=1_000, num_chains=4)
mcmc_11_13.run(jrng, **train_data)
mcmc_11_13.print_summary(prob=0.89)
idata_11_13 = az.from_numpyro(mcmc_11_13)
az.plot_trace(idata_11_13, var_names=["p"], kind="rank_bars", compact=False)
plt.tight_layout()
```

### Code 11.58

```python
posterior_samples_11_13 = mcmc_11_13.get_samples()
p_orig = posterior_samples_11_13["p"]
a = jnp.concat([posterior_samples_11_13["a"], jnp.zeros((4_000, 1))], axis=1)
career_income_cf = career_income * jnp.array([1, 2, 1])
score_cf = a + posterior_samples_11_13["b"][:, None] * career_income_cf
p_cf = jax.nn.softmax(score_cf)
p_diff = p_cf[:, 1] - p_orig[:, 1]
numpyro.diagnostics.print_summary(
    {"p_diff": p_diff},
    prob=0.89,
    group_by_chain=False,
)
```

###  Code 11.59

```python
N = 500
# simulate family income for each individual
family_income = dist.Uniform(0, 1).sample(jrng, sample_shape=(N,))
# assign a unique coefficient for each type of event
b = jnp.array([-2, 0, 2])
score = 0.5 * jnp.arange(1, 4) + b * family_income[:, None]
p = jax.nn.softmax(score)
career = dist.Categorical(probs=p).sample(jrng)


def model_11_14(K, family_income, career):
    with numpyro.plate(name="family_income", size=K - 1):
        a = numpyro.sample("a", dist.Normal(0, 1.5))
        b = numpyro.sample("b", dist.Normal(1))
    score = a + b * family_income[:, None]
    score = jnp.concatenate([score, jnp.zeros((family_income.shape[0], 1))], axis=1)
    p = numpyro.deterministic("p", jax.nn.softmax(score))
    career = numpyro.sample("career", dist.Categorical(probs=p), obs=career)
    return career


train_data = {"K": 3, "family_income": family_income, "career": career}
mcmc_11_14 = MCMC(NUTS(model_11_14), num_warmup=1_000, num_samples=1_000, num_chains=4)
mcmc_11_14.run(jrng, **train_data)
mcmc_11_14.print_summary(prob=0.89)
idata_11_14 = az.from_numpyro(mcmc_11_14)
az.plot_trace(idata_11_14, var_names=["a", "b"], kind="rank_bars", compact=False)
plt.tight_layout()
```

### Code 11.60

```python
df = pd.read_csv("../data/UCBadmit.csv", sep=";")
df
```

### Code 11.61

```python
# binomial model of overall admission probability
def model_binom(applications, admit):
    a = numpyro.sample("a", dist.Normal(0, 1.5))
    logit_p = a
    p = numpyro.deterministic("p", jax.scipy.special.expit(logit_p))
    admit = numpyro.sample(
        "admit",
        dist.Binomial(total_count=applications, probs=p),
        obs=admit,
    )
    return admit


training_data = {"applications": df["applications"].values, "admit": df["admit"].values}
mcmc_binom = MCMC(NUTS(model_binom), num_warmup=500, num_samples=1_000, num_chains=4)
mcmc_binom.run(jrng, **training_data)
mcmc_binom.print_summary(prob=0.89)
numpyro.diagnostics.print_summary(
    {"p": mcmc_binom.get_samples()["p"]}, prob=0.89, group_by_chain=False
)
idata_binom = az.from_numpyro(mcmc_binom)
az.plot_trace(idata_binom, var_names=["p"], kind="rank_bars", compact=False)
plt.tight_layout()
```

```python
# poisson model of overall admision and rejections rate
def model_poisson(admit, reject):
    a_admit = numpyro.sample("a_admit", dist.Normal(0, 1.5))
    a_reject = numpyro.sample("a_reject", dist.Normal(0, 1.5))
    log_lambda_admit = a_admit
    log_lambda_reject = a_reject
    lambda_admit = jnp.exp(log_lambda_admit)
    lambda_reject = jnp.exp(log_lambda_reject)
    p = numpyro.deterministic("p", lambda_admit / (lambda_admit + lambda_reject))
    admit = numpyro.sample("admit", dist.Poisson(lambda_admit), obs=admit)
    reject = numpyro.sample("reject", dist.Poisson(lambda_reject), obs=reject)
    return {"admit": admit, "reject": reject}


training_data = {"admit": df["admit"].values, "reject": df["reject"].values}
mcmc_poisson = MCMC(
    NUTS(model_poisson), num_warmup=500, num_samples=1_000, num_chains=4
)
mcmc_poisson.run(jrng, **training_data)
mcmc_poisson.print_summary(prob=0.89)
numpyro.diagnostics.print_summary(
    {"p": mcmc_poisson.get_samples()["p"]},
    prob=0.89,
    group_by_chain=False,
)
idata_poisson = az.from_numpyro(mcmc_poisson)
az.plot_trace(
    idata_poisson,
    var_names=["a_admit", "a_reject", "p"],
    kind="rank_bars",
    compact=False,
)
plt.tight_layout()
```

### Code 11.62

```python
jax.scipy.special.expit(mcmc_binom.get_samples()["a"].mean())
```

### Code 11.63

```python
lambda_admit = jnp.exp(mcmc_poisson.get_samples()["a_admit"].mean())
lambda_reject = jnp.exp(mcmc_poisson.get_samples()["a_reject"].mean())
lambda_admit / (lambda_admit + lambda_reject)
```

### Code 11.64

```python
x2 = jnp.min(dist.Uniform(1, 100).sample(jrng, sample_shape=(100_000, 2)), axis=1)
x5 = jnp.min(dist.Uniform(1, 100).sample(jrng, sample_shape=(100_000, 5)), axis=1)
ax = az.plot_kde(x2, label="x2")
az.plot_kde(x5, ax=ax, label="x5", plot_kwargs={"color": "k"})
```

### Code 11.65

```python
N = 10
M = 2
x = jnp.sort(dist.Uniform(1, 100).sample(jrng, sample_shape=(100_000, N)), axis=1)[
    :, M - 1
]
az.plot_kde(x, label="Gamma 2/10 parts")
```

### Code 11.66

```python
df = pd.read_csv("../data/AustinCats.csv", sep=";")
df["is_adopted"] = df["out_event"] == "Adoption"
df["is_black"] = (df["color"].map(lambda x: x.lower()) == "black").astype(int)
df
```

```python
def model_11_15(days_to_event, is_adopted, is_black):
    with numpyro.plate("is_black", size=2):
        a = numpyro.sample("a", dist.Normal(0, 1))
    link = jnp.exp(a[is_black])
    lambda_ = numpyro.deterministic("lambda", 1 / link)

    with numpyro.handlers.mask(mask=is_adopted):
        days_to_adoption = numpyro.sample(
            "days_to_adoption",
            dist.Exponential(lambda_),
            obs=days_to_event,
        )

    with numpyro.handlers.mask(mask=~is_adopted):
        # CDF of exponentional: 1 - exp(-lambda * x)
        # Complemetary CDF: exp(-lambda * x)
        # potential energy aka log probability: -lambda * x
        days_without_adoption = numpyro.factor(
            name="days_without_adoption", log_factor=-lambda_ * days_to_event
        )
    return {
        "days_to_adoption": days_to_adoption,
        "days_without_adoption": days_without_adoption,
    }


training_data = {
    "days_to_event": df["days_to_event"].values,
    "is_adopted": df["is_adopted"].values,
    "is_black": df["is_black"].values,
}
mcmc_11_15 = MCMC(NUTS(model_11_15), num_warmup=500, num_samples=500, num_chains=4)
mcmc_11_15.run(jrng, **training_data)
mcmc_11_15.print_summary(prob=0.89)
idata_11_15 = az.from_numpyro(mcmc_11_15)
az.plot_trace(idata_11_15, var_names=["a"], kind="rank_bars", compact=False)
plt.tight_layout()
```

### Code 11.67

```python
numpyro.diagnostics.print_summary(
    {
        "a": mcmc_11_15.get_samples()["a"],
        "D": jnp.exp(mcmc_11_15.get_samples()["a"]),
    },
    prob=0.89,
    group_by_chain=False,
)
```

### Code 11.68

```python
print(inspect.getsource(model_11_15))
```

### Code 11.69

```python
print("\n".join(inspect.getsource(model_11_15).splitlines()[13:20]))
```

## Easy
### 11E1

```python
jnp.log(0.35 / (1 - 0.35))
```

```python
jax.scipy.special.logit(0.35)
```

### 11E2

```python
jax.scipy.special.expit(3.2)
```

```python
jnp.exp(3.2) / (1 + jnp.exp(3.2))
```

### 11E3

```python
jnp.exp(1.7)
```

Each unit change in the predictor variable multiples the odds of the event by 5.5

<!-- #region -->
### 11E4

Poisson regression model the count of event, parametrized on a rate of event occurence. 
If the data contains count over multiple durations, we need can model using an offset.

## Medium
### 11M1

Likelihood is the probability of observing a particular training sample. When aggregating, the likelihood of seeing the sequence is much larger than the product of the likelihoods, b/c aggregated model needs to count all the different orders that could yield that particular sample.

### 11M2

Each unit change in the predictor variable increases the arrival rate by a factor of 5.5

### 11M3

- correct input and output spaces: from real line onto [0, 1]
- intuitive to have near linear response around the mean, and asymptotes to 0 and 1 as we go to +/- infinity

### 11M4

Takes real number and returns a positive one.

### 11M5

A logit link on a Poisson GLM implies an arrival rate between 0 and 1. Could be appropriate if you think a predictor saturates: at it gets very large, it no longer contributes to increases in the expected arrival rate.

### 11M6

Discrete binary outcomes and constant probability of each event across trials. Poisson is binomical when number of trials goes to infinity.


## Hard
### 11H1
<!-- #endregion -->

```python
df = pd.read_csv("../data/chimpanzees.csv", sep=";")
df["treatment"] = df["prosoc_left"] + 2 * df["condition"]
df
```

```python
def model_11h1(actor, treatment, pulled_left):
    with numpyro.plate(name="actor", size=7):
        a = numpyro.sample("a", dist.Normal(0, 1.5))
    with numpyro.plate(name="treatment", size=4):
        b = numpyro.sample("b", dist.Normal(0, 0.5))
    logit_p = a[actor] + b[treatment]
    numpyro.deterministic("p_left_handed", 1 / (1 + jnp.exp(-a)))
    numpyro.deterministic("p_treatment", 1 / (1 + jnp.exp(-b)))
    p = numpyro.deterministic("p", 1 / (1 + jnp.exp(-logit_p)))
    pulled_left = numpyro.sample("pulled_left", dist.Bernoulli(p), obs=pulled_left)
    return pulled_left


training_data = {
    "pulled_left": df["pulled_left"].values,
    "actor": df["actor"].values - 1,  # NB: 0-indexing
    "treatment": df["treatment"].values,
}
guide_11h1 = AutoLaplaceApproximation(model_11h1)
svi_11h1 = SVI(model=model_11h1, guide=guide_11h1, optim=optim, loss=loss).run(
    jrng, num_steps=2_500, **training_data
)
posterior_samples_11h1a = guide_11h1.sample_posterior(
    jrng, svi_11h1.params, sample_shape=(1_000,)
)
numpyro.diagnostics.print_summary(
    prune_return_sites(
        posterior_samples_11h1a, depth=2, exclude=["p", "p_left_handed"]
    ),
    prob=0.89,
    group_by_chain=False,
)
```

```python
mcmc_11h1 = MCMC(NUTS(model_11h1), num_warmup=500, num_samples=1_000, num_chains=4)
mcmc_11h1.run(jrng, **training_data)
mcmc_11h1.print_summary(prob=0.89)
```

Makes no difference, because we use pretty decent priors.


### 11H2

```python
def model_11_1(pulled_left):
    a = numpyro.sample("a", dist.Normal(0, 10))
    logit_p = a
    p = numpyro.deterministic("p", 1 / (1 + jnp.exp(-logit_p)))
    pulled_left = numpyro.sample("pulled_left", dist.Bernoulli(p), obs=pulled_left)
    return pulled_left


training_data = {"pulled_left": df["pulled_left"].values}
mcmc_11_1 = MCMC(NUTS(model_11_1), num_warmup=500, num_samples=1_000, num_chains=4)
mcmc_11_1.run(jrng, **training_data)
idata_11_1 = az.from_numpyro(mcmc_11_1)


def model_11_2(treatment, pulled_left):
    a = numpyro.sample("a", dist.Normal(0, 1.5))
    with numpyro.plate(name="treatment", size=4):
        b = numpyro.sample("b", dist.Normal(0, 10))
    logit_p = a + b[treatment]
    p = numpyro.deterministic("p", 1 / (1 + jnp.exp(-logit_p)))
    pulled_left = numpyro.sample("pulled_left", dist.Bernoulli(p), obs=pulled_left)
    return pulled_left


training_data = {
    "treatment": df["treatment"].values,
    "pulled_left": df["pulled_left"].values,
}
mcmc_11_2 = MCMC(NUTS(model_11_2), num_warmup=500, num_samples=1_000, num_chains=4)
mcmc_11_2.run(jrng, **training_data)
idata_11_2 = az.from_numpyro(mcmc_11_2)


def model_11_3(treatment, pulled_left):
    a = numpyro.sample("a", dist.Normal(0, 1.5))
    with numpyro.plate(name="treatment", size=4):
        b = numpyro.sample("b", dist.Normal(0, 0.5))
    logit_p = a + b[treatment]
    p = numpyro.deterministic("p", 1 / (1 + jnp.exp(-logit_p)))
    pulled_left = numpyro.sample("pulled_left", dist.Bernoulli(p), obs=pulled_left)
    return pulled_left


training_data = {
    "treatment": df["treatment"].values,
    "pulled_left": df["pulled_left"].values,
}
mcmc_11_3 = MCMC(NUTS(model_11_3), num_warmup=500, num_samples=1_000, num_chains=4)
mcmc_11_3.run(jrng, **training_data)
idata_11_3 = az.from_numpyro(mcmc_11_3)


def model_11_4(actor, treatment, pulled_left):
    with numpyro.plate(name="actor", size=7):
        a = numpyro.sample("a", dist.Normal(0, 1.5))
    with numpyro.plate(name="treatment", size=4):
        b = numpyro.sample("b", dist.Normal(0, 0.5))
    logit_p = a[actor] + b[treatment]
    numpyro.deterministic("p_left_handed", 1 / (1 + jnp.exp(-a)))
    numpyro.deterministic("p_treatment", 1 / (1 + jnp.exp(-b)))
    p = numpyro.deterministic("p", 1 / (1 + jnp.exp(-logit_p)))
    pulled_left = numpyro.sample("pulled_left", dist.Bernoulli(p), obs=pulled_left)
    return pulled_left


training_data = {
    "pulled_left": df["pulled_left"].values,
    "actor": df["actor"].values - 1,  # NB: 0-indexing
    "treatment": df["treatment"].values,
}
mcmc_11_4 = MCMC(NUTS(model_11_4), num_warmup=500, num_samples=1_000, num_chains=4)
mcmc_11_4.run(jrng, **training_data)
mcmc_11_4.print_summary(prob=0.89)
idata_11_4 = az.from_numpyro(mcmc_11_4)
```

```python
compare = az.compare(
    {"11.1": idata_11_1, "11.2": idata_11_2, "11.3": idata_11_3, "11.4": idata_11_4},
    ic="loo",
    scale="deviance",
)
display(compare)
az.plot_compare(compare)
```

### 11H3

```python
df = pd.read_csv("../data/eagles.csv", sep=",")
df
```

```python
df["P"] = (df["P"] == "L").astype(int)
df["A"] = (df["A"] == "A").astype(int)
df["V"] = (df["V"] == "L").astype(int)
df
```

```python
def model_11h3(P, A, V, n, y):
    a = numpyro.sample("a", dist.Normal(0, 10))
    bP = numpyro.sample("bP", dist.Normal(0, 5))
    bA = numpyro.sample("bA", dist.Normal(0, 5))
    bV = numpyro.sample("bV", dist.Normal(0, 5))
    logit_p = a + bP * P + bA * A + bV * V
    p = numpyro.deterministic("p", jax.nn.sigmoid(logit_p))
    y = numpyro.sample("y", dist.Binomial(total_count=n, probs=p), obs=y)
    return y


training_data_11h3 = {
    "P": df["P"].values,
    "A": df["A"].values,
    "V": df["V"].values,
    "n": df["n"].values,
    "y": df["y"].values,
}
mcmc_11h3 = MCMC(NUTS(model_11h3), num_warmup=500, num_samples=1_000, num_chains=4)
mcmc_11h3.run(jrng, **training_data_11h3)
mcmc_11h3.print_summary(prob=0.89)
idata_11h3 = az.from_numpyro(mcmc_11h3)
az.plot_trace(
    idata_11h3, var_names=["a", "bP", "bA", "bV"], kind="rank_bars", compact=False
)
plt.tight_layout()
```

```python
posterior_predictive_11h3 = Predictive(model_11h3, mcmc_11h3.get_samples())

posterior_predictive_samples_11h3 = posterior_predictive_11h3(
    jrng, P=df["P"].values, A=df["A"].values, V=df["V"].values, n=df["n"].values, y=None
)
expected_probability_11h3 = posterior_predictive_samples_11h3["p"].mean(axis=0)
hpdi_probability_11h3 = numpyro.diagnostics.hpdi(
    posterior_predictive_samples_11h3["p"], 0.89
)
plt.errorbar(
    x=df["y"] / df["n"],
    y=expected_probability_11h3,
    yerr=jnp.abs(hpdi_probability_11h3 - expected_probability_11h3),
    fmt="ko",
    alpha=0.5,
)
plt.gca().set(xlabel="observed proportion", ylabel="predicted probability")
```

```python
expected_n_success_11h3 = posterior_predictive_samples_11h3["y"].mean(axis=0)
hpdi_n_success_11h3 = numpyro.diagnostics.hpdi(
    posterior_predictive_samples_11h3["y"], 0.89
)
plt.errorbar(
    x=df["y"],
    y=expected_n_success_11h3,
    yerr=jnp.abs(hpdi_n_success_11h3 - expected_n_success_11h3),
    fmt="ko",
    alpha=0.5,
)
plt.gca().set(xlabel="observed count", ylabel="predicted count")
```

Proportions are better since they normalize on number of attempts.
Count do hav ethe advantage of showing uncertainty due to the larger number of attempts.

```python
def model_11h3b(P, A, V, n, y):
    a = numpyro.sample("a", dist.Normal(0, 10))
    bP = numpyro.sample("bP", dist.Normal(0, 5))
    bA = numpyro.sample("bA", dist.Normal(0, 5))
    bV = numpyro.sample("bV", dist.Normal(0, 5))
    bPA = numpyro.sample("bPA", dist.Normal(0, 5))
    logit_p = a + bP * P + bA * A + bV * V + bPA * P * A
    p = numpyro.deterministic("p", jax.nn.sigmoid(logit_p))
    y = numpyro.sample("y", dist.Binomial(total_count=n, probs=p), obs=y)
    return y


training_data_11h3b = {
    "P": df["P"].values,
    "A": df["A"].values,
    "V": df["V"].values,
    "n": df["n"].values,
    "y": df["y"].values,
}
mcmc_11h3b = MCMC(NUTS(model_11h3b), num_warmup=500, num_samples=1_000, num_chains=4)
mcmc_11h3b.run(jrng, **training_data_11h3b)
mcmc_11h3b.print_summary(prob=0.89)
idata_11h3b = az.from_numpyro(mcmc_11h3b)
az.plot_trace(
    idata_11h3b,
    var_names=["a", "bP", "bA", "bV", "bPA"],
    kind="rank_bars",
    compact=False,
)
plt.tight_layout()
```

```python
compare_11h3b = az.compare(
    {"11h3": idata_11h3, "11h3b": idata_11h3b}, ic="loo", scale="deviance"
)
display(compare_11h3b)
az.plot_compare(compare_11h3b)
```

Interaction term improves predictions.


### 11H4

```python
df = pd.read_csv("../data/salamanders.csv", sep=";")
df["pct_cover"] = df["PCTCOVER"] / 100
df["forest_age"] = scale(df["FORESTAGE"])
df.head()
```

```python
def model_11h4(pct_cover, salamanders):
    a = numpyro.sample("a", dist.Normal(0, 1))
    b = numpyro.sample("b", dist.Normal(0, 0.5))
    lambda_ = numpyro.deterministic("lambda", jnp.exp(a + b * pct_cover))
    salamanders = numpyro.sample(
        "salamanders", dist.Poisson(rate=lambda_), obs=salamanders
    )
    return salamanders


pct_covers = jnp.linspace(0, 1, 100)
prior_predictive_samples_11h4 = Predictive(model_11h4, num_samples=1000)(
    jrng, pct_cover=pct_covers, salamanders=None
)
plt.scatter(df["pct_cover"], df["SALAMAN"], color="k", alpha=0.5)
prior_expected_count_11h4 = prior_predictive_samples_11h4["salamanders"].mean(axis=0)
prior_hpdi_11h4 = numpyro.diagnostics.hpdi(
    prior_predictive_samples_11h4["salamanders"], 0.89
)
plt.plot(pct_covers, prior_expected_count_11h4)
plt.fill_between(
    pct_covers, prior_hpdi_11h4[0], prior_hpdi_11h4[1], color="k", alpha=0.3
)
plt.gca().set(title="Prior Predictive Check", xlabel="pct cover", ylabel="salamanders")
```

```python
training_data = {
    "pct_cover": df["pct_cover"].values,
    "salamanders": df["SALAMAN"].values,
}
mcmc_11h4 = MCMC(NUTS(model_11h4), num_warmup=500, num_samples=2_000, num_chains=4)
mcmc_11h4.run(jrng, **training_data)
mcmc_11h4.print_summary(prob=0.89)
idata_11h4 = az.from_numpyro(mcmc_11h4)
az.plot_trace(idata_11h4, compact=False, var_names=["a", "b"], kind="rank_bars")
plt.tight_layout()
```

```python
pct_covers = jnp.linspace(0, 1, 100)
posterior_predictive_samples_11h4 = Predictive(
    model_11h4, posterior_samples=mcmc_11h4.get_samples()
)(jrng, pct_cover=pct_covers, salamanders=None)
plt.scatter(df["pct_cover"], df["SALAMAN"], color="k", alpha=0.5)
posterior_expected_count_11h4 = posterior_predictive_samples_11h4["salamanders"].mean(
    axis=0
)
posterior_hpdi_11h4 = numpyro.diagnostics.hpdi(
    posterior_predictive_samples_11h4["salamanders"], 0.89
)
plt.plot(pct_covers, posterior_expected_count_11h4)
plt.fill_between(
    pct_covers,
    posterior_hpdi_11h4[0],
    posterior_hpdi_11h4[1],
    color="k",
    alpha=0.3,
)
plt.gca().set(
    xlim=(0, 1),
    ylim=(0, 20),
    title="Prior Predictive Check",
    xlabel="pct cover",
    ylabel="salamanders",
)
```

Model does pretty well up until about 90th percentile, where it fails to explain the variation in observations.


```python
def model_11h4b(pct_cover, forest_age, salamanders):
    a = numpyro.sample("a", dist.Normal(0, 1))
    b_cover = numpyro.sample("b_cover", dist.Normal(0, 1))
    b_age = numpyro.sample("b_age", dist.Normal(0, 1))
    lambda_ = numpyro.deterministic(
        "lambda", jnp.exp(a + b_cover * pct_cover + b_age * forest_age)
    )
    salamanders = numpyro.sample(
        "salamanders", dist.Poisson(rate=lambda_), obs=salamanders
    )
    return salamanders


forest_ages = jnp.linspace(-3, 3, 100)
prior_predictive_samples_11h4b = Predictive(model_11h4b, num_samples=1000)(
    jrng,
    pct_cover=0.90,
    forest_age=forest_ages,
    salamanders=None,
)
_df = df.loc[(df["pct_cover"] >= 0.85) & (df["pct_cover"] <= 0.95), :]
plt.scatter(
    x=_df["forest_age"],
    y=_df["SALAMAN"],
    color="k",
    alpha=0.5,
)
prior_expected_count_11h4b = prior_predictive_samples_11h4b["salamanders"].mean(axis=0)
prior_hpdi_11h4b = numpyro.diagnostics.hpdi(
    prior_predictive_samples_11h4b["salamanders"], 0.89
)
plt.plot(forest_ages, prior_expected_count_11h4b)
plt.fill_between(
    forest_ages, prior_hpdi_11h4b[0], prior_hpdi_11h4b[1], color="k", alpha=0.3
)
plt.gca().set(
    title="Prior Predictive Check (cover ~ 90%)",
    xlabel="Forest Age",
    ylabel="# Salamanders",
)
```

```python
training_data = {
    "pct_cover": df["pct_cover"].values,
    "forest_age": df["forest_age"].values,
    "salamanders": df["SALAMAN"].values,
}
mcmc_11h4b = MCMC(NUTS(model_11h4b), num_warmup=500, num_samples=1_000, num_chains=4)
mcmc_11h4b.run(jrng, **training_data)
mcmc_11h4b.print_summary(prob=0.89)
idata_11h4b = az.from_numpyro(mcmc_11h4b)
az.plot_trace(
    idata_11h4b, compact=False, var_names=["a", "b_cover", "b_age"], kind="rank_bars"
)
plt.tight_layout()
```

```python
compare_11h4b = az.compare(
    {"11h4a": idata_11h4, "11h4b": idata_11h4b}, ic="loo", scale="deviance"
)
display(compare_11h4b)
az.plot_compare(compare_11h4b)
```

Forest age helps prediction, but not much, because it's correlated with cover percentage.

```python

```
