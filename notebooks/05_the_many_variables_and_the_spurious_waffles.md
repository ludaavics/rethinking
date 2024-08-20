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

# Chapter 5: The Many Variables And The Spurious Waffles 

```python
%load_ext jupyter_black
```

```python
import collections
import itertools

import arviz as az
import daft
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import networkx as nx
import numpyro
import numpyro.distributions as dist
from numpyro.infer.autoguide import AutoLaplaceApproximation
import pandas as pd
from sklearn.preprocessing import LabelEncoder

seed = 84735
jrng = jax.random.key(seed)
plt.rcParams["figure.figsize"] = [10, 6]

optim = numpyro.optim.Adam(step_size=1)
loss = numpyro.infer.Trace_ELBO()
```

## Code
### Code 5.1

```python
df = pd.read_csv("../data/WaffleDivorce.csv", sep=";")
df["A"] = (df["MedianAgeMarriage"] - df["MedianAgeMarriage"].mean()) / df[
    "MedianAgeMarriage"
].std()
df["D"] = (df["Divorce"] - df["Divorce"].mean()) / df["Divorce"].std()
df.head()
```

### Code 5.2

```python
df["MedianAgeMarriage"].std()
```

### Code 5.3

```python
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
```

### Code 5.4

```python
plt.subplot(
    xlabel="Median Marriage Age (std)",
    ylabel="Divorce Rate (std)",
)
age = jnp.linspace(-2, 2, 50)
prior_predictive_samples_5_1 = numpyro.infer.Predictive(model_5_1, num_samples=100)(
    jrng, age=age
)
for sample in prior_predictive_samples_5_1["mu"]:
    plt.plot(
        age,
        sample,
        "k",
        alpha=0.2,
    )
```

### Code 5.5

```python
guide_5_1 = AutoLaplaceApproximation(model_5_1)
svi_5_1 = numpyro.infer.SVI(
    model=model_5_1,
    guide=guide_5_1,
    optim=numpyro.optim.Adam(step_size=0.5),
    loss=numpyro.infer.Trace_ELBO(),
    age=df["A"].values,
    divorce=df["D"].values,
).run(jrng, 5_000)
```

```python
posterior_samples_5_1 = guide_5_1.sample_posterior(
    jrng, svi_5_1.params, sample_shape=(10_000,)
)
_posterior_samples_5_1 = {k: v for k, v in posterior_samples_5_1.items() if k != "mu"}
numpyro.diagnostics.print_summary(
    _posterior_samples_5_1, prob=0.89, group_by_chain=False
)
```

```python
age = jnp.linspace(-3, 3.2, 50)

posterior_predictive_5_1 = numpyro.infer.Predictive(
    guide_5_1.model, posterior_samples_5_1, return_sites=["mu"]
)
posterior_predictive_samples_5_1 = posterior_predictive_5_1(
    jrng,
    age=age,
    divorce=None,
)

mu_posterior_predictive_5_1 = pd.DataFrame(
    posterior_predictive_samples_5_1["mu"], columns=age
)
mu_mean_5_1 = mu_posterior_predictive_5_1.mean(axis=0).to_frame().T
mu_hpdi_5_1 = pd.DataFrame(
    numpyro.diagnostics.hpdi(mu_posterior_predictive_5_1, prob=0.89),
    columns=age,
)
df.plot(kind="scatter", x="A", y="D")
plt.plot(age, mu_mean_5_1.loc[0, :], "k")
plt.fill_between(
    age, mu_hpdi_5_1.iloc[0, :], mu_hpdi_5_1.iloc[1, :], color="k", alpha=0.5
)
```

### Code 5.6

```python
df["M"] = (df["Marriage"] - df["Marriage"].mean()) / df["Marriage"].std()


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
    optim=numpyro.optim.Adam(step_size=0.5),
    loss=numpyro.infer.Trace_ELBO(),
    marriage=df["M"].values,
    divorce=df["D"].values,
).run(jrng, 5_000)
```

```python
posterior_samples_5_2 = guide_5_2.sample_posterior(
    jrng, svi_5_2.params, sample_shape=(10_000,)
)
_posterior_samples_5_2 = {k: v for k, v in posterior_samples_5_2.items() if k != "mu"}
numpyro.diagnostics.print_summary(
    _posterior_samples_5_2, prob=0.89, group_by_chain=False
)
```

```python
marriage = jnp.linspace(-3, 3.2, 50)

posterior_predictive_5_2 = numpyro.infer.Predictive(
    guide_5_2.model, posterior_samples_5_2, return_sites=["mu"]
)
posterior_predictive_samples_5_2 = posterior_predictive_5_2(
    jrng,
    marriage=marriage,
    divorce=None,
)

mu_posterior_predictive_5_2 = pd.DataFrame(
    posterior_predictive_samples_5_2["mu"], columns=age
)
mu_mean_5_2 = mu_posterior_predictive_5_2.mean(axis=0).to_frame().T
mu_hpdi_5_2 = pd.DataFrame(
    numpyro.diagnostics.hpdi(mu_posterior_predictive_5_2, prob=0.89),
    columns=age,
)
df.plot(kind="scatter", x="M", y="D")
plt.plot(age, mu_mean_5_2.loc[0, :], "k")
plt.fill_between(
    age, mu_hpdi_5_2.iloc[0, :], mu_hpdi_5_2.iloc[1, :], color="k", alpha=0.5
)
```

### Code 5.7

```python
dag_5_1 = nx.DiGraph()
dag_5_1.add_edges_from([("A", "D"), ("A", "M"), ("M", "D")])
pgm = daft.PGM()
coordinates = {"A": (0, 0), "D": (1, -1), "M": (2, 0)}
for node in dag_5_1.nodes:
    pgm.add_node(node, node, *coordinates[node])
for edge in dag_5_1.edges:
    pgm.add_edge(*edge)
pgm.render()
```

### Code 5.8

```python
DMA_dag2 = nx.DiGraph()
DMA_dag2.add_edges_from([("A", "D"), ("A", "M")])
pgm = daft.PGM()
coordinates = {"A": (0, 0), "D": (1, -1), "M": (2, 0)}
for node in DMA_dag2.nodes:
    pgm.add_node(node, node, *coordinates[node])
for edge in DMA_dag2.edges:
    pgm.add_edge(*edge)
pgm.render()
conditional_independencies = collections.defaultdict(list)
for edge in itertools.combinations(sorted(DMA_dag2.nodes), 2):
    remaining = sorted(set(DMA_dag2.nodes) - set(edge))
    for size in range(len(remaining) + 1):
        for subset in itertools.combinations(remaining, size):
            if any(
                cond.issubset(set(subset)) for cond in conditional_independencies[edge]
            ):
                continue
            if nx.is_d_separator(DMA_dag2, {edge[0]}, {edge[1]}, set(subset)):
                conditional_independencies[edge].append(set(subset))
                print(
                    f"{edge[0]} _||_ {edge[1]}"
                    + (f" | {' '.join(subset)}" if subset else "")
                )
```

### Code 5.9

```python
DMA_dag1 = nx.DiGraph()
DMA_dag1.add_edges_from([("A", "D"), ("A", "M"), ("M", "D")])
pgm = daft.PGM()
coordinates = {"A": (0, 0), "D": (1, -1), "M": (2, 0)}
for node in DMA_dag1.nodes:
    pgm.add_node(node, node, *coordinates[node])
for edge in DMA_dag1.edges:
    pgm.add_edge(*edge)
pgm.render()
conditional_independencies = collections.defaultdict(list)
for edge in itertools.combinations(sorted(DMA_dag1.nodes), 2):
    remaining = sorted(set(DMA_dag1.nodes) - set(edge))
    for size in range(len(remaining) + 1):
        for subset in itertools.combinations(remaining, size):
            if any(
                cond.issubset(set(subset)) for cond in conditional_independencies[edge]
            ):
                continue
            if nx.is_d_separator(DMA_dag1, {edge[0]}, {edge[1]}, set(subset)):
                conditional_independencies[edge].append(set(subset))
                print(
                    f"{edge[0]} _||_ {edge[1]}"
                    + (f" | {' '.join(subset)}" if subset else "")
                )
```

### Code 5.10

```python
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
).run(jrng, 5_000)

posterior_samples_5_3 = guide_5_3.sample_posterior(
    jrng, svi_5_3.params, sample_shape=(10_000,)
)
_posterior_samples_5_3 = {
    k: v for k, v in posterior_samples_5_3.items() if k not in ["mu"]
}
numpyro.diagnostics.print_summary(
    _posterior_samples_5_3, prob=0.89, group_by_chain=False
)
```

### Code 5.11

```python
_, jrng = jax.random.split(jrng)
coeftab = {
    "m5.1": guide_5_1.sample_posterior(jrng, svi_5_1.params, sample_shape=(1, 1_000)),
    "m5.2": guide_5_2.sample_posterior(jrng, svi_5_2.params, sample_shape=(1, 1_000)),
    "m5.3": guide_5_3.sample_posterior(jrng, svi_5_3.params, sample_shape=(1, 1_000)),
}
az.plot_forest(
    list(coeftab.values()),
    model_names=list(coeftab.keys()),
    var_names=["beta_age", "beta_marriage"],
    hdi_prob=0.89,
)
plt.show()
```

### Code 5.12

```python
N = 50
age = dist.Normal().sample(jrng, sample_shape=(N,))
_, jrng = jax.random.split(jrng)
marriage = dist.Normal(loc=-age).sample(jrng, sample_shape=(1,))
_, jrng = jax.random.split(jrng)
divorce = dist.Normal(loc=age).sample(jrng)
```

### Code 5.13

```python
def model_5_4(
    age,
    *,
    alpha_prior={"loc": 0, "scale": 0.2},
    beta_age_prior={"loc": 0, "scale": 0.5},
    sigma_prior={"rate": 1},
    marriage=None,
):
    alpha = numpyro.sample("alpha", dist.Normal(**alpha_prior))
    beta_age = numpyro.sample("beta_age", dist.Normal(**beta_age_prior))
    mu = numpyro.deterministic("mu", alpha + beta_age * age)
    sigma = numpyro.sample("sigma", dist.Exponential(**sigma_prior))
    marriage = numpyro.sample("marriage", dist.Normal(mu, sigma), obs=marriage)
    return marriage


_, jrng = jax.random.split(jrng)
guide_5_4 = AutoLaplaceApproximation(model_5_4)
svi_5_4 = numpyro.infer.SVI(
    model=model_5_4,
    guide=guide_5_4,
    optim=numpyro.optim.Adam(step_size=1),
    loss=numpyro.infer.Trace_ELBO(),
    age=df["A"].values,
    marriage=df["M"].values,
).run(jrng, num_steps=5_000)
```

### Code 5.14

```python
posterior_samples_5_4 = guide_5_4.sample_posterior(
    jrng, svi_5_4.params, sample_shape=(10_000,)
)
_posterior_samples_5_4 = {
    k: v for (k, v) in posterior_samples_5_4.items() if k not in ["mu"]
}
numpyro.diagnostics.print_summary(
    _posterior_samples_5_4, prob=0.89, group_by_chain=False
)

mu_mean_5_4 = posterior_samples_5_4["mu"].mean(axis=0)
mu_residual_5_4 = (df["M"] - mu_mean_5_4).rename("M_residual").to_frame().T
mu_residual_5_4
```

```python
ax = plt.subplot(
    xlabel="Age at Marriage (std)",
    ylabel="Marriage Rate (std)",
)
plt.plot(df["A"], df["M"], "o")
plt.plot(df["A"], mu_mean_5_4, "k", alpha=0.5)
```

### Code 5.15

```python
mu_mean_5_3 = pd.DataFrame(posterior_samples_5_3["mu"].mean(axis=0)).T
mu_hpdi_5_3 = pd.DataFrame(
    numpyro.diagnostics.hpdi(posterior_samples_5_3["mu"], prob=0.89)
)
posterior_predictive_5_3 = numpyro.infer.Predictive(
    guide_5_3.model, posterior_samples_5_3, return_sites=["divorce"]
)
posterior_predictive_samples_5_3 = posterior_predictive_5_3(
    jrng,
    age=df["A"].values,
    marriage=df["M"].values,
    divorce=None,
)

D_hpdi = numpyro.diagnostics.hpdi(
    posterior_predictive_samples_5_3["divorce"], prob=0.89
)
```

### Code 5.16

```python
ax = plt.subplot(
    xlabel="Observed Divorce",
    ylabel="Predicted Divorce",
)
plt.plot(df["D"], mu_mean_5_3.iloc[0, :], "o")
x = jnp.linspace(mu_hpdi_5_3.min().min(), mu_hpdi_5_3.max().max(), 101)
plt.plot(x, x, "--")
for i in range(df.shape[0]):
    plt.plot([df["D"][i]] * 2, mu_hpdi_5_3.iloc[:, i], "b")
fig = plt.gcf()
```

### Code 5.17

```python
for i in range(df.shape[0]):
    if df["Loc"][i] in ["ID", "UT", "RI", "ME"]:
        ax.annotate(
            df["Loc"][i],
            (df["D"][i], mu_mean_5_3[i]),
            xytext=(-25, -5),
            textcoords="offset pixels",
        )
fig
```

### Code 5.18

```python
N = 100
x_real = dist.Normal(0, 1).sample(jrng, sample_shape=(N,))
_, jrng = jax.random.split(jrng)
x_spur = dist.Normal(x_real, 1).sample(jrng)
y = dist.Normal(x_real, 1).sample(jrng)
df = pd.DataFrame([x_real, x_spur, y], index=["x_real", "x_spur", "y"]).T
df.head()
```

### Code 5.19

```python
df = pd.read_csv("../data/WaffleDivorce.csv", sep=";")
df["A"] = (df["MedianAgeMarriage"] - df["MedianAgeMarriage"].mean()) / df[
    "MedianAgeMarriage"
].std()
df["M"] = (df["Marriage"] - df["Marriage"].mean()) / df["Marriage"].std()
df["D"] = (df["Divorce"] - df["Divorce"].mean()) / df["Divorce"].std()


def model_5_3a(
    age,
    marriage,
    *,
    alpha_marriage_prior={"loc": 0, "scale": 0.2},
    beta_marriage_age_prior={"loc": 0, "scale": 0.5},
    sigma_marriage_prior={"rate": 1},
    alpha_divorce_prior={"loc": 0, "scale": 0.2},
    beta_divorce_age_prior={"loc": 0, "scale": 0.5},
    beta_divorce_marriage_prior={"loc": 0, "scale": 0.5},
    sigma_divorce_prior={"rate": 1},
    divorce=None,
):
    # A -> M
    alpha_marriage = numpyro.sample(
        "alpha_marriage",
        dist.Normal(**alpha_marriage_prior),
    )
    beta_marriage_age = numpyro.sample(
        "beta_marriage_age",
        dist.Normal(**beta_marriage_age_prior),
    )
    mu_marriage = numpyro.deterministic(
        "mu_marriage",
        alpha_marriage + beta_marriage_age * age,
    )
    sigma_marriage = numpyro.sample(
        "sigma_marriage",
        dist.Exponential(**sigma_marriage_prior),
    )
    marriage = numpyro.sample(
        "marriage",
        dist.Normal(mu_marriage, sigma_marriage),
        obs=marriage,
    )

    # A -> D <- M
    alpha_divorce = numpyro.sample(
        "alpha_divorce",
        dist.Normal(**alpha_divorce_prior),
    )
    beta_divorce_age = numpyro.sample(
        "beta_divorce_age",
        dist.Normal(**beta_divorce_age_prior),
    )
    beta_divorce_marriage = numpyro.sample(
        "beta_divorce_marriage",
        dist.Normal(**beta_divorce_marriage_prior),
    )
    mu_divorce = numpyro.deterministic(
        "mu_divorce",
        alpha_divorce + beta_divorce_age * age + beta_divorce_marriage * marriage,
    )
    sigma_divorce = numpyro.sample(
        "sigma_divorce",
        dist.Exponential(**sigma_divorce_prior),
    )
    divorce = numpyro.sample(
        "divorce",
        dist.Normal(mu_divorce, sigma_divorce),
        obs=divorce,
    )

    return divorce


guide_5_3a = AutoLaplaceApproximation(model_5_3a)
svi_5_3a = numpyro.infer.SVI(
    model=model_5_3a,
    guide=guide_5_3a,
    optim=numpyro.optim.Adam(step_size=1),
    loss=numpyro.infer.Trace_ELBO(),
    age=df["A"].values,
    marriage=df["M"].values,
    divorce=df["D"].values,
).run(jrng, 5_000)
```

```python
posterior_samples_5_3a = guide_5_3a.sample_posterior(
    jrng, svi_5_3a.params, sample_shape=(10_000,)
)
_posterior_samples_5_3a = {
    k: v
    for (k, v) in posterior_samples_5_3a.items()
    if k not in ["mu_marriage", "mu_divorce"]
}
numpyro.diagnostics.print_summary(
    _posterior_samples_5_3a, prob=0.89, group_by_chain=False
)
```

### Code 5.20

```python
ages = jnp.linspace(-2, 2, 30)
```

### Codes 5.21

```python
posterior_predictive_5_3a = numpyro.infer.Predictive(
    model=model_5_3a,
    posterior_samples=posterior_samples_5_3a,
    return_sites=["marriage", "divorce"],
)
divorce_age_counterfactual_5_3a = posterior_predictive_5_3a(
    jrng,
    age=ages,
    marriage=None,
    divorce=None,
)

marriage_mean_5_3a = (
    pd.DataFrame(divorce_age_counterfactual_5_3a["marriage"]).mean().to_frame().T
)
marriage_hpdi_5_3a = pd.DataFrame(
    numpyro.diagnostics.hpdi(divorce_age_counterfactual_5_3a["marriage"], prob=0.89)
)
divorce_mean_5_3a = (
    pd.DataFrame(divorce_age_counterfactual_5_3a["divorce"]).mean().to_frame().T
)
divorce_hpdi_5_3a = pd.DataFrame(
    numpyro.diagnostics.hpdi(divorce_age_counterfactual_5_3a["divorce"], prob=0.89)
)
```

### Code 5.22

```python
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(ages, divorce_mean_5_3a.iloc[0, :])
ax1.fill_between(
    ages,
    divorce_hpdi_5_3a.iloc[0, :],
    divorce_hpdi_5_3a.iloc[1, :],
    color="k",
    alpha=0.2,
)
ax1.set(xlabel="Manipulated Age (std)", ylabel="Counterfactual Divorce Rate (std)")

ax2.plot(ages, marriage_mean_5_3a.iloc[0, :])
ax2.fill_between(
    ages,
    marriage_hpdi_5_3a.iloc[0, :],
    marriage_hpdi_5_3a.iloc[1, :],
    color="k",
    alpha=0.2,
)
ax2.set(
    xlabel="Manipulated Age at Marriage (std)",
    ylabel="Counteractual Impact on Marriage Rate (std)",
)
```

### Code 5.23

```python
marriages = jnp.linspace(-2, 2, 30)
_, jrng = jax.random.split(jrng)
divorce_marriage_counterfactual_samples_5_3a = posterior_predictive_5_3a(
    jrng,
    age=0,
    marriage=marriages,
    divorce=None,
)

divorce_marriage_counterfactual_mean_5_3a = (
    pd.DataFrame(divorce_marriage_counterfactual_samples_5_3a["divorce"])
    .mean()
    .to_frame()
    .T
)
divorce_marriage_counterfactual_hpdi_5_3a = pd.DataFrame(
    numpyro.diagnostics.hpdi(
        divorce_marriage_counterfactual_samples_5_3a["divorce"], prob=0.89
    )
)

plt.plot(marriages, divorce_marriage_counterfactual_mean_5_3a.iloc[0, :])
plt.fill_between(
    marriages,
    divorce_marriage_counterfactual_hpdi_5_3a.iloc[0, :],
    divorce_marriage_counterfactual_hpdi_5_3a.iloc[1, :],
    color="k",
    alpha=0.2,
)
plt.xlabel("Manipulated Marriage Rate (std)")
plt.ylabel("Counterfactual Divorce Rate (std)")
plt.title("Total counterfactual impact of Marriage Rate on Divorce Rate")
```

### Code 5.24

```python
ages = jnp.linspace(-2, 2, 30)
```

### Code 5.25

```python
posterior_samples_5_3a = guide_5_3a.sample_posterior(
    jrng, svi_5_3a.params, sample_shape=(10_000,)
)
posterior_samples_5_3a = {k: v[..., None] for (k, v) in posterior_samples_5_3a.items()}
marriage_counterfactual_5_3a = dist.Normal(
    posterior_samples_5_3a["alpha_marriage"]
    + posterior_samples_5_3a["beta_marriage_age"] * ages,
    posterior_samples_5_3a["sigma_marriage"],
).sample(jrng)

marriage_mean_5_3a = marriage_counterfactual_5_3a.mean(axis=0)
marriage_hpdi_5_3a = pd.DataFrame(
    numpyro.diagnostics.hpdi(marriage_counterfactual_5_3a, prob=0.89)
)
```

### Code 5.26

```python
divorce_counterfactual_5_3a = dist.Normal(
    posterior_samples_5_3a["alpha_divorce"]
    + posterior_samples_5_3a["beta_divorce_age"] * ages
    + posterior_samples_5_3a["beta_divorce_marriage"] * marriage_counterfactual_5_3a,
    posterior_samples_5_3a["sigma_divorce"],
).sample(jrng)
divorce_mean_5_3a = divorce_counterfactual_5_3a.mean(axis=0)
divorce_hpdi_5_3a = pd.DataFrame(
    numpyro.diagnostics.hpdi(divorce_counterfactual_5_3a, prob=0.89)
)
```

```python
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(ages, divorce_mean_5_3a)
ax1.fill_between(
    ages,
    divorce_hpdi_5_3a.iloc[0, :],
    divorce_hpdi_5_3a.iloc[1, :],
    color="k",
    alpha=0.2,
)
ax1.set(xlabel="Manipulated Age (std)", ylabel="Counterfactual Divorce Rate (std)")

ax2.plot(ages, marriage_mean_5_3a)
ax2.fill_between(
    ages,
    marriage_hpdi_5_3a.iloc[0, :],
    marriage_hpdi_5_3a.iloc[1, :],
    color="k",
    alpha=0.2,
)
ax2.set(
    xlabel="Manipulated Age at Marriage (std)",
    ylabel="Counteractual Impact on Marriage Rate (std)",
)
```

### Code 5.27

```python
df = pd.read_csv("../data/milk.csv", sep=";")
df.head()
```

### Code 5.28

```python
def scale(x):
    return (x - x.mean()) / x.std()


df["K"] = scale(df["kcal.per.g"])
df["N"] = scale(df["neocortex.perc"])
df["M"] = scale(jnp.log(df["mass"].values))
```

### Code 5.29

```python
def model_5_5_draft(
    N,
    *,
    alpha_prior={"loc": 0, "scale": 1},
    beta_N_prior={"loc": 0, "scale": 1},
    sigma_prior={"rate": 1},
    K=None,
):
    alpha = numpyro.sample("alpha", dist.Normal(**alpha_prior))
    beta_N = numpyro.sample("beta_N", dist.Normal(**beta_N_prior))
    mu = numpyro.deterministic("mu", alpha + beta_N * N)
    sigma = numpyro.sample("sigma", dist.Exponential(**sigma_prior))
    K = numpyro.sample("K", dist.Normal(mu, sigma), obs=K)
    return K


try:
    guide_5_5_draft = AutoLaplaceApproximation(model_5_5_draft)
    svi_5_5_draft = numpyro.infer.SVI(
        model=model_5_5_draft,
        guide=guide_5_5_draft,
        optim=numpyro.optim.Adam(step_size=1),
        loss=numpyro.infer.Trace_ELBO(),
        N=df["N"].values,
        K=df["K"].values,
    ).run(jrng, 5_000)
except ValueError as e:
    print(e)
```

### Code 5.30

```python
df["neocortex.perc"]
```

### 5.31

```python
dcc = df.dropna()
dcc
```

### Code 5.32

```python
svi_5_5_draft = numpyro.infer.SVI(
    model=model_5_5_draft,
    guide=guide_5_5_draft,
    optim=numpyro.optim.Adam(step_size=1),
    loss=numpyro.infer.Trace_ELBO(),
    N=dcc["N"].values,
    K=dcc["K"].values,
).run(jrng, 5_000)
```

### Code 5.33

```python
N = jnp.linspace(-2, 2, 30)
prior_samples_5_5_draft = numpyro.infer.Predictive(
    model=model_5_5_draft, num_samples=50
)(jrng, N=N)
plt.subplot(
    ylim=(-2, 2),
    xlabel="Neocortext Percentage (std)",
    ylabel="Kcal/g of Milk (std)",
    title="Priors:\nalpha ~ N(0,1)\nbeta ~ N(0,1)",
)
for mu in prior_samples_5_5_draft["mu"]:
    plt.plot(N, mu, color="k", alpha=0.2)
```

### Code 5.34

```python
def model_5_5(
    N,
    *,
    alpha_prior={"loc": 0, "scale": 0.2},
    beta_N_prior={"loc": 0, "scale": 0.5},
    sigma_prior={"rate": 1},
    K=None,
):
    alpha = numpyro.sample("alpha", dist.Normal(**alpha_prior))
    beta_N = numpyro.sample("beta_N", dist.Normal(**beta_N_prior))
    mu = numpyro.deterministic("mu", alpha + beta_N * N)
    sigma = numpyro.sample("sigma", dist.Exponential(**sigma_prior))
    K = numpyro.sample("K", dist.Normal(mu, sigma), obs=K)
    return K


guide_5_5 = AutoLaplaceApproximation(model_5_5)
svi_5_5 = numpyro.infer.SVI(
    model=model_5_5,
    guide=guide_5_5,
    optim=numpyro.optim.Adam(step_size=1),
    loss=numpyro.infer.Trace_ELBO(),
    N=dcc["N"].values,
    K=dcc["K"].values,
).run(jrng, 5_000)

prior_samples_5_5 = numpyro.infer.Predictive(model=model_5_5, num_samples=50)(jrng, N=N)
plt.subplot(
    ylim=(-2, 2),
    xlabel="Neocortext Percentage (std)",
    ylabel="Kcal/g of Milk (std)",
    title="Priors:\nalpha ~ N(0,0.2)\nbeta ~ N(0,0.5)",
)
for mu in prior_samples_5_5["mu"]:
    plt.plot(N, mu, color="k", alpha=0.2)
```

### Code 5.35

```python
posterior_samples_5_5 = guide_5_5.sample_posterior(
    jrng, svi_5_5.params, sample_shape=(1_000,)
)
_posterior_samples_5_5 = {
    k: v for (k, v) in posterior_samples_5_5.items() if k not in ["mu"]
}
numpyro.diagnostics.print_summary(_posterior_samples_5_5, group_by_chain=False)
```

### Code 5.36

```python
N = jnp.linspace(dcc["N"].min() - 0.15, dcc["N"].max() + 0.15, 30)
posterior_predictive_5_5 = numpyro.infer.Predictive(
    model=model_5_5, posterior_samples=posterior_samples_5_5, num_samples=1_000
)
posterior_predictive_samples_5_5 = posterior_predictive_5_5(jrng, N=N)

mu_mean_5_5 = posterior_predictive_samples_5_5["mu"].mean(axis=0)
mu_hpdi_5_5 = numpyro.diagnostics.hpdi(
    posterior_predictive_samples_5_5["mu"], prob=0.89
)
plt.plot(dcc["N"], dcc["K"], "o")
plt.plot(N, mu_mean_5_5, color="k")
plt.fill_between(N, mu_hpdi_5_5[0], mu_hpdi_5_5[1], color="k", alpha=0.2)
```

### Code 5.37

```python
def model_5_6(
    M,
    *,
    alpha_prior={"loc": 0, "scale": 0.2},
    beta_M_prior={"loc": 0, "scale": 0.5},
    sigma_prior={"rate": 1},
    K=None,
):
    alpha = numpyro.sample("alpha", dist.Normal(**alpha_prior))
    beta_M = numpyro.sample("beta_M", dist.Normal(**beta_M_prior))
    mu = numpyro.deterministic("mu", alpha + beta_M * M)
    sigma = numpyro.sample("sigma", dist.Exponential(**sigma_prior))
    K = numpyro.sample("K", dist.Normal(mu, sigma), obs=K)
    return K


guide_5_6 = AutoLaplaceApproximation(model_5_6)
svi_5_6 = numpyro.infer.SVI(
    model=model_5_6,
    guide=guide_5_6,
    optim=optim,
    loss=loss,
    M=dcc["M"].values,
    K=dcc["K"].values,
).run(jrng, 1_000)

posterior_samples_5_6 = guide_5_6.sample_posterior(
    jrng, svi_5_6.params, sample_shape=(1_000,)
)
_posterior_samples_5_6 = {
    k: v for (k, v) in posterior_samples_5_6.items() if k not in ["mu"]
}
numpyro.diagnostics.print_summary(_posterior_samples_5_6, group_by_chain=False)
```

```python
M = jnp.linspace(-2, 2, 30)
posterior_predictive_5_6 = numpyro.infer.Predictive(
    model=model_5_6, posterior_samples=posterior_samples_5_6
)
posterior_predictive_samples_5_6 = posterior_predictive_5_6(jrng, M=M)

mu_mean_5_6 = posterior_predictive_samples_5_6["mu"].mean(axis=0)
mu_hpdi_5_6 = numpyro.diagnostics.hpdi(
    posterior_predictive_samples_5_6["mu"], prob=0.89
)
plt.subplot(xlabel="log body mass (std)", ylabel="kcal/g (std)")
plt.plot(df["M"], df["K"], "o")
plt.plot(M, mu_mean_5_6, color="k")
plt.fill_between(M, mu_hpdi_5_6[0], mu_hpdi_5_6[1], color="k", alpha=0.2)
```

### Code 5.38

```python
def model_5_7(
    M,
    N,
    *,
    alpha_prior={"loc": 0, "scale": 0.2},
    beta_N_prior={"loc": 0, "scale": 0.5},
    beta_M_prior={"loc": 0, "scale": 0.5},
    sigma_prior={"rate": 1},
    K=None,
):
    alpha = numpyro.sample("alpha", dist.Normal(**alpha_prior))
    beta_M = numpyro.sample("beta_M", dist.Normal(**beta_M_prior))
    beta_N = numpyro.sample("beta_N", dist.Normal(**beta_N_prior))
    mu = numpyro.deterministic("mu", alpha + beta_M * M + beta_N * N)
    sigma = numpyro.sample("sigma", dist.Exponential(**sigma_prior))
    K = numpyro.sample("K", dist.Normal(mu, sigma), obs=K)
    return K


guide_5_7 = AutoLaplaceApproximation(model_5_7)
svi_5_7 = numpyro.infer.SVI(
    model=model_5_7,
    guide=guide_5_7,
    optim=optim,
    loss=loss,
    N=dcc["N"].values,
    M=dcc["M"].values,
    K=dcc["K"].values,
).run(jrng, 5_000)

posterior_samples_5_7 = guide_5_7.sample_posterior(
    jrng, svi_5_7.params, sample_shape=(1_000,)
)
_posterior_samples_5_7 = {
    k: v for k, v in posterior_samples_5_7.items() if k not in ["mu"]
}
numpyro.diagnostics.print_summary(_posterior_samples_5_7, group_by_chain=False)
```

### Code 5.39

```python
coeftab = {
    "m5.5": guide_5_5.sample_posterior(
        jax.random.PRNGKey(1), svi_5_5.params, sample_shape=(1, 1000)
    ),
    "m5.6": guide_5_6.sample_posterior(
        jax.random.PRNGKey(2), svi_5_6.params, sample_shape=(1, 1000)
    ),
    "m5.7": guide_5_7.sample_posterior(
        jax.random.PRNGKey(3), svi_5_7.params, sample_shape=(1, 1000)
    ),
}
az.plot_forest(
    list(coeftab.values()),
    model_names=list(coeftab.keys()),
    var_names=["beta_M", "beta_N"],
    hdi_prob=0.89,
)
plt.show()
```

### Code 5.40

```python
M = jnp.linspace(-2, 2, 30)
posterior_predictive_5_7 = numpyro.infer.Predictive(
    model=model_5_7, posterior_samples=posterior_samples_5_7
)
intervention_M_samples_5_7 = posterior_predictive_5_7(jrng, M=M, N=0)
intervention_M_mu_mean_5_7 = intervention_M_samples_5_7["mu"].mean(axis=0)
intervention_M_mu_hpdi_5_7 = numpyro.diagnostics.hpdi(
    intervention_M_samples_5_7["mu"], prob=0.89
)

plt.subplot(
    xlabel="log body mass (std)",
    ylabel="kcal/g (std)",
    title="Counterfactual holding N=0",
)
plt.plot(M, intervention_M_mu_mean_5_7)
plt.fill_between(
    M,
    intervention_M_mu_hpdi_5_7[0],
    intervention_M_mu_hpdi_5_7[1],
    color="k",
    alpha=0.2,
)
```

### Code 5.41

```python
# M -> K <- N
# M -> N
n = 100
M = dist.Normal().sample(jax.random.PRNGKey(0), sample_shape=(n,))
N = dist.Normal(M).sample(jax.random.PRNGKey(1))
K = dist.Normal(N - M).sample(jax.random.PRNGKey(2))
df_sim = pd.DataFrame({"M": M, "N": N, "K": K})
df_sim.corr()
```

### Code 5.42

```python
# M -> K <- N
# N -> M
N = dist.Normal().sample(jax.random.PRNGKey(0), sample_shape=(n,))
M = dist.Normal(N).sample(jax.random.PRNGKey(1))
K = dist.Normal(N - M).sample(jax.random.PRNGKey(2))
df_sim2 = pd.DataFrame({"M": M, "N": N, "K": K})
df_sim2.corr()
```

```python
# M -> K <- N
# M <- U -> N
U = dist.Normal().sample(jax.random.PRNGKey(0), sample_shape=(n,))
M = dist.Normal(U).sample(jax.random.PRNGKey(1))
N = dist.Normal(U).sample(jax.random.PRNGKey(2))
K = dist.Normal(N - M).sample(jax.random.PRNGKey(3))
df_sim2 = pd.DataFrame({"M": M, "N": N, "K": K})
df_sim2.corr()
```

### Code 5.43

```python
dag5_7 = nx.DiGraph()
dag5_7.add_edges_from([("M", "K"), ("N", "K"), ("M", "N")])
coordinates = {"M": (0, 0.5), "K": (1, 1), "N": (2, 0.5)}
MElist = []
for i in range(2):
    for j in range(2):
        for k in range(2):
            new_dag = nx.DiGraph()
            new_dag.add_edges_from(
                [
                    edge[::-1] if flip else edge
                    for edge, flip in zip(dag5_7.edges, (i, j, k))
                ]
            )
            if not list(nx.simple_cycles(new_dag)):
                MElist.append(new_dag)


def plot_dag(dag, coordinates):
    pgm = daft.PGM()
    for node in dag.nodes:
        pgm.add_node(node, node, *coordinates[node])
    for edge in dag.edges:
        pgm.add_edge(*edge)
    pgm.render()


print("Markov Equivalent DAGs")
for dag in MElist:
    plot_dag(dag, coordinates)
```

### Code 5.44

```python
df = pd.read_csv("../data/Howell1.csv", sep=";")
df.head()
```

### Code 5.45

```python
mu_female = dist.Normal(178, 20).sample(jax.random.PRNGKey(0), sample_shape=(10_000,))
mu_male = dist.Normal(178, 20).sample(
    jax.random.PRNGKey(1), sample_shape=(10_000,)
) + dist.Normal(0, 10).sample(jax.random.PRNGKey(2), sample_shape=(10_000,))
numpyro.diagnostics.print_summary(
    {"mu_female": mu_female, "mu_male": mu_male}, group_by_chain=False
)
```

### Code 5.46

```python
df["sex"] = [1 if male else 0 for male in df["male"]]
df.head()
```

### Code 5.47

```python
def model_5_8(
    sex,
    *,
    alpha_prior={"loc": 178, "scale": 20},
    sigma_prior={"low": 0, "high": 50},
    height=None,
):
    with numpyro.plate(name="sex", size=len(set(sex))):
        alpha = numpyro.sample("alpha", dist.Normal(**alpha_prior))
    mu = numpyro.deterministic("mu", alpha[sex])
    sigma = numpyro.sample("sigma", dist.Uniform(**sigma_prior))
    height = numpyro.sample("height", dist.Normal(mu, sigma), obs=height)
    return height


guide_5_8 = AutoLaplaceApproximation(model_5_8)
svi_5_8 = numpyro.infer.SVI(
    model=model_5_8,
    guide=guide_5_8,
    optim=optim,
    loss=loss,
    sex=df["sex"].values,
    height=df["height"].values,
).run(jrng, 5_000)

posterior_samples_5_8 = guide_5_8.sample_posterior(
    jrng, svi_5_8.params, sample_shape=(1_000,)
)
_posterior_samples_5_8 = {
    k: v for (k, v) in posterior_samples_5_8.items() if k not in ["mu"]
}
numpyro.diagnostics.print_summary(
    _posterior_samples_5_8, prob=0.89, group_by_chain=False
)
```

### Code 5.48

```python
posterior_samples_5_8 = guide_5_8.sample_posterior(
    jrng, svi_5_8.params, sample_shape=(1_000,)
)
posterior_samples_5_8["diff_fm"] = (
    posterior_samples_5_8["alpha"][:, 0] - posterior_samples_5_8["alpha"][:, 1]
)
_posterior_samples_5_8 = {
    k: v for (k, v) in posterior_samples_5_8.items() if k not in ["mu"]
}
numpyro.diagnostics.print_summary(
    _posterior_samples_5_8, prob=0.89, group_by_chain=False
)
```

### Code 5.49

```python
df = pd.read_csv("../data/milk.csv", sep=";")
display(df.head())
pd.unique(df["clade"])
```

### Code 5.50

```python
labels = LabelEncoder().fit(df["clade"])
df["clade_id"] = labels.transform(df["clade"])
df.head()
```

### Code 5.51

```python
df["K"] = scale(df["kcal.per.g"])


def model_5_9(
    clade_id,
    *,
    alpha_prior={"loc": 0, "scale": 0.5},
    sigma_prior={"rate": 1},
    K=None,
):
    with numpyro.plate(name="clade", size=len(labels.classes_)):
        alpha = numpyro.sample("alpha", dist.Normal(**alpha_prior))
    mu = alpha[clade_id]
    sigma = numpyro.sample("sigma", dist.Exponential(**sigma_prior))
    K = numpyro.sample("K", dist.Normal(mu, sigma), obs=K)
    return K


guide_5_9 = AutoLaplaceApproximation(model_5_9)
svi_5_9 = numpyro.infer.SVI(
    model=model_5_9,
    guide=guide_5_9,
    optim=optim,
    loss=loss,
    clade_id=df["clade_id"].values,
    K=df["K"].values,
).run(jrng, 5_000)
```

```python
posterior_samples_5_9 = guide_5_9.sample_posterior(
    jrng, svi_5_9.params, sample_shape=(1_000,)
)
numpyro.diagnostics.print_summary(
    posterior_samples_5_9, prob=0.89, group_by_chain=False
)
az.plot_forest({"alpha": posterior_samples_5_9["alpha"][None, ...]}, hdi_prob=0.89)
plt.gca().set(
    yticklabels=[
        f"alpha[{i}]: {labels.inverse_transform([i])[0]}"
        for i in range(len(labels.classes_))
    ][::-1],
    xlabel="expected kcal (std)",
)
```

### Code 5.52

```python
df["house"] = dist.Categorical(probs=jnp.ones(4) / 4).sample(
    jrng, sample_shape=df.shape[:1]
)
```

### Code 5.53

```python
def model_5_10(
    clade,
    house,
    *,
    alpha_clade_prior={"loc": 0, "scale": 0.5},
    alpha_house_prior={"loc": 0, "scale": 0.5},
    sigma_prior={"rate": 1},
    K=None,
):
    with numpyro.plate(name="clade", size=len(labels.classes_)):
        alpha_clade = numpyro.sample("alpha_clade", dist.Normal(**alpha_clade_prior))

    with numpyro.plate(name="house", size=len(set(house))):
        alpha_house = numpyro.sample("alpha_house", dist.Normal(**alpha_house_prior))

    mu = alpha_clade[clade] + alpha_house[house]
    sigma = numpyro.sample("sigma", dist.Exponential(**sigma_prior))
    K = numpyro.sample("K", dist.Normal(mu, sigma), obs=K)
    return K


guide_5_10 = AutoLaplaceApproximation(model_5_10)
svi_5_10 = numpyro.infer.SVI(
    model=model_5_10,
    guide=guide_5_10,
    optim=optim,
    loss=loss,
    clade=df["clade_id"].values,
    house=df["house"].values,
    K=df["K"].values,
).run(jrng, 5_000)
posterior_samples_5_10 = guide_5_10.sample_posterior(
    jrng, svi_5_10.params, sample_shape=(1_000,)
)
_posterior_samples_5_10 = {
    k: v for (k, v) in posterior_samples_5_10.items() if k not in ["mu"]
}
numpyro.diagnostics.print_summary(
    _posterior_samples_5_10, prob=0.89, group_by_chain=False
)
```

<!-- #region -->
## Easy
### 5E1

Models 2 and 4.

### 5E2

$$
\begin{split}
A_i & \sim N(\mu_i, \sigma) \\
\mu_i & = a + b_l \cdot L_o + b_{p} \cdot P_i
\end{split}
$$

### 5E3

$$
\begin{split}
T_i & \sim N(\mu_i, \sigma) \\
\mu_i & = a + b_F \cdot F_i + b_S \cdot S_i
\end{split}
$$

$b_F$ and $b_S$ should both be positive.

### 5E4

Models 1, 3, 4 and 5.

## Medium

### 5M1

Predictors: temperature during the summer and A/C usage. 
Target: number of sun burns

### 5M2

Predictors: temperature during the summer and A/C usage.
Target: number of heat strokes

### 5M3

High divorce rate might lead to multiple marriage and hence high marriage rate.


### 5M4
<!-- #endregion -->

```python
df.head()
```

```python
df = pd.read_csv("../data/WaffleDivorce.csv", sep=";")
df["_LDS"] = [
    0.75,
    4.53,
    6.18,
    1,
    2.01,
    2.82,
    0.43,
    0.55,
    0.38,
    0.75,
    0.82,
    5.18,
    26.35,
    0.44,
    0.66,
    0.87,
    1.25,
    0.77,
    0.64,
    0.81,
    0.72,
    0.39,
    0.44,
    0.58,
    0.72,
    1.14,
    4.78,
    1.29,
    0.61,
    0.37,
    3.34,
    0.41,
    0.82,
    1.48,
    0.52,
    1.2,
    3.85,
    0.4,
    0.37,
    0.83,
    1.27,
    0.75,
    1.21,
    67.97,
    0.74,
    1.13,
    3.99,
    0.92,
    0.44,
    11.5,
]
df["A"] = scale(df["MedianAgeMarriage"])
df["M"] = scale(df["Marriage"])
df["D"] = scale(df["Divorce"])
df["C"] = scale(df["_LDS"])
df.head()
```

```python
def model(
    age,
    marriage,
    church,
    *,
    alpha_prior={"loc": 0, "scale": 0.2},
    beta_age_prior={"loc": 0, "scale": 0.5},
    beta_marriage_prior={"loc": 0, "scale": 0.5},
    beta_church_prior={"loc": 0, "scale": 0.5},
    sigma_prior={"rate": 1},
    divorce=None,
):
    alpha = numpyro.sample("alpha", dist.Normal(**alpha_prior))
    beta_age = numpyro.sample("beta_age", dist.Normal(**beta_age_prior))
    beta_marriage = numpyro.sample("beta_marriage", dist.Normal(**beta_marriage_prior))
    beta_church = numpyro.sample("beta_church", dist.Normal(**beta_church_prior))
    mu = numpyro.deterministic(
        "mu", alpha + beta_age * age + beta_marriage * marriage + beta_church * church
    )
    sigma = numpyro.sample("sigma", dist.Exponential(**sigma_prior))
    divorce = numpyro.sample("divorce", dist.Normal(mu, sigma), obs=divorce)
    return divorce


guide = AutoLaplaceApproximation(model)
svi = numpyro.infer.SVI(
    model=model,
    guide=guide,
    optim=optim,
    loss=loss,
    age=df["A"].values,
    marriage=df["M"].values,
    church=df["C"].values,
    divorce=df["D"].values,
).run(jrng, 2_500)
```

<!-- #raw -->
posterior_samples = guide.sample_posterior(jrng, svi.params, sample_shape=(1_000,))
posterior_samples.pop("mu")
numpyro.diagnostics.print_summary(posterior_samples, prob=0.89, group_by_chain=False)
<!-- #endraw -->

### 5M5

regress price of gasoline against avg number of steps per day per person and average number of resturant meals per day per person


## Hard

```python
df = pd.read_csv("../data/foxes.csv", sep=";")

df["F"] = scale(df["avgfood"])
df["S"] = scale(df["groupsize"])
df["A"] = scale(df["area"])
df["W"] = scale(df["weight"])
df.head()
```

### 5H1

```python
def model_h1a(
    area,
    *,
    alpha_prior={"loc": 0, "scale": 0.2},
    beta_area_prior={"loc": 0, "scale": 0.5},
    sigma_prior={"rate": 1},
    weight=None,
):
    alpha = numpyro.sample("alpha", dist.Normal(**alpha_prior))
    beta_area = numpyro.sample("beta_area", dist.Normal(**beta_area_prior))
    mu = numpyro.deterministic("mu", alpha + beta_area * area)
    sigma = numpyro.sample("sigma", dist.Exponential(**sigma_prior))
    weight = numpyro.sample("weight", dist.Normal(mu, sigma), obs=weight)
    return weight


guide_h1a = AutoLaplaceApproximation(model_h1a)
svi_h1a = numpyro.infer.SVI(
    model=model_h1a,
    guide=guide_h1a,
    optim=optim,
    loss=loss,
    area=df["A"].values,
    weight=df["W"].values,
).run(jrng, 2_500)
```

```python
posterior_samples_h1a = guide_h1a.sample_posterior(
    jrng, svi_h1a.params, sample_shape=(1_000,)
)
_posterior_samples_h1a = {k: v for (k, v) in posterior_samples_h1a.items() if k != "mu"}
numpyro.diagnostics.print_summary(
    _posterior_samples_h1a, prob=0.89, group_by_chain=False
)
```

```python
area = jnp.linspace(-2, 2, 30)
predictive_h1a = numpyro.infer.Predictive(model_h1a, posterior_samples_h1a)
predictive_samples_h1a = predictive_h1a(jrng, area=area)

mu_mean_h1a = predictive_samples_h1a["mu"].mean(axis=0)
mu_hpdi_h1a = numpyro.diagnostics.hpdi(predictive_samples_h1a["mu"], prob=0.89)

plt.subplot(xlim=(-2, 2), ylim=(-2, 2), xlabel="Area (std)", ylabel="Body weight (std)")
plt.plot(df["A"], df["W"], "o")
plt.plot(area, mu_mean_h1a)
plt.fill_between(area, mu_hpdi_h1a[0, :], mu_hpdi_h1a[1, :], color="k", alpha=0.2)
```

```python
def model_h1b(
    group_size,
    *,
    alpha_prior={"loc": 0, "scale": 0.2},
    beta_group_size_prior={"loc": 0, "scale": 0.5},
    sigma_prior={"rate": 1},
    weight=None,
):
    alpha = numpyro.sample("alpha", dist.Normal(**alpha_prior))
    beta_group_size = numpyro.sample(
        "beta_group_size", dist.Normal(**beta_group_size_prior)
    )
    mu = numpyro.deterministic("mu", alpha + beta_group_size * group_size)
    sigma = numpyro.sample("sigma", dist.Exponential(**sigma_prior))
    weight = numpyro.sample("weight", dist.Normal(mu, sigma), obs=weight)
    return weight


guide_h1b = AutoLaplaceApproximation(model_h1b)
svi_h1b = numpyro.infer.SVI(
    model=model_h1b,
    guide=guide_h1b,
    optim=optim,
    loss=loss,
    group_size=df["S"].values,
    weight=df["W"].values,
).run(jrng, 2_500)
```

```python
posterior_samples_h1b = guide_h1b.sample_posterior(
    jrng, svi_h1b.params, sample_shape=(1_000,)
)
_posterior_samples_h1b = {k: v for (k, v) in posterior_samples_h1b.items() if k != "mu"}
numpyro.diagnostics.print_summary(
    _posterior_samples_h1b, prob=0.89, group_by_chain=False
)
```

```python
group_size = jnp.linspace(-2, 2, 30)
predictive_h1b = numpyro.infer.Predictive(model_h1b, posterior_samples_h1b)
predictive_samples_h1b = predictive_h1b(jrng, group_size=group_size)

mu_mean_h1b = predictive_samples_h1b["mu"].mean(axis=0)
mu_hpdi_h1b = numpyro.diagnostics.hpdi(predictive_samples_h1b["mu"], prob=0.89)

plt.subplot(
    xlim=(-2, 2), ylim=(-2, 2), xlabel="Group Size (std)", ylabel="Body weight (std)"
)
plt.plot(df["S"], df["W"], "o")
plt.plot(group_size, mu_mean_h1b)
plt.fill_between(group_size, mu_hpdi_h1b[0, :], mu_hpdi_h1b[1, :], color="k", alpha=0.2)
```

Area not important, group size slightly more, but still not very strong.


### 5H2

```python
def model_h2(
    area,
    group_size,
    *,
    alpha_prior={"loc": 0, "scale": 0.2},
    beta_area_prior={"loc": 0, "scale": 0.5},
    beta_group_size_prior={"loc": 0, "scale": 0.5},
    sigma_prior={"rate": 1},
    weight=None,
):
    alpha = numpyro.sample("alpha", dist.Normal(**alpha_prior))
    beta_area = numpyro.sample("beta_area", dist.Normal(**beta_area_prior))
    beta_group_size = numpyro.sample(
        "beta_group_size", dist.Normal(**beta_group_size_prior)
    )
    mu = numpyro.deterministic(
        "mu", alpha + beta_area * area + beta_group_size * group_size
    )
    sigma = numpyro.sample("sigma", dist.Exponential(**sigma_prior))
    weight = numpyro.sample("weight", dist.Normal(mu, sigma), obs=weight)
    return weight


guide_h2 = AutoLaplaceApproximation(model_h2)
svi_h2 = numpyro.infer.SVI(
    model=model_h2,
    guide=guide_h2,
    optim=optim,
    loss=loss,
    area=df["A"].values,
    group_size=df["S"].values,
    weight=df["W"].values,
).run(jrng, 2_500)

posterior_samples_h2 = guide_h2.sample_posterior(
    jrng, svi_h2.params, sample_shape=(1_000,)
)
_posterior_samples_h2 = {k: v for (k, v) in posterior_samples_h2.items() if k != "mu"}
numpyro.diagnostics.print_summary(
    _posterior_samples_h2, prob=0.89, group_by_chain=False
)
```

```python
area = jnp.linspace(-2, 2, 30)
group_size = jnp.linspace(-2, 2, 30)
predictive_h2 = numpyro.infer.Predictive(model_h2, posterior_samples_h2)
area_counterfactual_h2 = predictive_h2(jrng, area=area, group_size=0)
group_size_counterfactual_h2 = predictive_h2(jrng, area=0, group_size=group_size)

area_counterfactual_mu_mean_h2 = area_counterfactual_h2["mu"].mean(axis=0)
area_counterfactual_mu_hpdi_h2 = numpyro.diagnostics.hpdi(
    area_counterfactual_h2["mu"], prob=0.89
)
group_size_counterfactual_mu_mean_h2 = group_size_counterfactual_h2["mu"].mean(axis=0)
group_size_counterfactual_mu_hpdi_h2 = numpyro.diagnostics.hpdi(
    group_size_counterfactual_h2["mu"], prob=0.89
)


fig, (ax1_h2, ax2_h2) = plt.subplots(1, 2)
fig.set_figwidth(20)
ax1_h2.set(
    xlim=(-2, 2),
    ylim=(-2, 2),
    title="Area Counterfactual",
    xlabel="Area (std)",
    ylabel="Body Weight (std)",
)
ax2_h2.set(
    xlim=(-2, 2),
    ylim=(-2, 2),
    title="Group Size Counterfactual",
    xlabel="Group Size (std)",
    ylabel="Body Weight (std)",
)

ax1_h2.plot(area, area_counterfactual_mu_mean_h2, "k")
ax1_h2.fill_between(
    area,
    area_counterfactual_mu_hpdi_h2[0, :],
    area_counterfactual_mu_hpdi_h2[1, :],
    color="k",
    alpha=0.2,
)

ax2_h2.plot(group_size, group_size_counterfactual_mu_mean_h2, "k")
ax2_h2.fill_between(
    group_size,
    group_size_counterfactual_mu_hpdi_h2[0, :],
    group_size_counterfactual_mu_hpdi_h2[1, :],
    color="k",
    alpha=0.2,
)
```

```python
df[["A", "S"]].corr()
```

<!-- #raw -->
Both relationships are now strong. 
This is because A and G and strongly positively correlated, but have opposite impacts on W. So when looking at A by itself, as we increase A, we tend to increase G, masking the impact of either on W (and vice versa when looking at G only).
<!-- #endraw -->

### 5H3

```python
def model_h3(
    area,
    food,
    group_size,
    *,
    alpha_prior={"loc": 0, "scale": 0.2},
    beta_area_prior={"loc": 0, "scale": 0.5},
    beta_food_prior={"loc": 0, "scale": 0.5},
    beta_group_size_prior={"loc": 0, "scale": 0.5},
    sigma_prior={"rate": 1},
    weight=None,
):
    alpha = numpyro.sample("alpha", dist.Normal(**alpha_prior))
    beta_area = numpyro.sample("beta_area", dist.Normal(**beta_area_prior))
    beta_food = numpyro.sample("beta_food", dist.Normal(**beta_food_prior))
    beta_group_size = numpyro.sample(
        "beta_group_size", dist.Normal(**beta_group_size_prior)
    )
    if area is None:
        mu = numpyro.deterministic(
            "mu", alpha + beta_food * food + beta_group_size * group_size
        )
    else:
        mu = numpyro.deterministic(
            "mu",
            alpha + beta_area * area + beta_food * food + beta_group_size * group_size,
        )
    sigma = numpyro.sample("sigma", dist.Exponential(**sigma_prior))
    weight = numpyro.sample("weight", dist.Normal(mu, sigma), obs=weight)
    return weight


guide_h3 = AutoLaplaceApproximation(model_h3)
svi_h3a = numpyro.infer.SVI(
    model=model_h3,
    guide=guide_h3,
    optim=optim,
    loss=loss,
    area=None,
    food=df["F"].values,
    group_size=df["S"].values,
    weight=df["W"].values,
).run(jrng, 2_500)

posterior_samples_h3a = guide_h3.sample_posterior(
    jrng, svi_h3a.params, sample_shape=(1_000,)
)
_posterior_samples_h3a = {
    k: v for (k, v) in posterior_samples_h3a.items() if k not in ["mu", "beta_area"]
}
numpyro.diagnostics.print_summary(
    _posterior_samples_h3a,
    prob=0.89,
    group_by_chain=False,
)


svi_h3b = numpyro.infer.SVI(
    model=model_h3,
    guide=guide_h3,
    optim=optim,
    loss=loss,
    area=df["A"].values,
    food=df["F"].values,
    group_size=df["S"].values,
    weight=df["W"].values,
).run(jrng, 2_500)

posterior_samples_h3b = guide_h3.sample_posterior(
    jrng, svi_h3b.params, sample_shape=(1_000,)
)
_posterior_samples_h3b = {k: v for (k, v) in posterior_samples_h3b.items() if k != "mu"}
numpyro.diagnostics.print_summary(
    _posterior_samples_h3b,
    prob=0.89,
    group_by_chain=False,
)
```

Food is better for two reasons:
 - beta coefficient is stronger
 - causally it makes more sense that food causes bigger weight and area is just a proxy for food

When both are included, their effect is reduced because they are correlated.

```python

```
