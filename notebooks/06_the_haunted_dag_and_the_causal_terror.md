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

# Chapter 6: The Haunted DAG & The Causal Terror

```python
%load_ext jupyter_black

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
import pandas as pd

from numpyro.infer import SVI
from numpyro.infer.autoguide import AutoLaplaceApproximation

seed = 84735
jrng = jax.random.key(seed)
plt.rcParams["figure.figsize"] = [10, 6]

optim = numpyro.optim.Adam(step_size=1)
loss = numpyro.infer.Trace_ELBO()
```

## Code

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

### Code 6.1

```python
N = 200  # number of grants proposals
p = 0.1  # proportion to select
_ = dist.Normal().sample(jrng, sample_shape=(2 * N,))
nw, tw = (_[:200], _[200:])
s = nw + tw
q = jnp.quantile(s, 1 - p)
selected = s >= q
jnp.corrcoef(nw[selected], tw[selected])
```

### Code 6.2

```python
N = 100  # number of individuals
height = dist.Normal(10, 2).sample(jrng, sample_shape=(N,))
_, jrng = jax.random.split(jrng)
leg_prop = dist.Uniform(low=0.4, high=0.5).sample(jrng, sample_shape=(N,))
_, jrng = jax.random.split(jrng)
leg_left = height * leg_prop + dist.Normal(0, 0.02).sample(jrng, sample_shape=(N,))
leg_right = height * leg_prop + dist.Normal(0, 0.02).sample(jrng, sample_shape=(N,))
df = pd.DataFrame(
    [height.tolist(), leg_left.tolist(), leg_right.tolist()],
    index=["height", "left", "right"],
).T
df
```

### Code 6.3

```python
def model_6_1(left, right, height):
    a = numpyro.sample("a", dist.Normal(10, 100))
    b_left = numpyro.sample("b_left", dist.Normal(2, 100))
    b_right = numpyro.sample("b_right", dist.Normal(2, 100))
    mu = numpyro.deterministic("mu", a + b_left * left + b_right * right)
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    height = numpyro.sample("height", dist.Normal(mu, sigma), obs=height)
    return height


guide_6_1 = AutoLaplaceApproximation(model_6_1)
svi_6_1 = SVI(
    model=model_6_1,
    guide=guide_6_1,
    optim=optim,
    loss=loss,
    left=df["left"].values,
    right=df["right"].values,
    height=df["height"].values,
).run(jrng, 2_500)

posterior_samples_6_1 = guide_6_1.sample_posterior(
    jrng, svi_6_1.params, sample_shape=(1_000,)
)
_posterior_samples_6_1 = prune_return_sites(posterior_samples_6_1)
numpyro.diagnostics.print_summary(
    _posterior_samples_6_1, prob=0.89, group_by_chain=False
)
```

### Code 6.4

```python
az.plot_forest(_posterior_samples_6_1, hdi_prob=0.89)
```

### Code 6.5

```python
az.plot_pair(
    {k: v[None, :] for k, v in posterior_samples_6_1.items()},
    var_names=["b_left", "b_right"],
    scatter_kwargs={"alpha": 0.1},
)
```

### Code 6.6

```python
ax = az.plot_kde(posterior_samples_6_1["b_left"] + posterior_samples_6_1["b_right"])
ax.set(title="KDE of b_left + b_right")
```

### Code 6.7

```python
def model_6_2(left, height):
    a = numpyro.sample("a", dist.Normal(10, 100))
    b_left = numpyro.sample("b_left", dist.Normal(2, 100))
    mu = numpyro.deterministic("mu", a + b_left * left)
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    height = numpyro.sample("height", dist.Normal(mu, sigma), obs=height)
    return height


guide_6_2 = AutoLaplaceApproximation(model_6_2)
svi_6_2 = SVI(
    model=model_6_2,
    guide=guide_6_2,
    optim=optim,
    loss=loss,
    left=df["left"].values,
    height=df["height"].values,
).run(jrng, 2_500)

posterior_samples_6_2 = guide_6_2.sample_posterior(
    jrng, svi_6_2.params, sample_shape=(1_000,)
)
_posterior_samples_6_2 = prune_return_sites(posterior_samples_6_2)
numpyro.diagnostics.print_summary(
    _posterior_samples_6_2, prob=0.89, group_by_chain=False
)
```

### Code 6.8

```python
df = pd.read_csv("../data/milk.csv", sep=";")
df["K"] = scale(df["kcal.per.g"])
df["F"] = scale(df["perc.fat"])
df["L"] = scale(df["perc.lactose"])
df.head()
```

### Code 6.9

```python
def model_6_3(F, K=None):
    a = numpyro.sample("a", dist.Normal(0, 0.2))
    b_F = numpyro.sample("b_F", dist.Normal(0, 0.5))
    mu = numpyro.deterministic("mu", a + b_F * F)
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    K = numpyro.sample("K", dist.Normal(mu, sigma), obs=K)
    return K


guide_6_3 = AutoLaplaceApproximation(model_6_3)
svi_6_3 = SVI(
    model=model_6_3,
    guide=guide_6_3,
    optim=optim,
    loss=loss,
    F=df["F"].values,
    K=df["K"].values,
).run(jrng, 2_500)

posterior_samples_6_3 = guide_6_3.sample_posterior(
    jrng, svi_6_3.params, sample_shape=(1_000,)
)
_posterior_samples_6_3 = prune_return_sites(posterior_samples_6_3)
numpyro.diagnostics.print_summary(
    _posterior_samples_6_3, prob=0.89, group_by_chain=False
)


def model_6_4(L, K=None):
    a = numpyro.sample("a", dist.Normal(0, 0.2))
    b_L = numpyro.sample("b_L", dist.Normal(0, 0.5))
    mu = numpyro.deterministic("mu", a + b_L * L)
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    K = numpyro.sample("K", dist.Normal(mu, sigma), obs=K)
    return K


guide_6_4 = AutoLaplaceApproximation(model_6_4)
svi_6_4 = SVI(
    model=model_6_4,
    guide=guide_6_4,
    optim=optim,
    loss=loss,
    L=df["L"].values,
    K=df["K"].values,
).run(jrng, 2_500)

posterior_samples_6_4 = guide_6_4.sample_posterior(
    jrng, svi_6_4.params, sample_shape=(1_000,)
)
_posterior_samples_6_4 = prune_return_sites(posterior_samples_6_4)
numpyro.diagnostics.print_summary(
    _posterior_samples_6_4, prob=0.89, group_by_chain=False
)
```

### Code 6.10

```python
def model_6_5(F, L, K=None):
    a = numpyro.sample("a", dist.Normal(0, 0.2))
    b_F = numpyro.sample("b_F", dist.Normal(0, 0.5))
    b_L = numpyro.sample("b_L", dist.Normal(0, 0.5))
    mu = numpyro.deterministic("mu", a + b_F * F + b_L * L)
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    K = numpyro.sample("K", dist.Normal(mu, sigma), obs=K)
    return K


guide_6_5 = AutoLaplaceApproximation(model_6_5)
svi_6_5 = SVI(
    model=model_6_5,
    guide=guide_6_5,
    optim=optim,
    loss=loss,
    F=df["F"].values,
    L=df["L"].values,
    K=df["K"].values,
).run(jrng, 2_500)

posterior_samples_6_5 = guide_6_5.sample_posterior(
    jrng, svi_6_5.params, sample_shape=(1_000,)
)
_posterior_samples_6_5 = prune_return_sites(posterior_samples_6_5)
numpyro.diagnostics.print_summary(
    _posterior_samples_6_5, prob=0.89, group_by_chain=False
)
```

### Code 6.11

```python
az.plot_pair(
    df.to_dict(orient="list"), var_names=["kcal.per.g", "perc.fat", "perc.lactose"]
)
```

### Code 6.12

```python
milk = pd.read_csv("../data/milk.csv", sep=";")
d = milk


def sim_coll(i, r=0.9):
    sd = jnp.sqrt((1 - r**2) * jnp.var(d["perc.fat"].values))
    x = dist.Normal(r * d["perc.fat"].values, sd).sample(jax.random.PRNGKey(3 * i))

    def model(perc_fat, kcal_per_g):
        intercept = numpyro.sample("intercept", dist.Normal(0, 10))
        b_perc_flat = numpyro.sample("b_perc.fat", dist.Normal(0, 10))
        b_x = numpyro.sample("b_x", dist.Normal(0, 10))
        sigma = numpyro.sample("sigma", dist.HalfCauchy(2))
        mu = intercept + b_perc_flat * perc_fat + b_x * x
        numpyro.sample("kcal.per.g", dist.Normal(mu, sigma), obs=kcal_per_g)

    m = AutoLaplaceApproximation(model)
    svi = SVI(
        model,
        m,
        numpyro.optim.Adam(0.01),
        numpyro.infer.Trace_ELBO(),
        perc_fat=d["perc.fat"].values,
        kcal_per_g=d["kcal.per.g"].values,
    )
    svi_result = svi.run(jax.random.PRNGKey(3 * i + 1), 20000, progress_bar=False)
    params = svi_result.params
    samples = m.sample_posterior(
        jax.random.PRNGKey(3 * i + 2), params, sample_shape=(1000,)
    )
    vcov = jnp.cov(jnp.stack(list(samples.values()), axis=0))
    stddev = jnp.sqrt(jnp.diag(vcov))  # stddev of parameter
    return dict(zip(samples.keys(), stddev))["b_perc.fat"]


def rep_sim_coll(r=0.9, n=100):
    stddev = jax.lax.map(lambda i: sim_coll(i, r=r), jnp.arange(n))
    return jnp.nanmean(stddev)


r_seq = jnp.arange(start=0, stop=1, step=0.01)
stddev = jax.lax.map(lambda z: rep_sim_coll(r=z, n=100), r_seq)
plt.plot(r_seq, stddev)
plt.xlabel("correlation")
plt.show()
```

### Code 6.13

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
```

### Code 6.14

```python
sim_p = dist.LogNormal(0, 0.25).sample(jrng, sample_shape=(10_000,))
display(pd.Series(sim_p).describe())
pd.DataFrame(sim_p).plot(kind="kde")
```

### Code 6.15

```python
def model_6_6(h0, h1):
    p = numpyro.sample("p", dist.LogNormal(0, 0.25))
    mu = numpyro.deterministic("mu", h0 * p)
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    h1 = numpyro.sample("h1", dist.Normal(mu, sigma), obs=h1)
    return h1


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
    jrng, svi_6_6.params, sample_shape=(2_500,)
)
numpyro.diagnostics.print_summary(
    prune_return_sites(posterior_samples_6_6),
    prob=0.89,
    group_by_chain=False,
)
```

### Code 6.16

```python
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
).run(jrng, 5_000)
posterior_samples_6_7 = guide_6_7.sample_posterior(
    jrng, svi_6_7.params, sample_shape=(2_500,)
)
numpyro.diagnostics.print_summary(
    prune_return_sites(posterior_samples_6_7),
    prob=0.89,
    group_by_chain=False,
)
```

### Code 6.17

```python
def model_6_8(h0, treatment, h1):
    alpha = numpyro.sample("alpha", dist.LogNormal(0, 0.25))
    beta_treatment = numpyro.sample("beta_treatment", dist.Normal(0, 0.5))
    p = numpyro.deterministic("p", alpha + beta_treatment * treatment)
    mu = numpyro.deterministic("mu", h0 * p)
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    h1 = numpyro.sample("h1", dist.Normal(mu, sigma), obs=h1)
    return h1


guide_6_8 = AutoLaplaceApproximation(model_6_8)
svi_6_8 = SVI(
    model=model_6_8,
    guide=guide_6_8,
    optim=optim,
    loss=loss,
    h0=df["h0"].values,
    treatment=df["treatment"].values,
    h1=df["h1"].values,
).run(jrng, 5_000)
posterior_samples_6_8 = guide_6_8.sample_posterior(
    jrng, svi_6_8.params, sample_shape=(2_500,)
)
numpyro.diagnostics.print_summary(
    prune_return_sites(posterior_samples_6_8),
    prob=0.89,
    group_by_chain=False,
)
```

### Code 6.18

```python
plant_dag = nx.DiGraph()
plant_dag.add_edges_from([("H0", "H1"), ("F", "H1"), ("T", "F")])
pgm = daft.PGM()
coordinates = {"H0": (0, 0), "T": (4, 0), "F": (3, 0), "H1": (2, 0)}
for node in plant_dag.nodes:
    pgm.add_node(node, node, *coordinates[node])
for edge in plant_dag.edges:
    pgm.add_edge(*edge)
with plt.rc_context({"figure.constrained_layout.use": False}):
    pgm.render()
```

### Code 6.19

```python
conditional_independencies = collections.defaultdict(list)
for edge in itertools.combinations(sorted(plant_dag.nodes), 2):
    remaining = sorted(set(plant_dag.nodes) - set(edge))
    for size in range(len(remaining) + 1):
        for subset in itertools.combinations(remaining, size):
            if any(
                cond.issubset(set(subset)) for cond in conditional_independencies[edge]
            ):
                continue
            if nx.is_d_separator(plant_dag, {edge[0]}, {edge[1]}, set(subset)):
                conditional_independencies[edge].append(set(subset))
                print(
                    f"{edge[0]} _||_ {edge[1]}"
                    + (f" | {' '.join(subset)}" if subset else "")
                )
```

### Code 6.20

```python
N = 1_000
h0 = dist.Normal(10, 2).sample(jrng, sample_shape=(N,))
_, jrng = jax.random.split(jrng)
treatment = dist.Categorical(probs=jnp.array([1 / 2, 1 / 2])).sample(
    jrng, sample_shape=(N,)
)
_, jrng = jax.random.split(jrng)
M = dist.Bernoulli(probs=jnp.array(1 / 2)).sample(jrng, sample_shape=(N,))
_, jrng = jax.random.split(jrng)
fungus = dist.Bernoulli(probs=0.5 - treatment * 0.4 + 0.4 * M).sample(jrng)
_, jrng = jax.random.split(jrng)
h1 = h0 + dist.Normal(5 + 3 * M).sample(jrng)
df2 = pd.DataFrame({"h0": h0, "h1": h1, "treatment": treatment, "fungus": fungus})
df2.head()
```

```python
svi_6_7 = SVI(
    model=model_6_7,
    guide=guide_6_7,
    optim=optim,
    loss=loss,
    h0=df2["h0"].values,
    treatment=df2["treatment"].values,
    fungus=df2["fungus"].values,
    h1=df2["h1"].values,
).run(jrng, 10_000)
posterior_samples_6_7 = guide_6_7.sample_posterior(
    jrng, svi_6_7.params, sample_shape=(2_500,)
)
numpyro.diagnostics.print_summary(
    prune_return_sites(posterior_samples_6_7),
    prob=0.89,
    group_by_chain=False,
)
```

```python
svi_6_8 = SVI(
    model=model_6_8,
    guide=guide_6_8,
    optim=optim,
    loss=loss,
    h0=df2["h0"].values,
    treatment=df2["treatment"].values,
    h1=df2["h1"].values,
).run(jrng, 10_000)
posterior_samples_6_8 = guide_6_8.sample_posterior(
    jrng, svi_6_8.params, sample_shape=(2_500,)
)
numpyro.diagnostics.print_summary(
    prune_return_sites(posterior_samples_6_8),
    prob=0.89,
    group_by_chain=False,
)
```

### Code 6.21

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
numpyro.diagnostics.print_summary(dict(zip(df.columns, df.T.values)), 0.89, False)
```

### Code 6.22

```python
df2 = df.loc[df["age"] > 17, :].copy()
df2["A"] = (df2["age"] - 18) / (65 - 18)
df2.head()
```

### Code 6.23

```python
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
```

### Code 6.24

```python
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
```

### Code 6.25

```python
N = 200
b_GP = 1
b_GC = 0
b_PC = 1
b_U = 2
```

```python
U = 2 * dist.Bernoulli(0.5).sample(jrng, sample_shape=(N,)) - 1
_, jrng = jax.random.split(jrng)
G = dist.Normal(0, 1).sample(jrng, sample_shape=(N,))
_, jrng = jax.random.split(jrng)
P = dist.Normal(b_GP * G + b_U * U, 1).sample(jrng)
_, jrng = jax.random.split(jrng)
C = dist.Normal(b_GC * G + b_PC * P + b_U * U, 1).sample(jrng)
_, jrng = jax.random.split(jrng)
df = pd.DataFrame({"C": C, "G": G, "P": P, "U": U})
df.head()
```

### Code 6.27

```python
def model_6_11(G, P, C):
    a = numpyro.sample("a", dist.Normal(0, 1))
    b_GC = numpyro.sample("b_GC", dist.Normal(0, 1))
    b_PC = numpyro.sample("b_PC", dist.Normal(0, 1))
    mu = a + b_GC * G + b_PC * P
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    C = numpyro.sample("C", dist.Normal(mu, sigma), obs=C)
    return C


guide_6_11 = AutoLaplaceApproximation(model_6_11)
svi_6_11 = SVI(model=model_6_11, guide=guide_6_11, optim=optim, loss=loss).run(
    jrng,
    G=df["G"].values,
    P=df["P"].values,
    C=df["C"].values,
    num_steps=2_500,
)

posterior_samples_6_11 = guide_6_11.sample_posterior(
    jrng, svi_6_11.params, sample_shape=(2500,)
)
numpyro.diagnostics.print_summary(
    prune_return_sites(posterior_samples_6_11), group_by_chain=False
)
```

```python
def model_6_12(G, P, C, U):
    a = numpyro.sample("a", dist.Normal(0, 1))
    b_GC = numpyro.sample("b_GC", dist.Normal(0, 1))
    b_PC = numpyro.sample("b_PC", dist.Normal(0, 1))
    b_U = numpyro.sample("b_U", dist.Normal(0, 1))
    mu = a + b_GC * G + b_PC * P + b_U * U
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    C = numpyro.sample("C", dist.Normal(mu, sigma), obs=C)
    return C


guide_6_12 = AutoLaplaceApproximation(model_6_12)
svi_6_12 = SVI(model=model_6_12, guide=guide_6_12, optim=optim, loss=loss).run(
    jrng,
    G=df["G"].values,
    P=df["P"].values,
    C=df["C"].values,
    U=df["U"].values,
    num_steps=5_000,
)

posterior_samples_6_12 = guide_6_12.sample_posterior(
    jrng, svi_6_12.params, sample_shape=(2500,)
)
numpyro.diagnostics.print_summary(
    prune_return_sites(posterior_samples_6_12), group_by_chain=False
)
```

### Code 6.29

```python
dag_6_1 = nx.DiGraph()
dag_6_1.add_edges_from(
    [("X", "Y"), ("U", "X"), ("A", "U"), ("A", "C"), ("C", "Y"), ("U", "B"), ("C", "B")]
)
backdoor_paths = [
    path
    for path in nx.all_simple_paths(dag_6_1.to_undirected(), "X", "Y")
    if dag_6_1.has_edge(path[1], "X")
]
remaining = sorted(
    set(dag_6_1.nodes) - {"X", "Y", "U"} - set(nx.descendants(dag_6_1, "X"))
)
adjustment_sets = []
for size in range(len(remaining) + 1):
    for subset in itertools.combinations(remaining, size):
        subset = set(subset)
        if any(s.issubset(subset) for s in adjustment_sets):
            continue
        need_adjust = True
        for path in backdoor_paths:
            d_separated = False
            for x, z, y in zip(path[:-2], path[1:-1], path[2:]):
                if dag_6_1.has_edge(x, z) and dag_6_1.has_edge(y, z):
                    if set(nx.descendants(dag_6_1, z)) & subset:
                        continue
                    d_separated = z not in subset
                else:
                    d_separated = z in subset
                if d_separated:
                    break
            if not d_separated:
                need_adjust = False
                break
        if need_adjust:
            adjustment_sets.append(subset)
            print(subset)
```

### Code 6.30

```python
dag_6_2 = nx.DiGraph()
dag_6_2.add_edges_from(
    [("S", "A"), ("A", "D"), ("S", "M"), ("M", "D"), ("S", "W"), ("W", "D"), ("A", "M")]
)
backdoor_paths = [
    path
    for path in nx.all_simple_paths(dag_6_2.to_undirected(), "W", "D")
    if dag_6_2.has_edge(path[1], "W")
]
remaining = sorted(set(dag_6_2.nodes) - {"W", "D"} - set(nx.descendants(dag_6_2, "W")))
adjustment_sets = []
for size in range(len(remaining) + 1):
    for subset in itertools.combinations(remaining, size):
        subset = set(subset)
        if any(s.issubset(subset) for s in adjustment_sets):
            continue
        need_adjust = True
        for path in backdoor_paths:
            d_separated = False
            for x, z, y in zip(path[:-2], path[1:-1], path[2:]):
                if dag_6_2.has_edge(x, z) and dag_6_2.has_edge(y, z):
                    if set(nx.descendants(dag_6_2, z)) & subset:
                        continue
                    d_separated = z not in subset
                else:
                    d_separated = z in subset
                if d_separated:
                    break
            if not d_separated:
                need_adjust = False
                break
        if need_adjust:
            adjustment_sets.append(subset)
            print(subset)
```

### Code 6.31

```python
conditional_independencies = collections.defaultdict(list)
for edge in itertools.combinations(sorted(dag_6_2.nodes), 2):
    remaining = sorted(set(dag_6_2.nodes) - set(edge))
    for size in range(len(remaining) + 1):
        for subset in itertools.combinations(remaining, size):
            if any(
                cond.issubset(set(subset)) for cond in conditional_independencies[edge]
            ):
                continue
            if nx.is_d_separator(dag_6_2, {edge[0]}, {edge[1]}, set(subset)):
                conditional_independencies[edge].append(set(subset))
                print(
                    f"{edge[0]} _||_ {edge[1]}"
                    + (f" | {' '.join(subset)}" if subset else "")
                )
```

## Medium

### 6M1

There are now 3 paths (other then the direct path) between X and Y. The new path is a fork, so it's open. We can close it by conditionning on V.


## Hard
### 6H1

```python
dag = nx.DiGraph()
dag.add_edges_from(
    [("S", "W"), ("S", "A"), ("S", "M"), ("A", "M"), ("A", "D"), ("M", "D"), ("W", "D")]
)
pgm = daft.PGM()
coordinates = {"S": (0, 3), "W": (3, 3), "A": (0, 0), "M": (1, 2), "D": (3, 0)}
for node in dag.nodes:
    pgm.add_node(node, node, *coordinates[node])
for edge in dag.edges:
    pgm.add_edge(*edge, directed=True)
with plt.rc_context({"figure.constrained_layout.use": False}):
    pgm.render()
```

There are three backdoor paths from W to D, which are all open and we can all close by conditionning on S.

```python
df = pd.read_csv("../data/WaffleDivorce.csv", sep=";")
df["W"] = scale(df["WaffleHouses"])
df["S"] = [
    (
        1
        if x
        in [
            "Alabama",
            "Arkansas",
            "Georgia",
            "Indiana",
            "Louisiana",
            "Mississippi",
            "Oklahoma",
            "South Carolina",
            "Tennessee",
            "Virginia",
        ]
        else 0
    )
    for x in df["Location"]
]
df["D"] = scale(df["Divorce"])
df.head()
```

```python
def model_6h1(S, W, D):
    a = numpyro.sample("a", dist.Normal(0, 0.5))
    b_W = numpyro.sample("b_W", dist.Normal(0, 0.1))
    b_S = numpyro.sample("b_S", dist.Normal(0, 0.2))
    mu = a + b_W * W + b_S * S
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    D = numpyro.sample("D", dist.Normal(mu, sigma), obs=D)
    return D


guide_6h1 = AutoLaplaceApproximation(model_6h1)
svi_6h1 = SVI(model=model_6h1, guide=guide_6h1, optim=optim, loss=loss).run(
    jrng, S=df["S"].values, W=df["W"].values, D=df["D"].values, num_steps=5_000
)
posterior_samples_6h1 = guide_6h1.sample_posterior(
    jrng, svi_6h1.params, sample_shape=(2_500,)
)
numpyro.diagnostics.print_summary(
    prune_return_sites(posterior_samples_6h1), group_by_chain=False
)
```

### 6H2

```python
df["A"] = scale(df["MedianAgeMarriage"])
```

```python
# A _||_ W | S
def model_6h2_1(S, W, A):
    a = numpyro.sample("a", dist.Normal(0, 0.5))
    b_W = numpyro.sample("b_W", dist.Normal(0, 0.1))
    b_S = numpyro.sample("b_S", dist.Normal(0, 0.2))
    mu = a + b_W * W + b_S * S
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    A = numpyro.sample("A", dist.Normal(mu, sigma), obs=A)
    return A


guide_6h2_1 = AutoLaplaceApproximation(model_6h2_1)
svi_6h2_1 = SVI(model=model_6h2_1, guide=guide_6h2_1, optim=optim, loss=loss).run(
    jrng, S=df["S"].values, W=df["W"].values, A=df["A"].values, num_steps=5_000
)
posterior_samples_6h2_1 = guide_6h2_1.sample_posterior(
    jrng, svi_6h2_1.params, sample_shape=(2_500,)
)
numpyro.diagnostics.print_summary(
    prune_return_sites(posterior_samples_6h2_1), group_by_chain=False
)
print("Given b_W, A appears to be independant from W conditionned on S")
```

```python
df["M"] = scale(df["Marriage"])
```

```python
# D _||_ S | A, M, W
def model_6h2_1(S, A, M, W, D):
    a = numpyro.sample("a", dist.Normal(0, 0.5))
    b_S = numpyro.sample("b_S", dist.Normal(0, 0.2))
    b_A = numpyro.sample("b_A", dist.Normal(0, 0.2))
    b_M = numpyro.sample("b_M", dist.Normal(0, 0.2))
    b_W = numpyro.sample("b_W", dist.Normal(0, 0.2))
    mu = a + b_S * S + b_A * A + b_M * M + b_W * W
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    D = numpyro.sample("D", dist.Normal(mu, sigma), obs=D)
    return D


guide_6h2_1 = AutoLaplaceApproximation(model_6h2_1)
svi_6h2_1 = SVI(model=model_6h2_1, guide=guide_6h2_1, optim=optim, loss=loss).run(
    jrng,
    S=df["S"].values,
    A=df["A"].values,
    M=df["M"].values,
    W=df["W"].values,
    D=df["D"].values,
    num_steps=5_000,
)
posterior_samples_6h2_1 = guide_6h2_1.sample_posterior(
    jrng, svi_6h2_1.params, sample_shape=(2_500,)
)
numpyro.diagnostics.print_summary(
    prune_return_sites(posterior_samples_6h2_1), group_by_chain=False
)
print("Given b_S, D appears to be independant from S conditionned on A, M and W")
```

```python
# M _||_ W | S
print("As seen on 2h1, not enough evidence to reject the DAG.")
```

```python

```
