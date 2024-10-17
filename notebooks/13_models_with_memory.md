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

# Chapter 13: Models With Memory

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

## Code
### Code 13.1

```python
df = pd.read_csv("../data/reedfrogs.csv", sep=";")
df
```

### Code 13.2

```python
# make the tank cluster variable
df["tank"] = jnp.arange(df.shape[0])


def model_13_1(tank, S, N):
    with numpyro.plate("tank", 48):
        a = numpyro.sample("a", dist.Normal(0, 1.5))

    logit_p = a[tank]
    p = numpyro.deterministic("p", jax.scipy.special.expit(logit_p))
    numpyro.sample("S", dist.Binomial(total_count=N, probs=p), obs=S)


training_data = {
    "tank": df["tank"].values,
    "S": df["surv"].values,
    "N": df["density"].values,
}
mcmc_13_1 = MCMC(NUTS(model_13_1), num_warmup=500, num_samples=500, num_chains=4)
mcmc_13_1.run(jrngs[0], **training_data)
mcmc_13_1.print_summary()
idata_13_1 = az.from_numpyro(mcmc_13_1)
```

### Code 13.13

```python
def model_13_2(tank, S, N):
    a_bar = numpyro.sample("a_bar", dist.Normal(0, 1.5))
    sigma_a = numpyro.sample("sigma_a", dist.Exponential(1))
    with numpyro.plate("tank", 48):
        a = numpyro.sample("a", dist.Normal(a_bar, sigma_a))

    logit_p = a[tank]
    p = numpyro.deterministic("p", jax.scipy.special.expit(logit_p))
    numpyro.sample("S", dist.Binomial(total_count=N, probs=p), obs=S)


training_data = {
    "tank": df["tank"].values,
    "S": df["surv"].values,
    "N": df["density"].values,
}
mcmc_13_2 = MCMC(NUTS(model_13_2), num_warmup=500, num_samples=500, num_chains=4)
mcmc_13_2.run(jrngs[1], **training_data)
mcmc_13_2.print_summary()
idata_13_2 = az.from_numpyro(mcmc_13_2)
```

### Code 13.4

```python
az.compare(
    {"mcmc_13_1": idata_13_1, "mcmc_13_2": idata_13_2}, scale="deviance", ic="waic"
)
```

### Code 13.5

```python
# extract samples
posterior_samples_13_2 = mcmc_13_2.get_samples()

# compute median intercept for each tank
# also transform to probability with logistic
df["propsurv.est"] = jax.scipy.special.expit(posterior_samples_13_2["a"].mean(axis=0))

# display raw proportions surviving in each tank
plt.subplot(xlabel="tank", ylabel="proportion surviving", ylim=(-0.05, 1.05))
plt.plot(range(1, 49), df["propsurv"], "o", alpha=0.5, zorder=3)
plt.gca().set(ylim=(-0.05, 1.05), xlabel="tank", ylabel="proportion survival")
plt.gca().set(xticks=[1, 16, 32, 48], xticklabels=[1, 16, 32, 48])

# overlay posterior means
plt.plot(jnp.arange(1, 49), df["propsurv.est"], "ko", mfc="w")

# mark posterior mean probability across tanks
plt.gca().axhline(
    y=jnp.mean(jax.scipy.special.expit(posterior_samples_13_2["a_bar"])),
    c="k",
    ls="--",
    lw=1,
)

# draw vertical dividers between tank densities
plt.gca().axvline(x=16.5, c="k", lw=0.5)
plt.gca().axvline(x=32.5, c="k", lw=0.5)
plt.annotate("small tanks", (8, 0), ha="center")
plt.annotate("medium tanks", (16 + 8, 0), ha="center")
plt.annotate("large tanks", (32 + 8, 0), ha="center")
plt.show()
```

### Code 13.6

```python
# show first 100 populations in the posterior
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.set(xlabel="log-odds survival", ylabel="density", xlim=(-3, 4), ylim=(0, 0.35))
x = jnp.linspace(-3, 4, 101)
for i in range(100):
    y = jax.scipy.stats.norm.pdf(
        x,
        loc=posterior_samples_13_2["a_bar"][i],
        scale=posterior_samples_13_2["sigma_a"][i],
    )
    ax1.plot(x, y, "k", alpha=0.2)

# sample 8000 imaginary tanks from the poserior distribution
idxs = jax.random.randint(jrngs[0], (8000,), minval=0, maxval=3999)
sim_tanks = dist.Normal(
    posterior_samples_13_2["a_bar"][idxs], posterior_samples_13_2["sigma_a"][idxs]
).sample(jrngs[1])
az.plot_kde(jax.scipy.special.expit(sim_tanks), bw=0.3, ax=ax2)
ax2.set(xlabel="probability survival", ylabel="density")
```

### Code 13.7

```python
a_bar = 1.5
sigma = 1.5
n_ponds = 60
N_i = jnp.repeat(jnp.array([5, 10, 25, 35]), 15)
```

### Code 13.8

```python
a_pond = dist.Normal(a_bar, sigma).sample(jrng, (n_ponds,))
a_pond
```

### Code 13.9

```python
dsim = pd.DataFrame({"pond": jnp.arange(n_ponds), "N": N_i, "true": a_pond})
dsim
```

### Code 13.10

```python
display(jnp.arange(3).dtype)
display(jnp.array([1.0, 2, 3]).dtype)
```

### Code 13.11

```python
dsim["S"] = dist.Binomial(
    total_count=dsim["N"].values,
    probs=jax.scipy.special.expit(dsim["true"].values),
).sample(jrng)
dsim.head()
```

### Code 13.12

```python
dsim["p_no_pool"] = dsim["S"] / dsim["N"]
dsim.head()
```

### Code 13.13

```python
def model_13_3(N, pond, S):
    a_bar = numpyro.sample("a_bar", dist.Normal(0, 1.5))
    sigma_a = numpyro.sample("sigma_a", dist.Exponential(1))
    with numpyro.plate("pond", 60):
        a = numpyro.sample("a", dist.Normal(a_bar, sigma_a))

    logit_p = a[pond]
    p = numpyro.deterministic("p", jax.scipy.special.expit(logit_p))
    numpyro.sample("S", dist.Binomial(total_count=N, probs=p), obs=S)


training_data = {
    "N": dsim["N"].values,
    "pond": dsim["pond"].values,
    "S": dsim["S"].values,
}
mcmc_13_3 = MCMC(NUTS(model_13_3), num_warmup=500, num_samples=500, num_chains=4)
mcmc_13_3.run(jrng, **training_data)
idata_13_3 = az.from_numpyro(mcmc_13_3)
```

### Code 13.14

```python
mcmc_13_3.print_summary()
```

### Code 13.15

```python
posterior_samples_13_3 = mcmc_13_3.get_samples()
dsim["p_partial_pool"] = jax.scipy.special.expit(
    posterior_samples_13_3["a"].mean(axis=0)
)
dsim.head()
```

### Code 13.16

```python
dsim["p_true"] = jax.scipy.special.expit(dsim["true"].values)
dsim.head()
```

### Code 13.17

```python
dsim["no_pool_error"] = dsim["p_no_pool"] - dsim["p_true"]
dsim["partial_pool_error"] = dsim["p_partial_pool"] - dsim["p_true"]
dsim.head()
```

### Code 13.18

```python
plt.scatter(range(1, 61), dsim["no_pool_error"], label="nopool", alpha=0.8)
plt.gca().set(xlabel="pond", ylabel="absolute error")
plt.scatter(
    range(1, 61),
    dsim["partial_pool_error"],
    label="partpool",
    s=50,
    edgecolor="black",
    facecolor="none",
)
plt.legend()
plt.show()
```

### Code 13.19

```python
no_pool_avg = dsim.groupby("N")["no_pool_error"].mean()
part_pool_avg = dsim.groupby("N")["partial_pool_error"].mean()
pd.concat([no_pool_avg, part_pool_avg], axis=1)
```

### Code 13.20

```python
a = 1.5
sigma = 1.5
n_ponds = 60
Ni = jnp.repeat(jnp.array([5, 10, 25, 35]), 15)
a_pond = dist.Normal(a, sigma).sample(jrng, (n_ponds,))
dsim = pd.DataFrame({"pond": jnp.arange(n_ponds), "N": Ni, "true": a_pond})
dsim["S"] = dist.Binomial(
    total_count=dsim["N"].values,
    probs=jax.scipy.special.expit(dsim["true"].values),
).sample(jrngs[1])
dsim["p_no_pool"] = dsim["S"] / dsim["N"]
new_training_data = {
    "N": dsim["N"].values,
    "pond": dsim["pond"].values,
    "S": dsim["S"].values,
}
model_13_3new = MCMC(
    NUTS(model_13_3), num_warmup=1_000, num_samples=1_000, num_chains=4
)
model_13_3new.run(jrngs[2], **new_training_data)

post = model_13_3new.get_samples()
dsim["p_partial_pool"] = jnp.mean(jax.scipy.special.expit(post["a"]), 0)
dsim["p_true"] = jax.scipy.special.expit(dsim["true"].values)
nopool_error = (dsim["p_no_pool"] - dsim.p_true).abs()
partpool_error = (dsim["p_partial_pool"] - dsim.p_true).abs()
plt.scatter(range(1, 61), nopool_error, label="nopool", alpha=0.8)
plt.gca().set(xlabel="pond", ylabel="absolute error")
plt.scatter(
    range(1, 61),
    partpool_error,
    label="partpool",
    s=50,
    edgecolor="black",
    facecolor="none",
)
plt.legend()
plt.show()
```

### Code 13.21

```python
df = pd.read_csv("../data/chimpanzees.csv", sep=";")
df["treatment"] = df["prosoc_left"] + 2 * df["condition"]
df
```

```python
def model_13_4(treatment, actor, block, pulled_left):
    # hyper-priors
    a_bar = numpyro.sample("a_bar", dist.Normal(0, 1.5))
    sigma_a = numpyro.sample("sigma_a", dist.Exponential(1))
    sigma_g = numpyro.sample("sigma_g", dist.Exponential(1))

    # adaptive priors
    with numpyro.plate("actor", 7):
        a = numpyro.sample("a", dist.Normal(a_bar, sigma_a))
    with numpyro.plate("block", 6):
        g = numpyro.sample("g", dist.Normal(0, sigma_g))

    # old fashioned priors
    with numpyro.plate("treatment", 4):
        b_treatment = numpyro.sample("b_treatment", dist.Normal(0, 0.5))

    # likelihood
    p = numpyro.deterministic(
        "p", jax.scipy.special.expit(a[actor] + g[block] + b_treatment[treatment])
    )
    numpyro.sample("pulled_left", dist.Bernoulli(probs=p), obs=pulled_left)


training_data = {
    "treatment": df["treatment"].values,
    "actor": df["actor"].values - 1,
    "block": df["block"].values - 1,
    "pulled_left": df["pulled_left"].values,
}
mcmc_13_4 = MCMC(NUTS(model_13_4), num_warmup=500, num_samples=500, num_chains=4)
mcmc_13_4.run(jrngs[0], **training_data)
idata_13_4 = az.from_numpyro(mcmc_13_4)
az.plot_trace(
    idata_13_4,
    var_names=["a_bar", "sigma_a", "sigma_g", "a", "g", "b_treatment"],
    compact=False,
    kind="rank_bars",
)
plt.tight_layout()
```

### Code 13.22

```python
mcmc_13_4.print_summary()
```

```python
az.plot_forest(idata_13_4, var_names=["a", "g", "b_treatment"], combined=True)
```

### Code 13.23

```python
def model_13_5(treatment, actor, pulled_left):
    a_bar = numpyro.sample("a_bar", dist.Normal(0, 1.5))
    sigma_a = numpyro.sample("sigma_a", dist.Exponential(1))

    with numpyro.plate("actor", 7):
        a = numpyro.sample("a", dist.Normal(a_bar, sigma_a))
    with numpyro.plate("treatment", 4):
        b_treatment = numpyro.sample("b_treatment", dist.Normal(0, 0.5))

    p = numpyro.deterministic(
        "p", jax.scipy.special.expit(a[actor] + b_treatment[treatment])
    )
    numpyro.sample("pulled_left", dist.Bernoulli(probs=p), obs=pulled_left)


training_data = {
    "treatment": df["treatment"].values,
    "actor": df["actor"].values - 1,
    "pulled_left": df["pulled_left"].values,
}
mcmc_13_5 = MCMC(NUTS(model_13_5), num_warmup=500, num_samples=500, num_chains=4)
mcmc_13_5.run(jrngs[1], **training_data)
mcmc_13_5.print_summary()
idata_13_5 = az.from_numpyro(mcmc_13_5)
az.plot_trace(
    idata_13_5,
    var_names=["a_bar", "sigma_a", "a", "b_treatment"],
    compact=False,
    kind="rank_bars",
)
plt.tight_layout()
```

### Code 13.24

```python
az.compare(
    {"mcmc_13_4": idata_13_4, "mcmc_13_5": idata_13_5}, scale="deviance", ic="waic"
)
```

### Code 13.25

```python
def model_13_6(actor, block, treatment, pulled_left):
    a_bar = numpyro.sample("a_bar", dist.Normal(0, 1.5))
    sigma_a = numpyro.sample("sigma_a", dist.Exponential(1))
    sigma_g = numpyro.sample("sigma_g", dist.Exponential(1))
    sigma_b = numpyro.sample("sigma_b", dist.Exponential(1))

    with numpyro.plate("actor", 7):
        a = numpyro.sample("a", dist.Normal(a_bar, sigma_a))
    with numpyro.plate("block", 6):
        g = numpyro.sample("g", dist.Normal(0, sigma_g))
    with numpyro.plate("treatment", 4):
        b_treatment = numpyro.sample("b_treatment", dist.Normal(0, sigma_b))

    p = numpyro.deterministic(
        "p",
        jax.scipy.special.expit(a[actor] + g[block] + b_treatment[treatment]),
    )
    numpyro.sample("pulled_left", dist.Bernoulli(probs=p), obs=pulled_left)


training_data = {
    "actor": df["actor"].values - 1,
    "block": df["block"].values - 1,
    "treatment": df["treatment"].values,
    "pulled_left": df["pulled_left"].values,
}
mcmc_13_6 = MCMC(NUTS(model_13_6), num_warmup=500, num_samples=500, num_chains=4)
mcmc_13_6.run(jrngs[2], **training_data)
idata_13_6 = az.from_numpyro(mcmc_13_6)
pd.DataFrame(
    {
        "model 13.4": jnp.mean(mcmc_13_4.get_samples()["b_treatment"], 0),
        "model 13.6": jnp.mean(mcmc_13_6.get_samples()["b_treatment"], 0),
    }
)
```

```python
pd.DataFrame(
    {
        "model 13.4": jnp.mean(mcmc_13_4.get_samples()["b_treatment"], 0),
        "model 13.6": jnp.mean(mcmc_13_6.get_samples()["b_treatment"], 0),
    }
)
```

### Code 13.26

```python
def model_13_7():
    v = numpyro.sample("v", dist.Normal(0, 3))
    numpyro.sample("x", dist.Normal(0, jnp.exp(v)))


mcmc_13_7 = MCMC(NUTS(model_13_7), num_warmup=500, num_samples=500, num_chains=4)
mcmc_13_7.run(jrng)
mcmc_13_7.print_summary()
idata_13_7 = az.from_numpyro(mcmc_13_7)
az.plot_trace(idata_13_7, var_names=["v", "x"], compact=False, kind="rank_bars")
plt.tight_layout()
```

### Code 13.27

```python
def model_13_7_nc():
    v = numpyro.sample("v", dist.Normal(0, 3))
    z = numpyro.sample("z", dist.Normal(0, 1))
    numpyro.deterministic("x", z * jnp.exp(v))


mcmc_13_7_nc = MCMC(NUTS(model_13_7_nc), num_warmup=500, num_samples=500, num_chains=4)
mcmc_13_7_nc.run(jrng)
mcmc_13_7_nc.print_summary(exclude_deterministic=False)
idata_13_7_nc = az.from_numpyro(mcmc_13_7_nc)
az.plot_trace(idata_13_7_nc, compact=False, kind="rank_bars")
plt.tight_layout()
```

### Code 13.28

```python
mcmc_13_4b = MCMC(
    NUTS(model_13_4, target_accept_prob=0.9),
    num_warmup=500,
    num_samples=500,
    num_chains=4,
)
mcmc_13_4b.run(jrngs[0], **training_data)
mcmc_13_4b.print_summary()
```

### Code 13.29

```python
def model_13_4nc(treatment, actor, block, pulled_left):
    # hyper-priors
    a_bar = numpyro.sample("a_bar", dist.Normal(0, 1.5))
    sigma_a = numpyro.sample("sigma_a", dist.Exponential(1))
    sigma_g = numpyro.sample("sigma_g", dist.Exponential(1))

    # adaptive priors
    with numpyro.handlers.reparam(
        config={"a": numpyro.infer.reparam.LocScaleReparam(0)}
    ):
        with numpyro.plate("actor", 7):
            a = numpyro.sample("a", dist.Normal(a_bar, sigma_a))
    with numpyro.plate("block", 6):
        z_block = numpyro.sample("z_block", dist.Normal(0, 1))
        g = numpyro.deterministic("g", z_block * sigma_g)

    # old fashioned priors
    with numpyro.plate("treatment", 4):
        b_treatment = numpyro.sample("b_treatment", dist.Normal(0, 0.5))

    # likelihood
    p = numpyro.deterministic(
        "p", jax.scipy.special.expit(a[actor] + g[block] + b_treatment[treatment])
    )
    numpyro.sample("pulled_left", dist.Bernoulli(probs=p), obs=pulled_left)


training_data = {
    "treatment": df["treatment"].values,
    "actor": df["actor"].values - 1,
    "block": df["block"].values - 1,
    "pulled_left": df["pulled_left"].values,
}
mcmc_13_4nc = MCMC(NUTS(model_13_4nc), num_warmup=500, num_samples=500, num_chains=4)
mcmc_13_4nc.run(jrngs[0], **training_data)
mcmc_13_4nc.print_summary(exclude_deterministic=False)
idata_13_4nc = az.from_numpyro(mcmc_13_4nc)
az.plot_trace(
    idata_13_4nc,
    var_names=["a_bar", "sigma_a", "sigma_g", "a", "g", "b_treatment"],
    compact=False,
    kind="rank_bars",
)
plt.tight_layout()
```

### Code 13.30

```python
neff_c = {
    k: numpyro.diagnostics.effective_sample_size(v)
    for k, v in mcmc_13_4.get_samples(group_by_chain=True).items()
}
neff_nc = {
    k: numpyro.diagnostics.effective_sample_size(v)
    for k, v in mcmc_13_4nc.get_samples(group_by_chain=True).items()
}
par_names = []
keys_c = ["b_treatment", "a", "g", "a_bar", "sigma_a", "sigma_g"]
keys_nc = ["b_treatment", "a", "z_block", "a_bar", "sigma_a", "sigma_g"]
for k in keys_c:
    if jnp.ndim(neff_c[k]) == 0:
        par_names += [k]
    else:
        par_names += [k + "[{}]".format(i) for i in range(neff_c[k].size)]
neff_c = jnp.concatenate([neff_c[k].reshape(-1) for k in keys_c])
neff_nc = jnp.concatenate([neff_nc[k].reshape(-1) for k in keys_nc])
neff_table = pd.DataFrame(dict(neff_c=neff_c, neff_nc=neff_nc))
neff_table.index = par_names
neff_table.round()
```

### Code 13.31

```python
actors = jnp.repeat(1, 4)
treatments = jnp.arange(4)
blocks = jnp.repeat(0, 4)

posterior_predictive_13_4 = Predictive(model_13_4, mcmc_13_4.get_samples())
posterior_predictive_samples_13_4 = posterior_predictive_13_4(
    jrngs[0], actor=actors, treatment=treatments, block=blocks, pulled_left=None
)
p_mu = jnp.mean(posterior_predictive_samples_13_4["p"], axis=0)
p_hpdi = numpyro.diagnostics.hpdi(posterior_predictive_samples_13_4["p"], prob=0.89)
plt.plot(treatments, p_mu, "o", c="k")
plt.fill_between(treatments, p_hpdi[0, :], p_hpdi[1, :], color="k", alpha=0.2)
```

### Code 13.32

```python
posterior_samples_13_4 = mcmc_13_4.get_samples()
{k: v.reshape(-1)[:5] for k, v in posterior_samples_13_4.items()}
```

### Code 13.33

```python
az.plot_kde(posterior_samples_13_4["a"][:, 4])
```

### Code 13.34

```python
def p_link(treatment, actor, block):
    return jax.scipy.special.expit(
        posterior_samples_13_4["a"][:, actor]
        + posterior_samples_13_4["g"][:, block]
        + posterior_samples_13_4["b_treatment"][:, treatment]
    )
```

### Code 13.35

```python
p_raw = p_link(treatments, actors, blocks)
p_mu = jnp.mean(p_raw, axis=0)
p_hpdi = numpyro.diagnostics.hpdi(p_raw, prob=0.89)
```

### Code 13.36

```python
def p_link_abar(treatment):
    return jax.scipy.special.expit(
        posterior_samples_13_4["a_bar"][:, None]
        + posterior_samples_13_4["b_treatment"][:, treatment]
    )
```

### Code 13.37

```python
treatments = jnp.arange(4)
p_raw = p_link_abar(treatments)
p_mu = jnp.mean(p_raw, axis=0)
p_hpdi = numpyro.diagnostics.hpdi(p_raw, prob=0.89)

plt.subplot(xlabel="treatment", ylabel="proportion pulled left", xticks=treatments)
plt.plot(treatments, p_mu, c="k")
plt.fill_between(treatments, p_hpdi[0, :], p_hpdi[1, :], color="k", alpha=0.2)
```

### Code 13.38

```python
a_sim = dist.Normal(
    posterior_samples_13_4["a_bar"], posterior_samples_13_4["sigma_a"]
).sample(jrngs[0])


def p_link_asim(treatment):
    return jax.scipy.special.expit(
        a_sim[:, None] + posterior_samples_13_4["b_treatment"][:, treatment]
    )


p_raw = p_link_asim(treatments)
p_mu = jnp.mean(p_raw, axis=0)
p_hpdi = numpyro.diagnostics.hpdi(p_raw, prob=0.89)

plt.subplot(xlabel="treatment", ylabel="proportion pulled left", xticks=treatments)
plt.plot(treatments, p_mu, c="k")
plt.fill_between(treatments, p_hpdi[0, :], p_hpdi[1, :], color="k", alpha=0.2)
```

### Code 13.39

```python
plt.subplot(xlabel="treatment", ylabel="proportion pulled left", xticks=treatments)
for i in range(100):
    plt.plot(treatments, p_raw[i], "k", alpha=0.2)
```

<!-- #region -->
## Easy
### 13E1

(a) 


### 13E2

$$
\begin{align*}
y_i & \sim Binom(1, p_i) \\
logit(p_i) = \alpha_{group[i]} + \beta x_i \\
\alpha_{group[i]} & \sim \mathcal{N}(\bar{\alpha}, \sigma) \\
\beta & \sim \mathcal{N}(0, 1) \\
\bar{\alpha} & \sim \mathcal{N}(0, 10) \\
\sigma & \sim Exp(1)
\end{align*}
$$

### 13E3

$$
\begin{align*}
y_i & \sim \mathcal{N}(\mu_i, \sigma)  \\
\mu_i & =  \alpha_{group[i]} + \beta x_i \\
\alpha_{group[i]} & \sim \mathcal{N}(\bar{\alpha}, \sigma_{\bar{\alpha}}) \\
\beta & \sim \mathcal{N}(0, 1) \\
\bar{\alpha} & \sim \mathcal{N}(0, 10) \\
\sigma_{\bar{\alpha}} & \sim Exp(1) \\
\sigma & \sim HalfCauchy(0, 2)
\end{align*}
$$

### 13E4

$$
\begin{align*}
fishcaught & \sim Poisson(\lambda_i) \\
log(\lambda_i) & = \alpha_{country[i]} + \gamma_{agegroup[i]} + \beta Persons + \log(\tau_i) \\
\alpha_{country[i]} & \sim \mathcal{N}(\bar{\alpha}, \sigma_{\bar{\alpha}}) \\
\gamma_{agegroup[i]} & \sim \mathcal{N}(0, \sigma_{\bar{\gamma}}) \\
\bar{\alpha} & \sim \mathcal{N}(0, 10) \\
\sigma_{\bar{\alpha}} & \sim Exp(1) \\
\sigma_{\bar{\gamma}} & \sim Exp(1)
\end{align*}
$$
<!-- #endregion -->

## Medium
### 13M1

```python
df = pd.read_csv("../data/reedfrogs.csv", sep=";")
df["pred"] = [1 if x == "pred" else 0 for x in df["pred"]]
df["size"] = jnp.where(df["size"].values == "big", 1, 0)
df["tank"] = jnp.arange(df.shape[0])
df
```

```python
def model_factory(has_pred: bool, has_size: bool, has_interaction: bool):
    def model(N, pred, size, tank, surv):
        a_bar = numpyro.sample("a_bar", dist.Normal(0, 10))
        sigma_a = numpyro.sample("sigma_a", dist.Exponential(1))

        with numpyro.plate("tank", 48):
            a = numpyro.sample("a", dist.Normal(a_bar, sigma_a))
        logit_p = a[tank]
        if has_pred:
            b_pred = numpyro.sample("b_pred", dist.Normal(0, 1))
            logit_p += b_pred * pred
        if has_size:
            b_size = numpyro.sample("b_size", dist.Normal(0, 1))
            logit_p += b_size * size
        if has_interaction:
            b_interaction = numpyro.sample("b_interaction", dist.Normal(0, 1))
            logit_p += b_interaction * size * pred

        numpyro.deterministic("p", jax.scipy.special.expit(logit_p))
        numpyro.sample("surv", dist.Binomial(total_count=N, logits=logit_p), obs=surv)

    return model


mcmc_13m1_none = MCMC(
    NUTS(model_factory(has_pred=False, has_size=False, has_interaction=False)),
    num_warmup=500,
    num_samples=500,
    num_chains=4,
)
mcmc_13m1_pred = MCMC(
    NUTS(model_factory(has_pred=True, has_size=False, has_interaction=False)),
    num_warmup=500,
    num_samples=500,
    num_chains=4,
)
mcmc_13m1_size = MCMC(
    NUTS(model_factory(has_pred=False, has_size=True, has_interaction=False)),
    num_warmup=500,
    num_samples=500,
    num_chains=4,
)
mcmc_13m1_both = MCMC(
    NUTS(model_factory(has_pred=True, has_size=True, has_interaction=False)),
    num_warmup=500,
    num_samples=500,
    num_chains=4,
)
mcmc_13m1_interaction = MCMC(
    NUTS(model_factory(has_pred=True, has_size=True, has_interaction=True)),
    num_warmup=500,
    num_samples=500,
    num_chains=4,
)
mcmcs = {
    "none": mcmc_13m1_none,
    "pred": mcmc_13m1_pred,
    "size": mcmc_13m1_size,
    "both": mcmc_13m1_both,
    "interaction": mcmc_13m1_interaction,
}
training_data = {
    "N": df["density"].values,
    "pred": df["pred"].values,
    "size": df["size"].values,
    "tank": df["tank"].values,
    "surv": df["surv"].values,
}
for name, mcmc in mcmcs.items():
    mcmc.run(jrngs[0], **training_data)
```

```python
pd.Series(
    {
        key: f"{jnp.mean(mcmc.get_samples()['sigma_a'], axis=0):.2f}"
        for key, mcmc in mcmcs.items()
    }
)
```

Adding predictors reduce the need for `sigma_a` to capture variation accross tanks.
In theory, if we had all the predictors, we wouldn't need multi level model.


### 13M2

```python
az.compare({key: az.from_numpyro(mcmc) for key, mcmc in mcmcs.items()}, ic="waic")
```

```python
pd.DataFrame(
    {
        key: [
            f"{jnp.mean(mcmc.get_samples().get(var, jnp.repeat(jnp.nan, 10)), axis=0):.2f}"
            for var in ["b_pred", "b_size", "b_interaction"]
        ]
        for key, mcmc in mcmcs.items()
    },
    index=["b_pred", "b_size", "b_interaction"],
)
```

Size doesn't matter


### 13M3

```python
def model_13m3(N, tank, surv):
    a_bar = numpyro.sample("a_bar", dist.Normal(0, 1))
    sigma_a = numpyro.sample("sigma_a", dist.Exponential(1))

    with numpyro.plate("tank", 48):
        a = numpyro.sample("a", dist.Cauchy(a_bar, sigma_a))
    logit_p = a[tank]

    numpyro.deterministic("p", jax.scipy.special.expit(logit_p))
    numpyro.sample("surv", dist.Binomial(total_count=N, logits=logit_p), obs=surv)


training_data = {
    "N": df["density"].values,
    "tank": df["tank"].values,
    "surv": df["surv"].values,
}
mcmc_13m3 = MCMC(NUTS(model_13m3), num_warmup=500, num_samples=500, num_chains=4)
mcmc_13m3.run(jrngs[0], **training_data)
```

```python
result = pd.DataFrame(
    {
        "gaussian": jnp.mean(mcmc_13m1_none.get_samples()["a"], axis=0),
        "cauchy": jnp.mean(mcmc_13m3.get_samples()["a"], axis=0),
    }
)
result["diff"] = result.diff(axis=1)["cauchy"]
result["diff"].plot(kind="bar")
```

Most of the time, the two are close, but the cauchy prior is comfortable with more extreme values.
However, since we're using a logit link, prob doesn't make a huge difference anyways: in probability space, we converge to 1. 


### 13M4

```python
df = pd.read_csv("../data/chimpanzees.csv", sep=";")
df["treatment"] = df["prosoc_left"] + 2 * df["condition"]
df
```

```python
def model_13m4(treatment, actor, block, pulled_left):
    # hyper-priors
    a_bar = numpyro.sample("a_bar", dist.Normal(0, 1.5))
    sigma_a = numpyro.sample("sigma_a", dist.Exponential(1))
    gamma_bar = numpyro.sample("gamma_bar", dist.Normal(0, 1.5))
    sigma_g = numpyro.sample("sigma_g", dist.Exponential(1))

    # adaptive priors
    with numpyro.handlers.reparam(
        config={"a": numpyro.infer.reparam.LocScaleReparam(0)}
    ):
        with numpyro.plate("actor", 7):
            a = numpyro.sample("a", dist.Normal(a_bar, sigma_a))
    with numpyro.plate("block", 6):
        z_block = numpyro.sample("z_block", dist.Normal(0, 1))
        g = numpyro.deterministic("g", gamma_bar + z_block * sigma_g)

    # old fashioned priors
    with numpyro.plate("treatment", 4):
        b_treatment = numpyro.sample("b_treatment", dist.Normal(0, 0.5))

    # likelihood
    p = numpyro.deterministic(
        "p", jax.scipy.special.expit(a[actor] + g[block] + b_treatment[treatment])
    )
    numpyro.sample("pulled_left", dist.Bernoulli(probs=p), obs=pulled_left)


training_data = {
    "treatment": df["treatment"].values,
    "actor": df["actor"].values - 1,
    "block": df["block"].values - 1,
    "pulled_left": df["pulled_left"].values,
}
mcmc_13m4 = MCMC(NUTS(model_13m4), num_warmup=500, num_samples=500, num_chains=4)
mcmc_13m4.run(jrngs[0], **training_data)
idata_13m4 = az.from_numpyro(mcmc_13m4)
az.plot_trace(
    idata_13m4,
    var_names=["a_bar", "gamma_bar"],
    compact=False,
    kind="rank_bars",
)
plt.tight_layout()
```

```python
posterior_sample_13m4 = mcmc_13m4.get_samples()
plt.plot(posterior_sample_13m4["a"][:, 0], posterior_sample_13m4["g"][:, 0], "o")
```

alpha and gammaa bar can't be separated, we can only know their sum. 


## Hard
### 13H1

```python
df = pd.read_csv("../data/bangladesh.csv", sep=";")
df
```

```python
jnp.unique(jnp.diff(pd.unique(df["district"])))
```

```python
df["district_id"] = pd.Categorical(df["district"]).codes
df
```

```python
def model_13h1a(district_id, use_contraception):

    with numpyro.plate("district", 60):
        a = numpyro.sample("a", dist.Normal(0, 10))
    logit_p = a[district_id]
    numpyro.deterministic("p", jax.scipy.special.expit(logit_p))
    numpyro.sample(
        "use_contraception", dist.Bernoulli(logits=logit_p), obs=use_contraception
    )


def model_13h1b(district_id, use_contraception):
    a_bar = numpyro.sample("a_bar", dist.Normal(0, 10))
    sigma_a = numpyro.sample("sigma_a", dist.Exponential(1))

    with numpyro.plate("district", 60):
        a = numpyro.sample("a", dist.Normal(a_bar, sigma_a))

    logit_p = a[district_id]
    numpyro.deterministic("p", jax.scipy.special.expit(logit_p))
    numpyro.sample(
        "use_contraception", dist.Bernoulli(logits=logit_p), obs=use_contraception
    )


mcmc_13h1a = MCMC(NUTS(model_13h1a), num_warmup=500, num_samples=500, num_chains=4)
mcmc_13h1b = MCMC(NUTS(model_13h1b), num_warmup=500, num_samples=500, num_chains=4)
training_data = {
    "district_id": df["district_id"].values,
    "use_contraception": df["use.contraception"].values,
}
mcmc_13h1a.run(jrng, **training_data)
mcmc_13h1b.run(jrng, **training_data)
idata_13h1a = az.from_numpyro(mcmc_13h1a)
idata_13h1b = az.from_numpyro(mcmc_13h1b)
```

```python
district_ids = jnp.arange(60)
posterior_predictive_samples_13h1a = Predictive(model_13h1a, mcmc_13h1a.get_samples())(
    jrng, district_id=district_ids, use_contraception=None
)
posterior_predictive_samples_13h1b = Predictive(model_13h1b, mcmc_13h1b.get_samples())(
    jrng, district_id=district_ids, use_contraception=None
)

p_mu_13h1a = jnp.mean(posterior_predictive_samples_13h1a["p"], axis=0)
p_hpdi_13h1a = numpyro.diagnostics.hpdi(
    posterior_predictive_samples_13h1a["p"], prob=0.89
)
p_mu_13h1b = jnp.mean(posterior_predictive_samples_13h1b["p"], axis=0)
p_hpdi_13h1b = numpyro.diagnostics.hpdi(
    posterior_predictive_samples_13h1b["p"], prob=0.89
)
plt.subplot(xlabel="district", ylabel="proportion use contraception")
plt.plot(
    district_ids,
    df.groupby("district_id")["use.contraception"].mean().sort_index(),
    "x",
    c="k",
    label="observed",
)
plt.plot(district_ids, p_mu_13h1a, "o", c="k", label="no pooling")
plt.fill_between(
    district_ids, p_hpdi_13h1a[0, :], p_hpdi_13h1a[1, :], color="k", alpha=0.2
)
plt.plot(district_ids, p_mu_13h1b, "o", c="b", label="partial pooling")
plt.fill_between(
    district_ids, p_hpdi_13h1b[0, :], p_hpdi_13h1b[1, :], color="b", alpha=0.2
)
plt.legend()
```

Partial pooling is more regularizing, pulling values towards the mean. Most extreme cases of disagrement occur when the observed value is extreme (near 1 or near 0). Partial pooling model regularizes harder.


### 13H2

```python
df = pd.read_csv("../data/Trolley.csv", sep=";")
df["pid"] = pd.Categorical(df["id"]).codes
df
```

```python
def model_13h2a(action, contact, intention, response):

    cutpoints = numpyro.sample(
        "cutpoints",
        dist.TransformedDistribution(
            dist.Normal(0, 1.5).expand([6]), dist.transforms.OrderedTransform()
        ),
    )
    b_action = numpyro.sample("b_action", dist.Normal(0, 1))
    b_contact = numpyro.sample("b_contact", dist.Normal(0, 1))
    b_intention = numpyro.sample("b_intention", dist.Normal(0, 1))
    phi = numpyro.deterministic(
        "phi", b_action * action + b_contact * contact + b_intention * intention
    )
    response = numpyro.sample(
        "response", dist.OrderedLogistic(phi, cutpoints), obs=response
    )
    return response


def model_13h2b(pid, action, contact, intention, response):

    cutpoints = numpyro.sample(
        "cutpoints",
        dist.TransformedDistribution(
            dist.Normal(0, 1.5).expand([6]), dist.transforms.OrderedTransform()
        ),
    )
    sigma_a = numpyro.sample("sigma_a", dist.Exponential(1))
    with numpyro.plate("pid", 331):
        z_a = numpyro.sample("z_a", dist.Normal(0, 1))
        a = numpyro.deterministic("a", z_a * sigma_a)
    b_action = numpyro.sample("b_action", dist.Normal(0, 1))
    b_contact = numpyro.sample("b_contact", dist.Normal(0, 1))
    b_intention = numpyro.sample("b_intention", dist.Normal(0, 1))
    phi = numpyro.deterministic(
        "phi",
        a[pid] + b_action * action + b_contact * contact + b_intention * intention,
    )
    response = numpyro.sample(
        "response", dist.OrderedLogistic(phi, cutpoints), obs=response
    )
    return response


mcmc_13h2a = MCMC(NUTS(model_13h2a), num_warmup=500, num_samples=500, num_chains=4)
mcmc_13h2b = MCMC(NUTS(model_13h2b), num_warmup=500, num_samples=500, num_chains=4)
training_data_a = {
    "action": df["action"].values,
    "contact": df["contact"].values,
    "intention": df["intention"].values,
    "response": df["response"].values - 1,
}
training_data_b = {
    "pid": df["pid"].values,
    "action": df["action"].values,
    "contact": df["contact"].values,
    "intention": df["intention"].values,
    "response": df["response"].values - 1,
}
mcmc_13h2a.run(jrng, **training_data_a)
mcmc_13h2b.run(jrng, **training_data_b)
idata_13h2a = az.from_numpyro(mcmc_13h2a)
idata_13h2b = az.from_numpyro(mcmc_13h2b)
az.plot_compare(
    az.compare({"mcmc_13h2a": idata_13h2a, "mcmc_13h2b": idata_13h2b}, scale="deviance")
)
```

There is a lot of variation between individuals, so modelling it greatly improves prediction


### 13H3

```python
df = pd.read_csv("../data/Trolley.csv", sep=";")
df["pid"] = pd.Categorical(df["id"]).codes
df["story_id"] = pd.Categorical(df["story"]).codes
df
```

```python
def model_13h3(pid, story_id, action, contact, intention, response):

    cutpoints = numpyro.sample(
        "cutpoints",
        dist.TransformedDistribution(
            dist.Normal(0, 1.5).expand([6]), dist.transforms.OrderedTransform()
        ),
    )
    sigma_a = numpyro.sample("sigma_a", dist.Exponential(1))
    with numpyro.plate("pid", 331):
        z_a = numpyro.sample("z_a", dist.Normal(0, 1))
        a = numpyro.deterministic("a", z_a * sigma_a)
    sigma_story = numpyro.sample("sigma_story", dist.Exponential(1))
    with numpyro.plate("story_id", 12):
        z_story = numpyro.sample("z_story", dist.Normal(0, 1))
        a_story = numpyro.deterministic("story", z_story * sigma_story)
    b_action = numpyro.sample("b_action", dist.Normal(0, 1))
    b_contact = numpyro.sample("b_contact", dist.Normal(0, 1))
    b_intention = numpyro.sample("b_intention", dist.Normal(0, 1))
    phi = numpyro.deterministic(
        "phi",
        a[pid]
        + a_story[story_id]
        + b_action * action
        + b_contact * contact
        + b_intention * intention,
    )
    response = numpyro.sample(
        "response", dist.OrderedLogistic(phi, cutpoints), obs=response
    )
    return response


mcmc_13h3 = MCMC(NUTS(model_13h3), num_warmup=500, num_samples=500, num_chains=4)
training_data = {
    "pid": df["pid"].values,
    "story_id": df["story_id"].values,
    "action": df["action"].values,
    "contact": df["contact"].values,
    "intention": df["intention"].values,
    "response": df["response"].values - 1,
}
mcmc_13h3.run(jrng, **training_data)
idata_13h3 = az.from_numpyro(mcmc_13h3)
```

```python
az.plot_compare(
    az.compare(
        {"mcmc_13h2b": idata_13h2b, "mcmc_13h3": idata_13h3},
        scale="deviance",
        ic="waic",
    )
)
```

Variation among stories is meaningful, and help us explain variability further.

```python

```
