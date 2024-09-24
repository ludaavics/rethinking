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

# Chapter 9: Markov Chain Monte Carlo

```python
%load_ext jupyter_black

import inspect
import os

import arviz as az
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpyro
import numpyro.distributions as dist
import pandas as pd

from numpyro.infer import MCMC, NUTS, SVI
from numpyro.infer.autoguide import AutoLaplaceApproximation

numpyro.set_host_device_count(os.cpu_count())

seed = 84735
jrng = jax.random.key(seed)
_, jrng2 = jax.random.split(jrng)
_, jrng3 = jax.random.split(jrng2)
_, jrng4 = jax.random.split(jrng3)
_, jrng5 = jax.random.split(jrng4)
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
### Code 9.1

```python
# Semi-naive implementation, runs in ~2 minutes

num_weeks = 0  # should be 100_000 to actually run
positions = [0] * num_weeks
coin_flips = dist.Bernoulli(probs=0.5).sample(jrng, sample_shape=(num_weeks,))
proposed_deltas = jnp.where(coin_flips, 1, -1)
proposal_evaluations = dist.Uniform(low=0, high=1).sample(
    jrng2, sample_shape=(num_weeks,)
)

current = 9
for i in range(num_weeks):
    positions[i] = current

    # flip a coin to get proposal
    proposal = current + proposed_deltas[i]
    if proposal < 0:
        proposal = 10
    elif proposal > 9:
        proposal = 9

    # decide whether to accept proposal
    probability_move = proposal / current
    current = proposal if proposal_evaluations[i] <= probability_move else current
```

```python
# Using jax, runs in ~1 secs
num_weeks = int(1e5)
positions = jnp.repeat(0, num_weeks)
current = 10


def metropolis_sample(i, val):
    positions, current = val

    positions = positions.at[i].set(current)

    # flip a coin to get proposal
    proposal = (
        current + 2 * dist.Bernoulli(probs=0.5).sample(jax.random.fold_in(jrng, i)) - 1
    )
    proposal = jnp.where(proposal > 10, 1, proposal)
    proposal = jnp.where(proposal < 1, 10, proposal)

    # decide whether to accept proposal
    probability_move = proposal / current
    evaluation = dist.Uniform().sample(jax.random.fold_in(jrng2, i))
    current = jnp.where(evaluation < probability_move, proposal, current)

    return (positions, current)


positions, current = jax.lax.fori_loop(
    0,
    num_weeks,
    metropolis_sample,
    (positions, current),
)
```

### Code 9.2

```python
plt.plot(range(1, 101), positions[:100], "o", mfc="none")
plt.show()
```

### Code 9.3

```python
plt.hist(jnp.asarray(positions), bins=range(1, 12), rwidth=0.1, align="left")
plt.show()
```

### Code 9.4

```python
T = 1_000
for D in [1, 10, 100]:
    Y = dist.MultivariateNormal(loc=0, covariance_matrix=jnp.identity(D)).sample(
        jrng, sample_shape=(T,)
    )
    rad_dist = jnp.sqrt(jnp.sum(Y**2, axis=1))
    az.plot_kde(rad_dist, bw=0.18)
plt.show()
```

### Code 9.5

```python
def U(q, a=0, b=1, k=0, d=1):
    "Return the negative log prob of the parameters q and the data."
    muy = q[0]
    mux = q[1]
    logprob_y = jnp.sum(dist.Normal(muy, 1).log_prob(y))
    logprob_x = jnp.sum(dist.Normal(mux, 1).log_prob(x))
    logprob_muy = dist.Normal(a, b).log_prob(muy)
    logprob_mux = dist.Normal(k, d).log_prob(mux)
    U = logprob_y + logprob_x + logprob_muy + logprob_mux
    return -U
```

### Code 9.6

```python
def U_gradient(q, a=0, b=1, k=0, d=1):
    "Gradient of U w.r.t parameters q"
    muy = q[0]
    mux = q[1]
    G1 = jnp.sum(y - muy) + (a - muy) / b**2  # dU/dmuy
    G2 = jnp.sum(x - mux) + (k - mux) / b**2  # dU/dmux
    return jnp.stack([-G1, -G2])  # negative bc energy is neg-log-prob


# test data
with numpyro.handlers.seed(rng_seed=seed):
    y = numpyro.sample("y", dist.Normal().expand([50]))
    x = numpyro.sample("x", dist.Normal().expand([50]))
    x = scale(x)
    y = scale(y)
```

### Code 9.7

```python
def HMC2(U, grad_U, epsilon, L, current_q, jrng):
    q = current_q
    # random flick - p is momentum
    p = dist.Normal(0, 1).sample(jax.random.fold_in(jrng, 0), (q.shape[0],))
    current_p = p
    # Make a half step for momentum at the beginning
    p = p - epsilon * grad_U(q) / 2
    # initialize bookkeeping - saves trajectory
    qtraj = jnp.full((L + 1, q.shape[0]), jnp.nan)
    ptraj = qtraj
    qtraj = qtraj.at[0].set(current_q)
    ptraj = ptraj.at[0].set(p)

    # Alternate full steps for position and momentum
    for i in range(L):
        q = q + epsilon * p  # Full step for the position
        # Make a full step for the momentum, except at end of trajectory
        if i != (L - 1):
            p = p - epsilon * grad_U(q)
            ptraj = ptraj.at[i + 1].set(p)
        qtraj = qtraj.at[i + 1].set(q)

    # Make a half step for momentum at the end
    p = p - epsilon * grad_U(q) / 2
    ptraj = ptraj.at[L].set(p)
    # Negate momentum at end of trajectory to make the proposal symmetric
    p = -p
    # Evaluate potential and kinetic energies at start and end of trajectory
    current_U = U(current_q)
    current_K = jnp.sum(current_p**2) / 2
    proposed_U = U(q)
    proposed_K = jnp.sum(p**2) / 2
    # Accept or reject the state at end of trajectory, returning either
    # the position at the end of the trajectory or the initial position
    accept = 0
    runif = dist.Uniform().sample(jax.random.fold_in(jrng, 1))
    if runif < jnp.exp(current_U - proposed_U + current_K - proposed_K):
        new_q = q  # accept
        accept = 1
    else:
        new_q = current_q  # reject
    return {
        "q": new_q,
        "traj": qtraj,
        "ptraj": ptraj,
        "accept": accept,
        "dH": proposed_U + proposed_K - (current_U + current_K),
    }


Q = {}
Q["q"] = jnp.array([-0.1, 0.2])
pr = 0.31
plt.subplot(ylabel="muy", xlabel="mux", xlim=(-pr, pr), ylim=(-pr, pr))
step = 0.03
L = 11  # 0.03/28 for U-turns --- 11 for working example
n_samples = 4
path_col = (0, 0, 0, 0.5)
for r in 0.075 * jnp.arange(2, 6):
    plt.gca().add_artist(plt.Circle((0, 0), r, alpha=0.2, fill=False))
plt.scatter(Q["q"][0], Q["q"][1], c="k", marker="x", zorder=4)
for i in range(n_samples):
    Q = HMC2(U, U_gradient, step, L, Q["q"], jax.random.fold_in(jrng, i))
    if n_samples < 10:
        for j in range(L):
            K0 = jnp.sum(Q["ptraj"][j] ** 2) / 2
            plt.plot(
                Q["traj"][j : j + 2, 0],
                Q["traj"][j : j + 2, 1],
                c=path_col,
                lw=1 + 2 * K0,
            )
        plt.scatter(Q["traj"][:, 0], Q["traj"][:, 1], c="white", s=5, zorder=3)
        # for fancy arrows
        dx = Q["traj"][L, 0] - Q["traj"][L - 1, 0]
        dy = Q["traj"][L, 1] - Q["traj"][L - 1, 1]
        d = jnp.sqrt(dx**2 + dy**2)
        plt.annotate(
            "",
            (Q["traj"][L - 1, 0], Q["traj"][L - 1, 1]),
            (Q["traj"][L, 0], Q["traj"][L, 1]),
            arrowprops={"arrowstyle": "<-"},
        )
        plt.annotate(
            str(i + 1),
            (Q["traj"][L, 0], Q["traj"][L, 1]),
            xytext=(3, 3),
            textcoords="offset points",
        )
    plt.scatter(
        Q["traj"][L + 1, 0],
        Q["traj"][L + 1, 1],
        c=("red" if jnp.abs(Q["dH"]) > 0.1 else "black"),
        zorder=4,
    )
```

### Code 9.8

```python
source_HMC2 = inspect.getsourcelines(HMC2)
print("".join("".join(source_HMC2[0]).split("\n\n")[0]))
```

### Code 9.9

```python
print("".join("".join(source_HMC2[0]).split("\n\n")[1]))
```

### Code 9.10

```python
print("".join("".join(source_HMC2[0]).split("\n\n")[2]))
```

### Code 9.11

```python
df = pd.read_csv("../data/rugged.csv", sep=";")
df["log_gdp"] = jnp.log(df["rgdppc_2000"].values)
df = df.dropna(subset=["rgdppc_2000"])
df["log_gdp_std"] = df.log_gdp / df.log_gdp.mean()
df["rugged_std"] = df.rugged / df.rugged.max()
df["cid"] = jnp.where(df.cont_africa.values == 1, 0, 1)
```

### Code 9.12

```python
def model_8_3(rugged, cid, log_gdp):
    with numpyro.plate(name="continent", size=2):
        a = numpyro.sample("a", dist.Normal(1, 0.1))
        b = numpyro.sample("b", dist.Normal(0, 0.3))
    mu = numpyro.deterministic("mu", a[cid] + b[cid] * (rugged - 0.215))
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
numpyro.diagnostics.print_summary(
    prune_return_sites(posterior_samples_8_3, depth=2, exclude=["mu"]),
    group_by_chain=False,
)
```

### Code 9.13

```python
_ = {k: v[:5] for k, v in df.items() if k in ["log_gdp_std", "rugged_std", "cid"]}
_
```

### Code 9.14

```python
model_9_1 = model_8_3

mcmc_9_1 = MCMC(NUTS(model_9_1), num_warmup=500, num_samples=500, num_chains=1)
mcmc_9_1.run(
    jrng,
    rugged=df["rugged_std"].values,
    cid=df["cid"].values,
    log_gdp=df["log_gdp_std"].values,
)
```

### Code 9.15

```python
mcmc_9_1.print_summary(prob=0.89)
```

### Code 9.16

```python
mcmc_9_1 = MCMC(
    NUTS(model_9_1),
    num_warmup=500,
    num_samples=500,
    num_chains=4,
    chain_method="parallel",
)
mcmc_9_1.run(
    jrng,
    rugged=df["rugged_std"].values,
    cid=df["cid"].values,
    log_gdp=df["log_gdp_std"].values,
)
```

### Code 9.17

```python
print("".join(inspect.getsourcelines(mcmc_9_1.sampler.model)[0]))
```

### Code 9.18

```python
mcmc_9_1.print_summary(0.89)
```

### Code 9.19

```python
idata_9_1 = az.from_numpyro(mcmc_9_1)
az.plot_pair(idata_9_1, var_names=["a", "b", "sigma"])
```

### Code 9.20

```python
az.plot_trace(idata_9_1, var_names=["a", "b", "sigma"], kind="trace")
plt.tight_layout()
```

```python
az.plot_rank(idata_9_1, var_names=["a", "b", "sigma"])
plt.tight_layout()
```

### Code 9.22

```python
y = jnp.array([-1, 1])


def model_9_2(y):
    a = numpyro.sample("a", dist.Normal(0, 1_000))
    mu = a
    sigma = numpyro.sample("sigma", dist.Exponential(0.0001))
    y = numpyro.sample("y", dist.Normal(mu, sigma), obs=y)
    return y


mcmc_9_2 = MCMC(NUTS(model_9_2), num_warmup=500, num_samples=500, num_chains=4)
mcmc_9_2.run(jrng, y=y)
idata_9_2 = az.from_numpyro(mcmc_9_2)
```

### Code 9.23

```python
mcmc_9_2.print_summary()
```

```python
az.plot_pair(idata_9_2, var_names=["a", "sigma"], divergences=True)
```

```python
az.plot_trace(idata_9_2, var_names=["a", "sigma"])
```

```python
az.plot_rank(idata_9_2, var_names=["a", "sigma"])
plt.tight_layout()
```

### Code 9.24

```python
y = jnp.array([-1, 1])


def model_9_3(y):
    a = numpyro.sample("a", dist.Normal(0, 1))
    mu = a
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    y = numpyro.sample("y", dist.Normal(mu, sigma), obs=y)
    return y


mcmc_9_3 = MCMC(NUTS(model_9_3), num_warmup=500, num_samples=500, num_chains=4)
mcmc_9_3.run(jrng, y=y)
idata_9_3 = az.from_numpyro(mcmc_9_3)
```

```python
mcmc_9_3.print_summary()
```

```python
az.plot_pair(idata_9_3, var_names=["a", "sigma"], divergences=True)
```

```python
az.plot_trace(idata_9_3, var_names=["a", "sigma"])
plt.tight_layout()
```

```python
az.plot_rank(idata_9_3, var_names=["a", "sigma"])
plt.tight_layout()
```

### Code 9.25

```python
y = dist.Normal(0, 1).sample(jrng, sample_shape=(100,))
```

### Code 9.26

```python
def model_9_4(y):
    a = numpyro.sample("a", dist.Normal(0, 1000).expand((2,)))
    mu = numpyro.deterministic("mu", jnp.sum(a))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    y = numpyro.sample("y", dist.Normal(mu, sigma), obs=y)
    return y


mcmc_9_4 = MCMC(NUTS(model_9_4), num_warmup=500, num_samples=500, num_chains=4)
mcmc_9_4.run(jrng, y=y)
idata_9_4 = az.from_numpyro(mcmc_9_4)
mcmc_9_4.print_summary()
```

```python
az.plot_pair(idata_9_4, var_names=["a", "sigma"], divergences=True)
```

```python
az.plot_trace(idata_9_4, var_names=["a", "sigma"])
plt.tight_layout()
```

```python
az.plot_rank(idata_9_4, var_names=["a", "sigma"])
plt.tight_layout()
```

### Code 9.27

```python
def model_9_5(y):
    a = numpyro.sample("a", dist.Normal(0, 10).expand((2,)))
    mu = numpyro.deterministic("mu", jnp.sum(a))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    y = numpyro.sample("y", dist.Normal(mu, sigma), obs=y)
    return y


mcmc_9_5 = MCMC(NUTS(model_9_5), num_warmup=500, num_samples=500, num_chains=4)
mcmc_9_5.run(jrng, y=y)
idata_9_5 = az.from_numpyro(mcmc_9_5)
mcmc_9_5.print_summary()
```

```python
az.plot_pair(idata_9_5, var_names=["a", "sigma"], divergences=True)
```

```python
az.plot_trace(idata_9_5, var_names=["a", "sigma"])
plt.tight_layout()
```

```python
az.plot_rank(idata_9_5, var_names=["a", "sigma"])
plt.tight_layout()
```

## Easy
### 9E1

The proposal distribution must be symmetric

### 9E2

Instead of random, symmetrical proposition, it makes propositions based on an analytical approximation of the posterior distribution, make more proposals towards high density regions, thus increasing acceptance rate.
The limitations are that are force to use priors for which we can derive posterior analytically (conjugate priors). And, because proposals are still made one parameter at a time, it will get stuck in small regions of the posterior when the posterior has either highly correlated params or high dimension.

### 9E3

HMC cannot handle discrete parameters, because the step that generates proposals explores the parameter space continuously: it's a particle that glides over the posterior surface.

### 9E4

The number of effective sample is an estimate of the number of samples we have drawn, if our draws were truly independant.

### 9E5

Rhat will approach 1.

### 9E6

```python
plt.plot(
    jnp.arange(500),
    dist.Normal(0, 1).sample(jrng, sample_shape=(500,)),
    jnp.arange(500),
    dist.Normal(0, 1).sample(jrng2, sample_shape=(500,)),
)
```

```python
plt.plot(
    jnp.arange(500),
    jnp.cumsum(dist.Normal(0, 1).sample(jrng, sample_shape=(500,))),
    jnp.arange(500),
    jnp.cumsum(dist.Normal(0, 1).sample(jrng2, sample_shape=(500,))),
)
```

## Medium
### 9M1

```python
df = pd.read_csv("../data/rugged.csv", sep=";")
df["log_gdp"] = jnp.log(df["rgdppc_2000"].values)
df = df.dropna(subset=["rgdppc_2000"])
df["log_gdp_std"] = df.log_gdp / df.log_gdp.mean()
df["rugged_std"] = df.rugged / df.rugged.max()
df["cid"] = jnp.where(df.cont_africa.values == 1, 0, 1)


def model_9m1a(rugged, cid, log_gdp):
    with numpyro.plate(name="continent", size=2):
        a = numpyro.sample("a", dist.Normal(1, 0.1))
        b = numpyro.sample("b", dist.Normal(0, 0.3))
    mu = numpyro.deterministic("mu", a[cid] + b[cid] * (rugged - 0.215))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    log_gdp = numpyro.sample("log_gdp", dist.Normal(mu, sigma), obs=log_gdp)
    return log_gdp


def model_9m1b(rugged, cid, log_gdp):
    with numpyro.plate(name="continent", size=2):
        a = numpyro.sample("a", dist.Normal(1, 0.1))
        b = numpyro.sample("b", dist.Normal(0, 0.3))
    mu = numpyro.deterministic("mu", a[cid] + b[cid] * (rugged - 0.215))
    sigma = numpyro.sample("sigma", dist.Uniform(0, 10))
    log_gdp = numpyro.sample("log_gdp", dist.Normal(mu, sigma), obs=log_gdp)
    return log_gdp


mcmc_9m1a = MCMC(NUTS(model_9m1a), num_warmup=500, num_samples=500, num_chains=4)
mcmc_9m1b = MCMC(NUTS(model_9m1b), num_warmup=500, num_samples=500, num_chains=4)

print("Exponential Prior")
mcmc_9m1a.run(
    jrng,
    rugged=df["rugged_std"].values,
    cid=df["cid"].values,
    log_gdp=df["log_gdp_std"].values,
)
mcmc_9m1a.print_summary()
print("Uniform Prior")
mcmc_9m1b.run(
    jrng,
    rugged=df["rugged_std"].values,
    cid=df["cid"].values,
    log_gdp=df["log_gdp_std"].values,
)
mcmc_9m1b.print_summary()
```

`sigma`'s prior doesn't have a measurable impact it's posterior


### 9M2

```python
df = pd.read_csv("../data/rugged.csv", sep=";")
df["log_gdp"] = jnp.log(df["rgdppc_2000"].values)
df = df.dropna(subset=["rgdppc_2000"])
df["log_gdp_std"] = df.log_gdp / df.log_gdp.mean()
df["rugged_std"] = df.rugged / df.rugged.max()
df["cid"] = jnp.where(df.cont_africa.values == 1, 0, 1)


def model_factory_9m2(sigma_scale):
    def model_9m2(rugged, cid, log_gdp):
        with numpyro.plate(name="continent", size=2):
            a = numpyro.sample("a", dist.Normal(0, 0.5))
            b = numpyro.sample("b", dist.Normal(0, 0.5))
        mu = numpyro.deterministic("mu", a[cid] + b[cid] * (rugged - 0.215))
        sigma = numpyro.sample("sigma", dist.Exponential(rate=sigma_scale))
        log_gdp = numpyro.sample("log_gdp", dist.Normal(mu, sigma), obs=log_gdp)
        return log_gdp

    return model_9m2


for sigma_scale in [1, 10, 100, 1_000]:
    _mcmc = MCMC(
        NUTS(model_factory_9m2(sigma_scale)),
        num_warmup=500,
        num_samples=500,
        num_chains=4,
    )
    _mcmc.run(
        jrng,
        rugged=df["rugged_std"].values,
        cid=df["cid"].values,
        log_gdp=df["log_gdp_std"].values,
    )
    _mcmc.print_summary()
```

Prior on sigma hardle makes a difference


### 9M3

```python
df = pd.read_csv("../data/rugged.csv", sep=";")
df["log_gdp"] = jnp.log(df["rgdppc_2000"].values)
df = df.dropna(subset=["rgdppc_2000"])
df["log_gdp_std"] = df.log_gdp / df.log_gdp.mean()
df["rugged_std"] = df.rugged / df.rugged.max()
df["cid"] = jnp.where(df.cont_africa.values == 1, 0, 1)


def model_9m3(rugged, cid, log_gdp):
    with numpyro.plate(name="continent", size=2):
        a = numpyro.sample("a", dist.Normal(1, 0.1))
        b = numpyro.sample("b", dist.Normal(0, 0.3))
    mu = numpyro.deterministic("mu", a[cid] + b[cid] * (rugged - 0.215))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    log_gdp = numpyro.sample("log_gdp", dist.Normal(mu, sigma), obs=log_gdp)
    return log_gdp


n_effs = []
num_warmups = [0, 10, 25, 50, 100, 250, 500]
for num_warmup in num_warmups:
    _mcmc = MCMC(NUTS(model_9m3), num_warmup=num_warmup, num_samples=500, num_chains=2)
    _mcmc.run(
        jrng,
        rugged=df["rugged_std"].values,
        cid=df["cid"].values,
        log_gdp=df["log_gdp_std"].values,
    )

    n_effs.append(
        {
            k: v["n_eff"].mean()
            for k, v in numpyro.diagnostics.summary(
                _mcmc.get_samples(), group_by_chain=False
            ).items()
        }
    )
n_effs = pd.DataFrame(n_effs, index=pd.Index(num_warmups, name="num_warmup"))
n_effs
```

## Hard
### 9H1


There's no data, so we're just sampling from the prior predictive distribution.
Becasue cauchy is fat tailed, the trace sometimes jumps, which is normal.

```python
def model_9h1():
    a = numpyro.sample("a", dist.Normal(0, 1))
    b = numpyro.sample("b", dist.Cauchy(0, 1))


mcmc_9h1 = MCMC(NUTS(model_9h1), num_warmup=100, num_samples=1_000, num_chains=1)
mcmc_9h1.run(jrng)
mcmc_9h1.print_summary()

idata_9h1 = az.from_numpyro(mcmc_9h1)
az.plot_trace(idata_9h1)
```

### 9H2

```python
df = pd.read_csv("../data/WaffleDivorce.csv", sep=";")
df["A"] = scale(df["MedianAgeMarriage"])
df["D"] = scale(df["Divorce"])
df["M"] = scale(df["Marriage"])
df.head()
```

```python
def model_5_1(A, D):
    a = numpyro.sample("a", dist.Normal(0, 0.2))
    b_A = numpyro.sample("b_A", dist.Normal(0, 0.5))
    mu = numpyro.deterministic("mu", a + b_A * A)
    sigma = numpyro.sample("sigma", dist.Exponential(rate=1))
    D = numpyro.sample("D", dist.Normal(mu, sigma), obs=D)
    return D


def model_5_2(M, D):
    a = numpyro.sample("a", dist.Normal(0, 0.2))
    b_M = numpyro.sample("b_M", dist.Normal(0, 0.5))
    mu = numpyro.deterministic("mu", a + b_M * M)
    sigma = numpyro.sample("sigma", dist.Exponential(rate=1))
    D = numpyro.sample("D", dist.Normal(mu, sigma), obs=D)
    return D


def model_5_3(A, M, D):
    a = numpyro.sample("a", dist.Normal(0, 0.2))
    b_A = numpyro.sample("b_A", dist.Normal(0, 0.5))
    b_M = numpyro.sample("b_M", dist.Normal(0, 0.5))
    mu = numpyro.deterministic("mu", a + b_A * A + b_M * M)
    sigma = numpyro.sample("sigma", dist.Exponential(rate=1))
    D = numpyro.sample("D", dist.Normal(mu, sigma), obs=D)
    return D


mcmc_5_1 = MCMC(NUTS(model_5_1), num_warmup=500, num_samples=500, num_chains=4)
mcmc_5_1.run(jrng, A=df["A"].values, D=df["D"].values)
idata_5_1 = az.from_numpyro(mcmc_5_1)

mcmc_5_2 = MCMC(NUTS(model_5_2), num_warmup=500, num_samples=500, num_chains=4)
mcmc_5_2.run(jrng, M=df["M"].values, D=df["D"].values)
idata_5_2 = az.from_numpyro(mcmc_5_2)

mcmc_5_3 = MCMC(NUTS(model_5_3), num_warmup=500, num_samples=500, num_chains=4)
mcmc_5_3.run(jrng, A=df["A"].values, M=df["M"].values, D=df["D"].values)
idata_5_3 = az.from_numpyro(mcmc_5_3)
```

```python
compare = az.compare(
    {"model 5.1": idata_5_1, "model 5.2": idata_5_2, "model 5.3": idata_5_3},
    ic="loo",
    scale="deviance",
)
display(compare)
az.plot_compare(compare)
```

```python
mcmc_5_3.print_summary()
```

Model 5.1 appears to have the best predictions. Model 5.3 is only very slightly worse. Even though it is over parametrized, the beta coefficient is converges to the correct value of 0.


### 9H3

```python
N = 100  # number of individuals
height = dist.Normal(10, 2).sample(jrng, sample_shape=(N,))
leg_prop = dist.Uniform(0.4, 0.5).sample(jrng2, sample_shape=(N,))
left_leg = leg_prop * height + dist.Normal(0, 0.02).sample(jrng3, sample_shape=(N,))
right_leg = leg_prop * height + dist.Normal(0, 0.02).sample(jrng4, sample_shape=(N,))
df = pd.DataFrame({"height": height, "left": left_leg, "right": right_leg})
df
```

```python
def model_9h3a(left_leg, right_leg, height):
    a = numpyro.sample("a", dist.Normal(10, 100))
    b_left = numpyro.sample("b_left", dist.Normal(2, 10))
    b_right = numpyro.sample("b_right", dist.Normal(2, 10))
    mu = numpyro.deterministic("mu", a + b_left * left_leg + b_right * right_leg)
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    height = numpyro.sample("height", dist.Normal(mu, sigma), obs=height)
    return height


mcmc_9h3a = MCMC(NUTS(model_9h3a), num_warmup=250, num_samples=500, num_chains=4)
mcmc_9h3a.run(
    jrng,
    left_leg=df["left"].values,
    right_leg=df["right"].values,
    height=df["height"].values,
)
idata_9h3a = az.from_numpyro(mcmc_9h3a)
mcmc_9h3a.print_summary()
```

```python
def model_9h3b(left_leg, right_leg, height):
    a = numpyro.sample("a", dist.Normal(10, 10))
    b_left = numpyro.sample("b_left", dist.Normal(2, 10))
    _b_right = numpyro.sample("_b_right", dist.Normal(2, 10))
    b_right = numpyro.deterministic("b_right", jnp.abs(_b_right))
    mu = numpyro.deterministic("mu", a + b_left * left_leg + b_right * right_leg)
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    height = numpyro.sample("height", dist.Normal(mu, sigma), obs=height)
    return height


mcmc_9h3b = MCMC(NUTS(model_9h3b), num_warmup=500, num_samples=500, num_chains=4)
mcmc_9h3b.run(
    jrng,
    left_leg=df["left"].values,
    right_leg=df["right"].values,
    height=df["height"].values,
)
idata_9h3b = az.from_numpyro(mcmc_9h3b)
mcmc_9h3b.print_summary()
```

```python
az.plot_trace(idata_9h3a, var_names=["a", "b_left", "b_right"])
plt.tight_layout()
```

```python
az.plot_trace(idata_9h3b, var_names=["a", "b_left", "b_right"])
plt.tight_layout()
```

Changing the prior on b_right changes the posterior on b_left because their posteriors are highly correlated.


### 9H4

```python
compare_9h4 = az.compare({"model a": idata_9h3a, "model b": idata_9h3b}, ic="loo")
display(compare_9h4)
```

Tie between the two models. The second one has fewer effective parameters because the prior is tighter.


### 9H5

```python
num_weeks = int(1e5)
positions = jnp.repeat(0, num_weeks)
populations = jax.random.permutation(jrng, jnp.array(range(1, 11)))
current = 10


def metropolis_sample(i, val):
    positions, current = val

    positions = positions.at[i].set(current)

    # flip a coin to get proposal
    proposal = (
        current + 2 * dist.Bernoulli(probs=0.5).sample(jax.random.fold_in(jrng, i)) - 1
    )
    proposal = jnp.where(proposal > 10, 1, proposal)
    proposal = jnp.where(proposal < 1, 10, proposal)

    # decide whether to accept proposal
    probability_move = populations[proposal - 1] / populations[current - 1]
    evaluation = dist.Uniform().sample(jax.random.fold_in(jrng2, i))
    current = jnp.where(evaluation < probability_move, proposal, current)

    return (positions, current)


positions, current = jax.lax.fori_loop(
    0,
    num_weeks,
    metropolis_sample,
    (positions, current),
)
```

```python
expected_visits = populations / populations.sum() * num_weeks
print("Expected Visits")
expected_visits
```

```python
plt.hist(jnp.asarray(positions), bins=range(1, 12), rwidth=0.1, align="left")
```

### 9H6

```python
num_tosses = 10_000
positions = jnp.repeat(0, num_tosses)
likelihoods = jnp.array([0.3, 0.7])
current = 1  # 1 = land, 2 = water


def metropolis_sample(i, val):
    positions, current = val

    positions = positions.at[i].set(current)

    # the proposal is always the one alternative
    proposal = jnp.where(current == 1, 2, 1)

    # decide whether to accept proposal
    probability_move = likelihoods[proposal - 1] / likelihoods[current - 1]
    evaluation = dist.Uniform().sample(jax.random.fold_in(jrng2, i))
    current = jnp.where(evaluation < probability_move, proposal, current)

    return (positions, current)


positions, current = jax.lax.fori_loop(
    0,
    num_weeks,
    metropolis_sample,
    (positions, current),
)
```

```python
plt.hist(
    jnp.asarray(positions), bins=range(1, 4), rwidth=0.1, align="left", density=True
)
```
