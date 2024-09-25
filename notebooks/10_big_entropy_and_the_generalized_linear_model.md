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

# Chapter 10: Big Entropy and the Generalized Linear Model

```python
%load_ext jupyter_black

import arviz as az
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpyro.distributions as dist

seed = 84735
jrng = jax.random.key(seed)
plt.rcParams["figure.figsize"] = [10, 6]
```

## Code
### Code 10.1

```python
p = {
    "A": jnp.array([0, 0, 10, 0, 0]),
    "B": jnp.array([0, 1, 8, 1, 0]),
    "C": jnp.array([0, 2, 6, 2, 0]),
    "D": jnp.array([1, 2, 4, 2, 1]),
    "E": jnp.array([2, 2, 2, 2, 2]),
}
p
```

### Code 10.2

```python
p_norm = jax.tree.map(lambda x: x / jnp.sum(x), p)
p_norm
```

### Code 10.3

```python
H = jax.tree.map(lambda q: -jax.scipy.special.xlogy(q, q).sum(), p_norm)
H
```

### Code 10.4

```python
ways = jnp.array([1, 90, 1260, 37800, 113400])
log_ways_pp = jnp.log(ways)
plt.scatter(log_ways_pp, H.values())
x = jnp.linspace(log_ways_pp.min(), log_ways_pp.max(), 30)
dy_dx = jnp.polyfit(log_ways_pp, jnp.array(list(H.values())), 1)
plt.plot(x, dy_dx[0] * x, "k--", alpha=0.3)
```

### Code 10.5

```python
p = jnp.array(
    [
        [1 / 4, 1 / 4, 1 / 4, 1 / 4],
        [2 / 6, 1 / 6, 1 / 6, 2 / 6],
        [1 / 6, 2 / 6, 2 / 6, 1 / 6],
        [1 / 8, 4 / 8, 2 / 8, 1 / 8],
    ]
)
(p * p.sum(axis=1)).sum(axis=1)
```

### Code 10.6

```python
jax.scipy.special.entr(p).sum(axis=1)
```

### Code 10.7

```python
p = 0.7
A = jnp.array([(1 - p) ** 2, p * (1 - p), (1 - p) * p, p**2])
A
```

### Code 10.8

```python
jax.scipy.special.entr(A).sum()
```

### Code 10.9

```python
def sim_p(i, G=1.4):
    x123 = dist.Uniform().sample(jax.random.fold_in(jrng, i), sample_shape=(3,))
    x4 = (G * jnp.sum(x123, keepdims=True) - x123[1] - x123[2]) / (2 - G)
    z = jnp.sum(jnp.concatenate([x123, x4]))
    p = jnp.concatenate([x123, x4]) / z
    return {"H": -jnp.sum(p * jnp.log(p)), "p": p}
```

### Code 10.10

```python
H = jax.vmap(lambda i: sim_p(i, G=1.4))(jnp.arange(int(1e5)))
az.plot_kde(H["H"], bw=0.0005)
plt.show()
```

### Code 10.11

```python
entropies = H["H"]
distributions = H["p"]
```

### Code 10.12

```python
entropies.max()
```

### Code 10.13

```python
distributions[jnp.argmax(entropies)]
```

```python

```
