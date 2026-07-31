# QuantKos : Math Guide

Welcome to the QuantKos Math Guide.
This document briefly describes the mathematical foundations of the QuantKos framework. It presents the stochastic models currently implemented to simulate the dynamics of the underlying asset, together with the option contracts that can be priced using these models.
More the details about the models can in found in :
- *The Heston Model and its Extensions in Matlab and C#*, by Fabrice D. ROUAH.

The framework has been designed to be modular and extensible, allowing additional stochastic differential equation (SDE) models and derivative contracts to be integrated with restricted changes to the pricing engine.

Currently, the following stochastic models are implemented:

- **Heston Stochastic Volatility Model**
- **Bates Stochastic Volatility Jump-Diffusion Model**

These models can be combined with all supported option contracts presented later in this guide.

***

## Stochastic Models

### Common Market Parameters

Both implemented stochastic models share the same market environment.

| Parameter | Description                                          |
|-----------|------------------------------------------------------|
| $S_t$ | Underlying asset price                               |
| $v_t$ | Instantaneous variance of the underlying asset price |
| $r$ | Annualized risk-free interest rate                   |
| $q$ | Annualized continuous dividend yield                 |
| $W_1$, $W_2$ | Correlated Assets and Variance Brownian motions      |
| $\rho$ | Correlation between the asset and variance processes |

The Brownian motions satisfy the Itô correlation relation

$$
dW_{1,t}\,dW_{2,t}=\rho\,dt.
$$


### I. Heston Stochastic Volatility Model

The Heston model extends the classical Black-Scholes framework by replacing the assumption of constant volatility with a stochastic variance process exhibiting mean reversion.

The dynamics are

$$
dS_t=(r-q)S_t\,dt+\sqrt{v_t}S_t\,dW_{1,t},
$$

$$
dv_t=\kappa(\theta-v_t)\,dt+\sigma\sqrt{v_t}\,dW_{2,t}.
$$

#### Parameters
| Parameter | Description                          |
|-----------|--------------------------------------|
| $r$ | Annualized risk-free interest rate   |
| $q$ | Annualized continuous dividend yield |
| $\kappa$ | Mean reversion speed of the variance |
| $\theta$ | Long-run variance                    |
| $\sigma$ | Volatility of variance               |
| $\rho$ | Asset-variance correlation           |

> **Note: Feller Condition**. 
> The variance process remains strictly positive whenever
$$
2\kappa\theta>\sigma^2.
$$
> When this condition is violated, the variance may reach zero during the simulation. The QuantKos framework reports this condition since it may influence numerical stability depending on the discretization scheme.

> **Note: Implemented Schemes**.
> QuantKos implements a variety of discretizations schemes detailed in the book *The Heston Model and its Extensions in Matlab and C#* by Fabrice D. ROUAH.

### II. Bates Stochastic Volatility Jump-Diffusion Model

The Bates model extends the Heston model by incorporating a Merton jump process into the asset dynamics while preserving the stochastic variance process.

The governing equations are

$$
dS_t=
(r-q-\lambda\kappa_J)S_t\,dt
+\sqrt{v_t}S_t\,dW_{1,t}
+S_{t^-}(J-1)\,dN_t,
$$

$$
dv_t=
\kappa(\theta-v_t)\,dt
+\sigma\sqrt{v_t}\,dW_{2,t}.
$$

The jump size satisfies

$$
\ln(J)\sim\mathcal N(\mu_J,\sigma_J^2),
$$

where

- $N_t$ is a Poisson process with intensity $\lambda$,
- $J$ is the multiplicative jump size,
- $S_{t^-}$ denotes the asset price immediately before a jump.

The martingale drift correction is

$$
\kappa_J=
E[J]-1=
\exp\!\left(\mu_J+\frac12\sigma_J^2\right)-1.
$$

#### Core Parameters

The Bates model inherits all Heston parameters.

| Parameter | Description                               |
|-----------|-------------------------------------------|
| $r$ | Annualized risk-free interest rate        |
| $q$ | Annualized continuous dividend yield      |
| $\kappa$ | Mean reversion speed of the variance (1/y) |
| $\theta$ | Long-run variance (1/y)                   |
| $\sigma$ | Volatility of variance (1/(y*y))          |
| $\rho$ | Asset-variance correlation                |

#### Jump Parameters

| Parameter | Description                                      |
|-----------|--------------------------------------------------|
| $\lambda$ | Average price jump intensity per unit time (1/y) |
| $\mu_J$ | Mean logarithmic jump size                       |
| $\sigma_J$ | Standard deviation of the jump size  (log-space) |
| $\kappa_J$ | Drift martingale compensation term               |

***

# Supported Option Contracts

The QuantKos framework supports a wide range of derivative contracts, all of which can be priced using any of the implemented stochastic models.

Unless otherwise specified, every option type is available in both **Call** and **Put** versions.

Throughout this section,

- $S_T$ denotes the asset price at maturity,
- $K$ the strike price,
- $B$ the barrier level,
- $A$ the arithmetic average asset price,
- $M$ the maximum asset price observed during the option lifetime,
- $m$ the minimum asset price observed during the option lifetime.

### European Option

A European option may only be exercised at maturity.

For a Call, the payoff is $$\max(S_T-K,0)$$, while for a Put the payoff is $$\max(K-S_T,0)$$.

The price is obtained as the discounted expectation of the terminal payoff under the risk-neutral measure.


### Asian Option

Asian options replace the terminal asset price with the arithmetic average

$$
A=\frac1N\sum_{i=1}^{N}S_{t_i},
$$

making them less sensitive to temporary price fluctuations.

The payoff is $$\max(A-K,0)$$ for a Call and $$\max(K-A,0)$$ for a Put.


### Barrier Options

Barrier options become active or inactive depending on whether the underlying crosses a barrier during its lifetime.

#### Up-and-Out

The option is cancelled if

$$
S_t\ge B.
$$

Otherwise, it behaves as a European option.

#### Down-and-Out

The option is cancelled if

$$
S_t\le B.
$$

Otherwise, it behaves as a European option.

#### Up-and-In

The option becomes active only after

$$
S_t\ge B.
$$

If the barrier is never reached, the payoff is zero.

#### Down-and-In

The option becomes active only after

$$
S_t\le B.
$$

If the barrier is never reached, the payoff is zero.


### Lookback Options

Lookback options exploit the historical extrema of the simulated asset path.

#### Floating Strike

The strike is determined by the path itself.

For a Call, the payoff is

$$
\max(S_T-m,0),
$$

where $m$ is the minimum observed asset price.

For a Put,

$$
\max(M-S_T,0),
$$

where $M$ is the maximum observed asset price.

#### Fixed Strike

The strike remains fixed.

The payoff is $\max(M-K,0)$ for a Call and $\max(K-m,0)$ for a Put.


### Binary Options

Binary options pay a discontinuous payoff whenever they expire in the money.

#### Cash-or-Nothing

A fixed cash amount $Q$ is paid.

The payoff is $$Q\mathbf{1}_{S_T>K}$$ for a Call and $$Q\mathbf{1}_{S_T<K}$$ for a Put.

#### Asset-or-Nothing

The payoff equals the terminal asset value.

It is $$S_T\mathbf{1}_{S_T>K}$$ for a Call and $$S_T\mathbf{1}_{S_T<K}$$ for a Put.


### American Option

American options may be exercised at any time before maturity.

Within QuantKos, the optimal stopping problem is solved using the **Longstaff-Schwartz method**.

***

# Pricing Methodology

Regardless of the chosen stochastic model or option contract, pricing follows the same workflow.

1. Simulate asset paths according to the selected stochastic model.
2. Track all path-dependent quantities required by the option contract.
3. Evaluate the payoff on each simulated path.
4. Discount the payoff
5. Estimate the option price as the average discounted payoff over all simulated paths.

This separation between stochastic models and payoff definitions allows the same Monte Carlo engine to efficiently price a large variety of derivative contracts.
