# KOptions: User Input Guide

Welcome to the KOptions User Input Guide.
KOptions uses YAML configuration files to control simulations and validation tests.

Two types of input files are supported:

1. **Simulation configuration file**

   Defines the financial model, option contract, numerical method, Monte Carlo parameters, and output settings used to price an option.

2. **Test configuration file**

   Selects one or more automated validation tests used to verify the correctness, numerical accuracy, and reliability of the library.

Although both files use the same YAML syntax, they serve different purposes.

---

# 1. Simulation Configuration File

The simulation configuration specifies all parameters required to run an option pricing simulation.

Example:

```yaml
Market:
  Price: 8.0
  Variance: 0.0625
  r: 0.10
  q: 0.00

Options:
  OptionType: American
  OptionRight: Put
  StrikePrice: 10.0

Model:
  Heston:
    k: 5.0
    theta: 0.16
    sigma: 0.9
    rho: 0.1

Numerics:
  Scheme: ImplicitMilstein

Time:
  T_End: 0.25
  Inp_DT: 0.00025

MC:
  N_Paths: 500000
  Batch_Size: 0
  RNG_Seed: 88471920573105
  Max_VRAM_MB: 256
  Max_CPU_RAM_MB: 8000

Output:
  out_dir: outputs
  Name_Paths_Out_File: Paths.KOpaths
  Name_Log_File: KOptions.KOlog
  Format: BIN
```

### Market

The Market block defines the initial market conditions.

| Parameter | Description | Units | Constraints                         |
|---|---|---|-------------------------------------|
| Price | Initial underlying asset price S0. | Currency | >0                                  |
| Variance | Initial instantaneous variance v0. The corresponding volatility is \sqrtv0. | year−1 | $\geq0$                             |
| r | Annualized continuously compounded risk-free interest rate. | year−1 | approximately between -50% and +50% |
| q | Annualized continuous dividend yield. | year−1 | $\geq0$                             |


### Options

The Options block defines the financial contract.

| Parameter | Description |
|---|---|
| OptionType | Option type. Examples include European, American, Asian, and Barrier options. |
| OptionRight | Option right: Call or Put. |
| StrikePrice | Strike price K. Must be positive. |
| BarrierPrice | Barrier level. Required only for barrier options. |

### Model
The Model block specifies the stochastic process considered followed by the associated parameters.

Currently supported models include:

- Heston stochastic volatility model  
- Bates jump-diffusion model

#### Heston Parameters

The Heston model uses the stochastic variance process

$$
dv_t=\kappa(\theta-v_t)\,dt+\sigma\sqrt{v_t}\,dW_{2,t}.
$$

The asset price follows

$$
dS_t=(r-q)S_t\,dt+\sqrt{v_t}S_t\,dW_{1,t}.
$$

The Brownian motions are correlated in the following way:

$$ E[dW_t(1)dW_t(2)]=\rho dt.$$

| Parameter          | Description | Units |
|--------------------|---|---|
| $k$ (k)            | Mean reversion speed \kappa. Controls how quickly variance returns toward its long-term value. | year−1 |
| $\theta$   (theta) | Long-run variance \theta. The corresponding long-run volatility is \sqrt\theta. | year−1 |
| $\sigma$ (sigma)   | Volatility of variance ("vol-of-vol"). | year−1/2 |
| $\rho$ (rho)       | Correlation between asset and variance Brownian motions. | dimensionless |

#### Bates Parameters

### Numerical Scheme

The Numerics block selects the discretization scheme used to solve the stochastic differential equations.


Available schemes include:

- Euler
- ImplicitMilstein
- AndersonQE
- ...

Different schemes provide different trade-offs between:

- numerical accuracy;
- stability;
- computational cost.


### Time

The Time block defines the simulation horizon and discretization.

| Parameter | Description |
|---|---|
| T_End | Time to maturity in years. |
| Inp_DT | Requested time step size in years. |

The solver automatically adjusts the time step so that an integer number of steps exactly reaches maturity:

$$
N=\left\lceil\frac{T}{\Delta t}\right\rceil
$$

$$
\Delta t=\frac{T}{N}.
$$

> **Note**: If the requested time step does not divide the maturity exactly, a warning is printed.

### Monte Carlo

The MC block controls the Monte Carlo simulation.

| Parameter | Description |
|---|---|
| N_Paths | Total number of simulated Monte Carlo paths. |
| Batch_Size | Number of paths processed simultaneously. |
| RNG_Seed | Seed used by the random number generator. |
| Max_VRAM_MB | Maximum GPU memory allowed. |
| Max_CPU_RAM_MB | Maximum host memory allowed. |

### Batch Size

Batch_Size has three possible modes:

| Value | Meaning |
|---|---|
| 0 | Automatic selection based on available hardware. |
| -1 | Process all paths simultaneously. |
| >0 | User-defined batch size. |

### Output 

This block controls generated files.

| Parameter	|Description| 
|---|---|
|out_dir |Output directory.  It is automatically created if missing.
| Name_Paths_Out_File | File containing generated Monte Carlo paths.
| Name_Log_File |Simulation log file.
|Format | Output format: TXT or BIN.

***
# 2. Automated Test Configuration

The test configuration file is used to execute the automated validation suite.

It does not price an option. Instead, it runs predefined tests designed to verify the correctness and numerical accuracy of KOptions.

Example:
```yaml
run_tests:
  - RNG
  - IOBin
  - Heston
  - AmericanOption
```

Multiple tests can be executed in a single run.

### Available Automated Tests

| Test | Purpose                                                                                                                      |
|---|------------------------------------------------------------------------------------------------------------------------------|
| RNG | Verifies that the Gaussian random number generator correctly samples a standard normal distribution.                         |
| IOBin | Verifies that binary files are correctly written to disk and reloaded without data corruption when using the binfary format. |
| NoNoise | Verifies the deterministic zero-variance limit of the stochastic differential equation solver.                               |
| BlackScholes | Verifies that, as the time discretization is refined, the numerical solution reproduces the exact Black-Scholes result.      |
| Heston | Verifies that, as the time discretization is refined, the numerical solution reproduces the reference Heston result.         |
| Bates | Verifies that, as the time discretization is refined, the numerical solution reproduces the reference Bates result.          |
| AmericanOption | Validates the Least-Squares Monte Carlo (LSM / Longstaff-Schwartz) implementation for American option pricing.               |


> **Note**: Tests involving the SDEs, by progressively reducing the simulation time step, verify that the implemented numerical schemes reproduce analytical solutions or high-accuracy reference solutions within the expected numerical error.

***