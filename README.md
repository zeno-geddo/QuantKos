# QuantKos: Performance-Portable Monte Carlo Option Pricing Engine

**QuantKos** is a high-performance Monte Carlo option pricing engine written in modern **C++20** and built on top of the **Kokkos** performance portability framework. It provides a single-source implementation capable of targeting serial execution, multi-core CPUs, and modern GPUs while maintaining the same code base.

QuantKos is being developed as a personal research project to explore performance-portable stochastic option pricing. It features a clean modular architecture that separates stochastic models, numerical integration schemes, payoff evaluations, Monte Carlo orchestration ans sensibility analysis, making it easy to extend with new financial models or derivative contracts.

**Please Note**: QuantKos is intended to be portable across all hardware backends supported by Kokkos, it has currently been developed and tested primarily on **Linux**, using **GCC**, **OpenMP**, and **NVIDIA CUDA**.

---

## ✨ Features

### Performance Portability

- Single C++ implementation for multiple hardware architectures
- Procedural (Serial) execution
- Multi-core CPU execution (OpenMP)
- NVIDIA GPU acceleration (CUDA)
- Support for additional Kokkos backends (HIP, SYCL, etc.) in principle

### Stochastic Models

Currently implemented:

- Heston stochastic volatility model
- Bates stochastic volatility jump-diffusion model

The modular architecture makes it reasonably easy to add additional stochastic differential equation (SDE) models.

### Supported Option Contracts

QuantKos currently supports pricing of

- European options
- American options (Longstaff-Schwartz Least-Squares Monte Carlo)
- Asian options
- Barrier options
- Lookback options
- Binary options

Additional options types can be readily implemented.

### Numerical Methods

Several discretization schemes are available, including

- Euler
- Implicit Milstein
- Andersen Quadratic-Exponential (QE)

The implementation follows the numerical methods described in

> Fabrice D. Rouah,
> *The Heston Model and its Extensions in Matlab and C#*

### Monte Carlo Engine

The simulation engine includes

- Parallel Monte Carlo path generation
- Automatic or customizable batching for very large simulations
- Memory-aware execution on both CPUs and GPUs
- Reproducible simulations through configurable random seeds
- Configurable single- or double-precision builds

### Sensitivity Analysis (Greeks)

QuantKos includes built-in capabilities for calculating risk sensitivities and hedge parameters (Greeks) across supported contracts and models:

- **Delta ($\Delta$):** Sensitivity of the option price to changes in the underlying asset price
- **Vega ($\nu$):** Sensitivity of the option price to changes in volatility (or volatility model parameters)
- **Theta ($\Theta$):** Sensitivity of the option price to the passage of time (time decay)
- **Rho ($\rho$):** Sensitivity of the option price to changes in the risk-free interest rate

- **Gamma ($\Gamma$):** Rate of change of Delta with respect to changes in the underlying asset price
- **Vanna:** Sensitivity of Delta with respect to volatility (or rate of change of Vega with respect to underlying price)
- **Vomma:** Rate of change of Vega with respect to volatility

Sensitivities are evaluated using finite-difference approximations. For more details, refer to the documentation.


---

## 📚 Documentation
- 📖 [Main Branch Documentation (Stable)](https://zeno-geddo.github.io/QuantKos/main/)
- 🧪 [Dev Branch Documentation (Latest)](https://zeno-geddo.github.io/QuantKos/dev-zen/)

Note that the documentation is split into dedicated guides.

| Document | Description                                                                                                  |
|----------|--------------------------------------------------------------------------------------------------------------|
| **COMPILATION_GUIDE.md** | Installation, dependencies, hardware backends, CUDA/OpenMP builds, and CMake configuration.                  |
| **USER_INPUT_GUIDE.md** | Complete description of the YAML configuration files used for simulations and validation tests.              |
| **MATH_GUIDE.md** | Quick mathematical background of the implemented stochastic models, numerical schemes, and option contracts. |

When Doxygen and Graphviz are installed, QuantKos can also generate complete HTML and PDF API documentation during compilation.

---

## 🚀 Quick Start

The simplest way to build QuantKos is to let CMake automatically download **Kokkos** and **yaml-cpp** using **FetchContent**.

### Clone the repository

```bash
git clone https://github.com/zeno-geddo/QuantKos.git
cd QuantKos
```


### Interactive Python Builder (Recommended)
You can quickly configure and build QuantKos interactively using the provided Python command-line interface. From the repository root, simply run:
```bash
python builder.py
```
This script will guide you through the configuration process—allowing you to easily select hardware backends, math precision, and dependency management strategies—before automatically compiling and installing the project. 
> **IMPORTANT NOTE:** The interactive builder interface is currently under active development. For advanced GPU configurations, custom architecture flags, or specialized Kokkos builds, it is recommended to use the manual build workflow by following the details in COMPILATION_GUIDE.md.




### Standard CMake Build and Install
The simplest way to build QuantKos manually is to let CMake automatically download Kokkos and yaml-cpp using FetchContent.

#### Create a build directory

```bash
mkdir build
cd build
```

#### Configure a Release build

#### Serial

```bash
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DQKOS_ENABLE_FETCHCONTENT=ON \
    -DQKOS_ENABLE_SINGLE_PRECISION=OFF \
    -DQKOS_USE_FAST_MATH=OFF \
    -DQKOS_ENABLE_TESTS=ON \
    -DQKOS_BUILD_DOC=ON \
    -DKokkos_ENABLE_SERIAL=ON \
    -DKokkos_ENABLE_OPENMP=OFF \
    -DKokkos_ENABLE_CUDA=OFF \
    -DKokkos_ARCH_NATIVE=ON
```

#### OpenMP

```bash
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DQKOS_ENABLE_FETCHCONTENT=ON \
    -DQKOS_ENABLE_TESTS=ON \
    -DQKOS_ENABLE_SINGLE_PRECISION=OFF \
    -DQKOS_USE_FAST_MATH=OFF \
    -DQKOS_BUILD_DOC=ON \
    -DQKOS_ENABLE_TESTS=ON \
    -DKokkos_ENABLE_SERIAL=ON \
    -DKokkos_ENABLE_OPENMP=ON \
    -DKokkos_ENABLE_CUDA=OFF \
    -DKokkos_ARCH_NATIVE=ON
```

> **IMPORTANT NOTE:** For CUDA builds, custom Kokkos installations, or advanced CMake configuration, see **COMPILATION_GUIDE.md**.

#### Build using cmake

```bash
cmake --build . -j$(nproc)
```

#### Install using cmake

```bash
cmake --install .
```

---

## ▶️ Running QuantKos

Execute the pricing engine by passing a simulation configuration file.

```bash
./bin/QuantKos share/QuantKosConfig.yaml
```

For OpenMP builds, the number of threads can be selected at runtime.

```bash
./bin/QuantKos share/QuantKosConfig.yaml --kokkos-threads=8
```

or equivalently

```bash
OMP_NUM_THREADS=8 ./bin/QuantKos share/QuantKosConfig.yaml
```

For CUDA builds on systems with multiple GPUs, a specific device can be selected.

```bash
./bin/QuantKos share/QuantKosConfig.yaml --kokkos-device-id=0
```

The complete description of all configuration parameters is available in **USER_INPUT_GUIDE.md**.

---

## ⚙️ Simulation Configuration

QuantKos uses YAML configuration files.

A simulation file specifies

- market parameters
- stochastic model
- option contract
- numerical integration scheme
- Monte Carlo settings
- output options

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

Output:
  out_dir: outputs
```

The full specification of every parameter is documented in **USER_INPUT_GUIDE.md**.

---

## 🧪 Validation Suite

QuantKos includes a standalone testing executable that validates the numerical implementation.

Compile with

```text
-DQKOS_ENABLE_TESTS=ON
```

and execute

```bash
./bin/Tester share/TesterConfig.yaml
```

The automated validation suite currently includes tests for

- Random number generation
- Binary I/O
- Deterministic limits
- Black-Scholes convergence
- Heston convergence
- Bates convergence
- Longstaff-Schwartz American option pricing
- Call-Put parity tests for EU options, checking that the greeks works as expected.

These tests are useful for validating new installations, verifying code modifications, and comparing different hardware backends.

---

## 🎯 Project Goals

QuantKos aims to provide

- an extensible C++ framework for option pricing
- performance portability across heterogeneous hardware

---

## 📄 License

QuantKos is distributed under the **GNU General Public License v3.0**.

See the `LICENSE` file for details.

---

## 🤝 Contributing

Bug reports, feature requests, and pull requests are always welcome.

If you encounter a portability issue on a hardware platform that is not yet officially tested, please consider opening an issue with your build configuration and compiler information.

---

## 👤 Author

QuantKos was designed and developed by **[Zeno Geddo](https://github.com/zeno-geddo)** as an independent research and software engineering project in quantitative finance.

The project has been developed out of personal interest in stochastic differential equations, numerical methods, high-performance computing, and performance-portable scientific software.

***
***