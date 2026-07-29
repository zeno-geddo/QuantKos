# KOptions: Performance-Portable Monte Carlo Option Pricing Engine

**KOptions** is a high-performance Monte Carlo option pricing engine written in modern **C++20** and built on top of the **Kokkos** performance portability framework. It provides a single-source implementation capable of targeting serial execution, multi-core CPUs, and modern GPUs while maintaining the same code base.

KOptions has been designed as both a research and development framework for stochastic option pricing and a production-oriented pricing engine. 
It cleanly separates stochastic models, numerical integration schemes, payoff evaluation and Monte Carlo orchestration, so that is it possible to easily extend it with new financial models or derivative contracts.

Although KOptions is intended to be portable across all hardware backends supported by Kokkos, it has currently been developed and tested primarily on **Linux**, using **GCC**, **OpenMP**, and **NVIDIA CUDA**.

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

KOptions currently supports pricing of

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

---

## 📚 Documentation

Detailed documentation is split into dedicated guides.

| Document | Description                                                                                                  |
|----------|--------------------------------------------------------------------------------------------------------------|
| **COMPILATION_GUIDE.md** | Installation, dependencies, hardware backends, CUDA/OpenMP builds, and CMake configuration.                  |
| **USER_INPUT_GUIDE.md** | Complete description of the YAML configuration files used for simulations and validation tests.              |
| **MATH_GUIDE.md** | Quick mathematical background of the implemented stochastic models, numerical schemes, and option contracts. |

When Doxygen and Graphviz are installed, KOptions can also generate complete HTML and PDF API documentation during compilation.

---

## 🚀 Quick Start

The simplest way to build KOptions is to let CMake automatically download **Kokkos** and **yaml-cpp** using **FetchContent**.

### Clone the repository

```bash
git clone https://github.com/zeno.geddo/KOptions.git
cd KOptions
```

### Create a build directory

```bash
mkdir build
cd build
```

### Configure a Release build

#### Serial

```bash
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DKOPS_ENABLE_FETCHCONTENT=ON \
    -DKokkos_ENABLE_SERIAL=ON \
    -DKokkos_ENABLE_OPENMP=OFF \
    -DKokkos_ENABLE_CUDA=OFF \
    -DKokkos_ARCH_NATIVE=ON
```

#### OpenMP

```bash
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DKOPS_ENABLE_FETCHCONTENT=ON \
    -DKokkos_ENABLE_SERIAL=ON \
    -DKokkos_ENABLE_OPENMP=ON \
    -DKokkos_ENABLE_CUDA=OFF \
    -DKokkos_ARCH_NATIVE=ON
```

> **IMPORTANT NOTE:** For CUDA builds, custom Kokkos installations, or advanced CMake configuration, see **COMPILATION_GUIDE.md**.

### Build

```bash
cmake --build . -j$(nproc)
```

### Install

```bash
cmake --install .
```

---

## ▶️ Running KOptions

Execute the pricing engine by passing a simulation configuration file.

```bash
./bin/KOptions share/KOptionsConfig.yaml
```

For OpenMP builds, the number of threads can be selected at runtime.

```bash
./bin/KOptions share/KOptionsConfig.yaml --kokkos-threads=8
```

or equivalently

```bash
OMP_NUM_THREADS=8 ./bin/KOptions share/KOptionsConfig.yaml
```

For CUDA builds on systems with multiple GPUs, a specific device can be selected.

```bash
./bin/KOptions share/KOptionsConfig.yaml --kokkos-device-id=0
```

The complete description of all configuration parameters is available in **USER_INPUT_GUIDE.md**.

---

## ⚙️ Simulation Configuration

KOptions uses YAML configuration files.

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

KOptions includes a standalone testing executable that validates the numerical implementation.

Compile with

```text
-DKOPS_ENABLE_TESTS=ON
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

These tests are useful for validating new installations, verifying code modifications, and comparing different hardware backends.

---

## 🎯 Project Goals

KOptions aims to provide

- a clean and extensible C++ architecture for quantitative finance
- performance portability across heterogeneous hardware
- reproducible scientific simulations

---

## 📄 License

KOptions is distributed under the **GNU General Public License v3.0**.

See the `LICENSE` file for details.

---

## 🤝 Contributing

Bug reports, feature requests, and pull requests are always welcome.

If you encounter a portability issue on a hardware platform that is not yet officially tested, please consider opening an issue with your build configuration and compiler information.

---

## 👤 Author

KOptions was designed and developed by **[Zeno Geddo](https://github.com/zeno-geddo)** as an independent research and software engineering project in quantitative finance.

The project has been developed out of personal interest in stochastic differential equations, numerical methods, high-performance computing, and performance-portable scientific software.