# QuantKos: Performance-Portable Monte Carlo Option Pricing Engine

**QuantKos** is a high-performance Monte Carlo option pricing engine written in modern **C++20** and built on top of the **Kokkos** performance portability framework. It provides a single-source implementation capable of targeting serial execution, multi-core CPUs, and modern GPUs while maintaining the same code base.

By leveraging GPU acceleration and precision-tuning, QuantKos can achieve nearly 1000x speedups over standard single-threaded CPU implementations.
Check out the *Performance & Benchmarks* section below for detailed metrics and hardware analysis.


**Please Note**: Although QuantKos is intended to be portable across all hardware backends supported by Kokkos, it has currently been developed and tested primarily on **Linux**, using **GCC**, **OpenMP**, and **NVIDIA CUDA**.


---

## ✨ Features

QuantKos is being developed as a personal research project to explore performance-portable stochastic option pricing. 
It features a clean modular architecture that separates stochastic models, numerical integration schemes, 
payoff evaluations, Monte Carlo orchestration and sensibility analysis, making it easy to extend with new financial models or derivative contracts.


### Performance Portability
Single C++ implementation for multiple hardware architectures:
- Procedural (Serial) CPU execution
- Multi-core CPU execution (OpenMP)
- NVIDIA GPU acceleration (CUDA)
- Support for additional Kokkos backends (HIP, SYCL, etc.) in principle

### Build Types
Flexible build options:
- Configurable Release, RelWithDebInfo or Debug builds
- Configurable single- or double-precision builds
- Configurable ieee-math or fast-math  builds

### Monte Carlo Engine
The simulation engine includes:
- Optimized and parallel Monte Carlo path generation, with automatic or customizable batching for very large simulations
- Memory-aware execution on both CPUs and GPUs
- Reproducible simulations through configurable random seeds
- Detailed stdout/log messages
- Possibility to save the computed trajectories


### Supported Option Contracts
QuantKos currently supports pricing of:

- European options
- American options (Longstaff-Schwartz Least-Squares Monte Carlo)
- Asian options
- Barrier options
- Lookback options
- Binary options

Additional options types can be readily implemented.

### Stochastic Models
Currently implemented:

- Heston stochastic volatility model
- Bates stochastic volatility jump-diffusion model

The modular architecture makes it reasonably easy to add additional stochastic differential equation (SDE) models.

### Numerical Methods
Several discretization schemes are available, including:

- Euler
- Implicit Milstein
- Andersen Quadratic-Exponential (QE)

The implementation follows the numerical methods described in

> Fabrice D. Rouah,
> *The Heston Model and its Extensions in Matlab and C#*

Additional numerical schemes can be readily implemented.


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
The procedure can be automatized using an interactive python builder script calling cmake under the hood.

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

## ⚡ Performance & Benchmarks

QuantKos includes an automated Python benchmarking orchestrator (`benchmarks/run_benchmarks.py`) that builds, executes, 
and compares the execution throughput across different hardware backends, floating-point precisions, and compilation math modes.
You can run the same benchmarks on your machine, just by calling the runner and following the interactive steps.

### Benchmark Hardware Environment

The benchmarks below were conducted on a Linux laptop with the following hardware specifications:
* **CPU:** 12th Gen Intel(R) Core(TM) i7-12650H (10 Cores / 16 Threads, up to 4.70 GHz)
* **GPU:** NVIDIA GeForce RTX 3060 Laptop GPU (Ampere Architecture, Compute Capability 8.6)

### Benchmark Setup
We here report the reults obtained considering the following implemented benchmark :
* **Stochastic Model:** Heston Model via AndersenQE Discretization
* **Simulation Scale:** 5,000,000 Paths × 365 Time Steps ($1.825 \times 10^9$ total SDE trajectory steps)
* **Details Target Contract:** The parameters used for Asian Option Pricing can be found at : `benchmark/configs/bench_asian_option_price_no_greeks.yaml`. 
    This benchmark can be selected interactively when running the python benchmark script. 

### Execution Results

 Build Variant | Wall(s) | Overhead(s) | Comp(s)  | WallSpd | CompSpd |
 :--- | :--- | :--- | :--- | :--- | :--- | 
 `Release_CUDA_SINGLE_FAST` | 0.2831 | 0.1855 | 0.0976   | **351.53x** | **1019.84x** |
 `Release_CUDA_SINGLE_IEEE` | 0.4836 | 0.3499 | 0.1336   | 205.82x | 744.55x |
 `Release_CUDA_DOUBLE_IEEE` | 5.1056 | 0.1865 | 4.9191   | 19.49x | 20.23x |
 `Release_CUDA_DOUBLE_FAST` | 5.0989 | 0.1727 | 4.9262   | 19.52x | 20.20x |
 `Release_OPENMP_SINGLE_FAST` | 12.7264 | 0.0137 | 12.7128   | 7.82x | 7.83x |
 `Release_OPENMP_SINGLE_IEEE` | 13.8612 | 0.0166 | 13.8447   | 7.18x | 7.19x |
 `Release_OPENMP_DOUBLE_FAST` | 14.9199 | 0.0147 | 14.9052   | 6.67x | 6.68x |
 `Release_OPENMP_DOUBLE_IEEE` | 15.8384 | 0.0203 | 15.8181   | 6.28x | 6.29x |
 `Release_SERIAL_SINGLE_FAST` | 67.6384 | 0.0116 | 67.6269   | 1.47x | 1.47x |
 `Release_SERIAL_SINGLE_IEEE` | 73.2229 | 0.0140 | 73.2089  | 1.36x | 1.36x |
 `Release_SERIAL_DOUBLE_FAST` | 95.4783 | 0.0220 | 95.4563   | 1.04x | 1.04x |
 `Release_SERIAL_DOUBLE_IEEE` | 99.5232 | 0.0143 | 99.5089   | **BASE** | **BASE** |

### Technical Insights

1. **Extreme Throughput on GPU (1000x Speedup):**
   The GPU engine running in Single Precision (FP32) with Fast-Math completes 1.825 billion SDE steps in **under 0.098 seconds**, achieving a **~1020x computation speedup** over the baseline single-threaded CPU execution (`SERIAL_DOUBLE_IEEE` at 99.5s).

2. **The FP32 vs. FP64 GPU Hardware Cliff:**
   Notice the massive performance gap between `CUDA_SINGLE` (~0.09s) and `CUDA_DOUBLE` (~4.92s) on the GPU—a **~50x speed difference**. 
   This is not an algorithmic bottleneck; it reflects the **physical architecture of consumer GPUs** (such as the RTX 3060). 
   Consumer NVIDIA cards have a hardware-restricted FP64-to-FP32 ALU ratio (typically 1:64): 
   this physical limitation explains why switching from `IEEE` to `FAST` math in double precision yields virtually no performance gain (~4.92s for both). 
   IN that case execution is bottlenecked by the raw lack of FP64 silicon cores, rendering algorithmic math shortcuts for transcendental functions almost irrelevant.
3. **Understanding Startup Overhead:**
   The `Overhead(s)` column measures the delta between total process execution wall-clock time and internal C++ orchestration time (Wall - Comp).
    * **On CPU backends**, overhead is negligible (~0.01s).
    * **On GPU backends**, overhead ranges from **0.17s to 0.35s**. This delay is consumed by Kokkos runtime initialization, 
   CUDA driver context creation, memory allocation, and waking the GPU from low-power idle states (Cold Start penalty). 
   The initial GPU invocation in a process sequence often pays a slightly higher cold-start tax while the driver initializes the context.
   Note, in a production environment, the overheads can be almost totally removed by initializing the program (and thus kokkos and getting ready the GPU) before launching the computations. 
4. **Run-to-Run Variance Note:**
   Slight variations in execution metrics (few points percent) can occur across consecutive benchmark runs due to CPU thermal throttling, variable background OS usage, etc. 
   However, these benchmarks accurately represent the architectural performance ratios of the QuantKos engine.

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