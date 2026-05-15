
---

# KOptions: High-Performance Heston-Bates Engine

KOptions is a high-performance Monte Carlo simulation engine for pricing European and American options under the Heston and Bates stochastic volatility models (See Section 6). 
Built in **C++20** and accelerated by **Kokkos**, it is designed for extreme scalability across multi-core CPUs and GPUs.

---

## 🚀 1. Prerequisites

To build and run KOptions, your system must meet the following requirements.

**Core Build Tools:**

* **[CMake](https://cmake.org/)** (v3.20 or higher)
* *(Optional but highly recommended: The `ccmake` terminal GUI for advanced configuration).*

* **C++20 Compiler** ([GCC 10+](https://gcc.gnu.org/) or [Clang 12+](https://clang.llvm.org/))

**Hardware Toolchains (Depending on your target):**

* **[OpenMP](https://www.openmp.org/)**: Required if compiling for multi-core CPU parallelism.
* **[NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)**: Provides the `nvcc` compiler, required if compiling for NVIDIA GPUs.

**C++ Libraries:**

* **[Kokkos](https://github.com/kokkos/kokkos)**: The performance portability framework.
* **[yaml-cpp](https://github.com/jbeder/yaml-cpp)**: For parsing configuration files.
  *(Note: The CMake build system is configured to download and build Kokkos and yaml-cpp automatically via `FetchContent` if they are not already installed on your system).*

---

### 🐧 Quick Install (Ubuntu / Debian Linux)

If you are starting from a fresh Linux environment, you can install the core build tools, C++ compilers, OpenMP, and the `ccmake` interface in a single command:

```bash
# Update package lists
sudo apt update

# Install GCC, CMake, the ccmake interface, and OpenMP
sudo apt install -y build-essential cmake cmake-curses-gui libomp-dev

```

*(Note: To install the CUDA Toolkit for GPU support, it is highly recommended to follow the official instructions on the [NVIDIA Developer site](https://developer.nvidia.com/cuda-downloads) to ensure compatibility with your specific graphics drivers).*

---

## ⚡ 2. Quick Start (Terminal)

If you want to build the engine for maximum performance immediately, use the following commands from the root directory:

```bash
# 1. Configure the build (Downloads missing dependencies automatically)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/KOps_dist -DKOPS_ENABLE_FETCHCONTENT=ON

# 2. Compile using 4 CPU cores
cmake --build build -j 4

# 3. Install the executable and config files to kops_dist
cmake --install build

# 4. Run the solver
cd $HOME/KOps_dist
cp share/KOptions/inputs.yaml .
./bin/KOptions inputs.yaml

```

---

## 🎛️ 3. Advanced Configuration (Using `ccmake`)

For developers, remembering long terminal flags (`-DKOPS_ENABLE_SINGLE_PRECISION=ON`) is tedious. **`ccmake`** is a visual, terminal-based GUI that lets you toggle project settings interactively.

### How to use `ccmake` with KOptions:

**Step 1: Launch the interface**

Instead of the standard `cmake` command, run:

```bash
ccmake -B build

```

*(If the `build` folder already exists, just type `ccmake build`)*.

**Step 2: Initial Configuration**

When the screen opens, it might be empty.

* Press **`c`** to run the initial configuration. CMake will scan your system and populate the screen with options.

**Step 3: Toggle your KOPS & Hardware Settings**

You will see a list of variables. Use your **Up/Down arrow keys** to navigate.

* **Hardware Backends:** Toggle `Kokkos_ENABLE_OPENMP`, `Kokkos_ENABLE_SERIAL`, or `Kokkos_ENABLE_CUDA` to `ON`/`OFF` depending on your target system (see Section 4).
* **Hardware Architecture:** Look for variables starting with `Kokkos_ARCH_` to optimize for your specific CPU or GPU (e.g., `Kokkos_ARCH_ZEN2` or `Kokkos_ARCH_AMPERE80`) (see Section 5).
* **Floating Point Precision:** Move to `KOPS_ENABLE_SINGLE_PRECISION` and press **`Enter`** to toggle it between `ON` and `OFF` (see Section 5).
* **Optimization Flags:** Move to `CMAKE_BUILD_TYPE` and type `Debug` or `Release` (see Section 5).

**Step 4: Fixing Missing Libraries**

If you have Kokkos or YAML-CPP installed locally but CMake couldn't find them, the interface will show an error at the bottom of the screen.

* Find the `Kokkos_ROOT` and/or `yaml-cpp_ROOT` variables.
* Press **`Enter`** to edit them, type the absolute path to your installations (e.g., `/opt/kokkos` and `/usr/local/yaml-cpp`), and press **`Enter`** to save.

**Step 5: Generate and Exit**

* Press **`c`** again to confirm your new settings.
* Once everything is resolved, the option to generate will appear. Press **`g`** to generate the build files and exit.

You can now compile normally using `cmake --build build -j 4`.

---

Here is the fully finalized, professional version of **Section 4**. It seamlessly incorporates the technical reality of SIMD floating-point processing and provides the exact, step-by-step instructions for unlocking the hidden architecture flags in `ccmake`.

You can replace your entire Section 4 in the `README.md` with this block:

---

## 🖥️ 4. Targeting Hardware (CPU vs. GPU)

Kokkos acts as a bridge between your C++ code and your physical hardware. You must enable the correct **backends** and **architectures** during the CMake configuration phase, and then you can control the exact resources used during the **Run phase**.

### Phase 1: Enabling Hardware Backends (Compile Time)

You can configure exactly *how* the engine compiles by adjusting Kokkos flags in `ccmake` (or via terminal arguments):

* **Pure Procedural (Single Core):** For maximum single-core efficiency without parallel overhead.
* Set `Kokkos_ENABLE_SERIAL=ON`
* Set `Kokkos_ENABLE_OPENMP=OFF`


* **Multi-Core (OpenMP):** For running across multiple CPU cores.
* Set `Kokkos_ENABLE_OPENMP=ON`


* **NVIDIA GPU (CUDA):** For massive parallel throughput.
* Set `Kokkos_ENABLE_CUDA=ON`
* *(Note: Compiling for CUDA requires the `nvcc` compiler).*



---

### Phase 2: Tuning for Specific Architectures (Compile Time)

**Why is this necessary?**

Simply turning on "OpenMP" or "CUDA" tells the compiler to use generic parallel code. However, explicitly telling Kokkos *which* chip you are using unlocks advanced hardware instructions (like AVX2/AVX-512 for CPUs).

Instead of processing one equation at a time, these architectures use SIMD (Single Instruction, Multiple Data) to process several **floating-point numbers** in a single clock cycle. This can speed up the Monte Carlo simulations.

**How to set native architectures in `ccmake`:**

Because Kokkos supports dozens of architectures, these flags are hidden by default.

1. Run `ccmake build` in your terminal.
2. Press **`t`** to toggle **Advanced Mode** `ON`.
3. Scroll down (or press **`/`** to search) to find the `Kokkos_ARCH_...` variables.
4. Highlight your specific architecture and press **`Enter`** to toggle it to `ON`. *(Make sure all other architecture flags are `OFF`)*.

**Common CPU Examples (Intel Core i5 / i7):**

* **`Kokkos_ARCH_HSW=ON`**: (Haswell) Use this for most general Intel i5/i7 processors. It enables **AVX2** vectorization, allowing the CPU to process **4 double-precision** (or 8 single-precision) floating-point numbers at the exact same time.
* **`Kokkos_ARCH_SKX=ON`**: (Skylake) Use this for newer/higher-end Intel chips. It enables **AVX-512**, allowing the CPU to process **8 double-precision** (or 16 single-precision) floating-point numbers simultaneously.

**Common GPU Examples (NVIDIA GeForce RTX):**

* **`Kokkos_ARCH_TURING75=ON`**: For **RTX 20xx** series cards (e.g., RTX 2060, 2080).
* **`Kokkos_ARCH_AMPERE86=ON`**: For **RTX 30xx** series consumer cards (e.g., RTX 3070, 3080). *(Note: Datacenter cards like the A100 use `AMPERE80` instead).*
* **`Kokkos_ARCH_ADA89=ON`**: For **RTX 40xx** series cards (e.g., RTX 4080, 4090).

*Terminal Example (Targeting an RTX 3080 directly without ccmake):*

```bash
cmake -B build -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE86=ON
cmake --build build -j 4

```

---

### Phase 3: Specifying Cores/GPUs (Run Time)

Once compiled, KOptions includes the native Kokkos command-line parser. You do not need to recompile to change the number of active cores or switch GPUs!

**Running on Multiple Cores (If OpenMP was enabled):**

You can specify exactly how many CPU threads to use at runtime:

```bash
# Use exactly 8 cores
./bin/KOptions inputs.yaml --kokkos-threads=8

# Alternatively, use standard OpenMP environment variables:
OMP_NUM_THREADS=8 ./bin/KOptions inputs.yaml

```

*(Note: If you compiled with pure `Serial` mode, these flags are safely ignored, and the program will run procedurally on one core).*

**Running on a GPU (If CUDA was enabled):**

Kokkos will automatically find and use your GPU. If you have a multi-GPU system (e.g., a server with multiple RTX cards), you can specify which device to use:

```bash
# Run on the first GPU (Device 0)
./bin/KOptions inputs.yaml --kokkos-device-id=0

# Run on the second GPU (Device 1)
./bin/KOptions inputs.yaml --kokkos-device-id=1

```
***

## 🔬 5. Further Technical Specifications & Build Modes

### Precision Control

The engine uses a dynamic `Real` typedef.

* **Double Precision (Default):** Standard IEEE-754 64-bit float.
* **Single Precision:** Enabled via `KOPS_ENABLE_SINGLE_PRECISION=ON`. Reduces memory bandwidth by 50%, highly recommended for GPU runs where absolute precision is secondary to speed.

### Build Profiles

**Release Mode (`-DCMAKE_BUILD_TYPE=Release`)**
Tuned for maximum computational throughput.

* `-march=native`: Generates CPU-specific vectorization instructions (e.g., AVX-512).
* **Link Time Optimization (LTO):** Flattens the call stack across translation units.
* *Warning:* Binaries compiled in Release mode are highly optimized for the host machine and may not be portable to older CPU architectures.

**Debug Mode (`-DCMAKE_BUILD_TYPE=Debug`)**
Tuned for safety and diagnostics.

* Disables optimizations (`-O0`) and includes debug symbols (`-g`).
* **AddressSanitizer (ASan):** Actively monitors memory to instantly catch out-of-bounds array access and memory leaks.
* **UndefinedBehaviorSanitizer (UBSan):** Catches integer overflows and division-by-zero during stochastic calculations.
* *Warning:* Performance will be massively degraded; use only for development and testing.

---

## 6. Mathematical Framework

### The Heston SDEs
This engine simulates the asset path using the log-price formulation of the Heston model under the risk-neutral measure $\mathbb{Q}$. Let $x_t = \ln(S_t)$ be the log-price and $v_t$ be the variance:

$$dx_t = \left(r - q - \frac{1}{2}v_t\right)dt + \sqrt{v_t} d\widetilde{W}_{1,t}$$
$$dv_t = \kappa(\theta - v_t)dt + \sigma \sqrt{v_t} d\widetilde{W}_{2,t}$$

Where the Brownian motions are correlated by $\mathbb{E}[d\widetilde{W}_{1,t} d\widetilde{W}_{2,t}] = \rho dt$.

**Model Parameters:**

| Parameter | Description | Constraint | Unit of Measure |
| :---: | :--- | :---: | :--- |
| **$r$** | Risk-free interest rate | - | $1 / \text{Years}$ (Annualized) |
| **$q$** | Continuous dividend yield | - | $1 / \text{Years}$ (Annualized) |
| **$\kappa$** | Mean reversion speed of the variance | $\kappa > 0$ | $1 / \text{Years}$ |
| **$\theta$** | Long-term mean (reversion level) of the variance | $\theta > 0$ | $1 / \text{Years}$ (Annualized) |
| **$\sigma$** | Volatility of the variance (vol-of-vol) | $\sigma > 0$ | $1 / \text{Years}$ |
| **$\rho$** | Correlation (captures the leverage or skew effect) | $[-1, 1]$ | Dimensionless |
| **$v_0$** | Initial (time zero) level of the variance | $v_0 > 0$ | $1 / \text{Years}$ (Annualized) |

> **Crucial Note on Units:** Because the simulation utilizes the dimensionless log-price $x_t$, the sole driving dimension of the system is time ($T$). All rate and variance parameters must be strictly entered as **annualized** values to ensure dimensional consistency with the time step $dt$ (which is measured in fractions of a year).
### Numerical Discretization Schemes
Because the Feller condition ($2\kappa\theta > \sigma^2$) is often violated in high-volatility environments, standard discretization schemes fail as $v_t$ becomes negative. 
KOptions implements several schemes with increasing accuracy to handle this:

* **Full Truncation Euler:** A fast, standard Euler-Maruyama scheme where the variance is truncated at zero ($v^+ = \max(v, 0)$) to prevent imaginary numbers during simulation.
* **Milstein Scheme:** A higher-order scheme that improves the strong convergence rate of the diffusion components.
* **Andersen Quadratic Exponential (QE):** The industry-standard scheme that uses a probabilistic switch between a non-central chi-square approximation and an exponential distribution to guarantee strictly non-negative variance without leaking mass.

> **Reference:** The mathematical implementation and parameter definitions heavily reference *Rouah, F. D. (2013). The Heston Model and its Extensions in Matlab and C#*.


***

## ⚙️ 7. Configuration (`inputs.yaml`)

KOptions uses a YAML file to define the financial and numerical parameters of the simulation. A default template is provided in `share/KOptions/inputs.yaml`.

Here is a complete example of a valid configuration file:

```yaml
# ==========================================
# KOptions: Heston Model Configuration
# ==========================================

Model:
  Heston:
    r: 0.05       # Risk-free interest rate
    q: 0.0        # Continuous dividend yield
    k: 2.0        # Mean reversion speed of the variance (kappa)
    theta: 0.04   # Long-term mean of the variance
    sigma: 0.3    # Volatility of the variance (vol-of-vol)
    rho: -0.7     # Correlation between price and variance Brownian motions

Init:
  Price: 100.0    # Initial asset price (S0)
  Variance: 0.04  # Initial variance (v0)

Numerics:
  Scheme: Euler   # Available schemes: Euler, Milstein, AndersonQE

Time:
  T_End: 1.0      # Time to maturity (in years)
  DT: 0.005       # Time step size (dt)

MC:
  N_Paths: 100000 # Number of Monte Carlo realizations/paths

Output:
  Name_Out_File: "KOptions.out"
  Name_Log_File: "KOptions.log"
  Format: TXT     # Available formats: TXT, BIN
