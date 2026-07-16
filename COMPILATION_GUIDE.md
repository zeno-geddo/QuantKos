# KOptions: Compilation Guide

Welcome to the KOptions Compilation Guide.
This document provides a comprehensive blueprint explaining how to compile, install, launch, and optimize the KOptions option pricing engine on modern multi-core CPU and GPU systems.

---
## 1. Prerequisites

This section outlines the requirements and toolchains necessary to compile, install, and run the KOptions option pricing engine.

### Part I: Mandatory Requirements
These components are strictly required to build and execute the KOptions engine on target hardware.

#### A. Core Build Tools
* **CMake** (v3.20 or higher): The build orchestrator.
* **C++20 Compiler** ([GCC 10+](https://gcc.gnu.org/) or [Clang 12+](https://clang.llvm.org/)): Required for modern standard library concepts and parallel code-generation rules.

#### B. Hardware Toolchains (Target-Dependent)
* **OpenMP**: Required if compiling for multi-core CPU thread-level parallelism.
* **NVIDIA CUDA Toolkit**: Provides the `nvcc` compiler, required if compiling for GPU-accelerated execution on NVIDIA hardware.

#### C. Dependencies (Automatic Fallbacks)
* **Kokkos**: The performance-portability framework.
* **yaml-cpp**: For parsing configuration files.

*(Note: The CMake build system can automatically configure, download, and statically build Kokkos and yaml-cpp via `FetchContent` if they are not detected locally on your host).*


### Part II: Recommended Development & Profiling Tools
While not required to execute standard pricing runs, these tools are highly recommended for active development, codebase maintenance, and performance optimization.

#### A. Documentation Engine
* **Doxygen & Graphviz**: Automated tools to parse C++ source annotations and CMake configurations to generate hyperlinked HTML (and Latex) documentation with architecture graphs.

#### B. GPU Profiling Suite (NVIDIA Nsight)
Used to identify hardware-level bottlenecks in the code:

* **NVIDIA Nsight Systems (`nsys`)**: A system-wide profiler. Provides an interactive timeline of CPU thread activity, OS events, CUDA API calls, and PCIe bus memory transfers. Important for diagnosing Host-Device synchronization stalls (`Kokkos::fence`).
* **NVIDIA Nsight Compute (`ncu`)**: A deep-dive CUDA kernel profiler. Provides hardware metrics such as warp occupancy, register pressure, memory access patterns, and cache hit rates. Important for fine-tuning SDE integration loops.

***

## 🐧 2.  Installing the System level Prerequisites on Linux

This step-by-step guide walks you through setting up your environment on a fresh Linux instance (Ubuntu / Debian).

### Step 1 (Required): Install Core Utilities & Documentation Tools

The baseline compilers, build generators, multi-threading libraries, and documentation utilities can be installed cleanly from the default package manager:

```bash
# 1. Update system package lists
sudo apt update && upgrade

# 2. Install base compilers, CMake, ccmake interface, OpenMP, and Doxygen (with Graphviz)
sudo apt install -y build-essential cmake cmake-curses-gui libomp-dev doxygen graphviz

```

### Step 2 (Required): Install GPU Toolchain (CUDA Compiler & SDK)

To execute your Kokkos solver on an NVIDIA GPU, you need the CUDA Toolkit.

**Why `nvcc` is Essential.**
The `nvcc` (NVIDIA CUDA Compiler) is the specific tool inside that toolkit that compiles C++ code into machine code that your GPU can understand.
Your Kokkos code is "Single Source," meaning it contains both CPU logic (Host) and GPU logic (Device) in the same file:

* The CPU compiler (e.g., `g++`) handles standard host-side C++ logic (I/O, orchestrators).
* The GPU compiler (`nvcc`) extracts the parallel kernels, compiles them for the targeted NVIDIA architecture, and links them back to the host binary.
* Kokkos uses a wrapper script (`nvcc_wrapper`) to coordinate this dual-compilation pipeline, which requires `nvcc` to be registered in your system's `PATH`.

**Recommended Installation Process.**
*Note: To prevent library mismatch errors and maintain system stability, it is always best to follow the official instructions on the NVIDIA Developer site. Avoid using `apt install nvidia-cuda-toolkit` directly from default Ubuntu repositories, as they are often outdated and may conflict with modern C++ standard requirements.*

If you are setting up on Ubuntu 22.04, the following sequence could be helpfull for you:

```bash
# 1. Download the repository pin file to prioritize NVIDIA's packages
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600

# 2. Download the local repository installer debian package
wget https://developer.download.nvidia.com/compute/cuda/13.1.1/local_installers/cuda-repo-ubuntu2204-13-1-local_13.1.1-590.48.01-1_amd64.deb

# 3. Register the local repository with package management
sudo dpkg -i cuda-repo-ubuntu2204-13-1-local_13.1.1-590.48.01-1_amd64.deb

# 4. Copy the repository keyring to system keyrings
sudo cp /var/cuda-repo-ubuntu2204-13-1-local/cuda-*-keyring.gpg /usr/share/keyrings/

# 5. Synchronize package indexes and install the SDK/Compiler Toolkit
sudo apt-get update
sudo apt-get -y install cuda-toolkit-13-1

```

**Crucial Environment Variables Configuration.**
Installing the toolkit places binary files in `/usr/local/cuda`. However, you must manually update your shell configuration to make it visible to Kokkos and CMake.
Append these environment mappings to the bottom of your shell configuration file (e.g., `~/.bashrc` or `~/.zshrc`):

```bash
# NVIDIA CUDA Toolkit Path Configuration
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

```

Apply the changes immediately to your active terminal:

```bash
source ~/.bashrc

```

Verify that the compiler is active and query its configuration:

```bash
nvcc --version

```

### Step 3 (Optional): Install Standalone GPU Profiling Tools

NVIDIA Nsight Systems (`nsys`) and Nsight Compute (`ncu`) are automatically bundled inside the full CUDA Toolkit installation.
However, on headless remote profiling servers, virtual staging partitions, or benchmark runtimes where a compiler is not needed but performance evaluation is required, you can install the profiling utilities as lightweight, standalone packages:

```bash
# Install only the standalone profiling suites (Ensure NVIDIA repos are configured first)
sudo apt update
sudo apt install -y nsight-systems nsight-compute
```

### Step 4 (Recommended) : Dowload, Build and Install Kokkos 5
In a clean development environment, you should never build software directly inside the source folder. The best approach is to create a dedicated software directory layout to keep your raw materials separate from your finished, installed libraries.

#### 1. Recommended Directory Structure

It is highly recommended to establish the following structure:

* **Source (`src`):** `~/software/src/kokkos-5`
* **Build (`build`):** `~/software/build/kokkos-cuda` (etc.)
* **Install (`install`):** `~/software/installations/kokkos-5/`

First, clone the Kokkos repository into your source folder:

```bash
mkdir -p ~/software/src
cd ~/software/src
git clone [https://github.com/kokkos/kokkos.git](https://github.com/kokkos/kokkos.git) kokkos-5
```

#### 2. Isolated Hardware Builds

To ensure maximum flexibility, it is best to compile Kokkos in several separated ways. 
For example, building a CUDA version, an OpenMP version, and a procedural single-core version into distinct installation directories.
This is optimal to deploy KOptions and compare its performances on different hardwares. Of course, if you are interested in running KOptions only on an hardware, stick with it and avoid to compile for others.  

Anyway, create a unique build directory for the target you want, execute the corresponding `cmake` command from the examples below, and then compile and install:

```bash
# Example for a specific build target
mkdir -p ~/software/build/kokkos-single
cd ~/software/build/kokkos-single

# [Run the appropriate CMake command from below]

# Build and Install
make -j$(nproc)
make install

```

#### Option A: Procedural Single Core (Release)

From your target build directory, use this for maximum procedural CPU compilation without multi-threading.

```bash
cmake $HOME/software/src/kokkos-5 \
  -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_CXX_STANDARD=20 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$HOME/software/installations/kokkos-5/single_core \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_OPENMP=OFF \
  -DKokkos_ENABLE_CUDA=OFF \
  -DKokkos_ARCH_NATIVE=ON \
  -DKokkos_ENABLE_DEBUG=OFF \
  -DKokkos_ENABLE_DEBUG_BOUNDS_CHECK=OFF

```

#### Option B: Multi-Core CPU (OpenMP / Release)

From your target build directory, use this to utilize all available physical cores on your CPU via OpenMP thread-level parallelism.

```bash
cmake $HOME/software/src/kokkos-5 \
  -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_CXX_STANDARD=20 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$HOME/software/installations/kokkos-5/openmp \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_OPENMP=ON \
  -DKokkos_ENABLE_CUDA=OFF \
  -DKokkos_ARCH_NATIVE=ON \
  -DKokkos_ENABLE_DEBUG=OFF \
  -DKokkos_ENABLE_DEBUG_BOUNDS_CHECK=OFF

```

#### Option C: GPU Accelerated (CUDA / Release)

From your target build directory, use this for massive parallel throughput on NVIDIA GPUs.

```bash
cmake $HOME/software/src/kokkos-5 \   
  -DCMAKE_CXX_COMPILER=$HOME/software/src/kokkos-5/bin/nvcc_wrapper \
  -DCMAKE_CXX_STANDARD=20 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$HOME/software/installations/kokkos-5/cuda \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_OPENMP=OFF \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ENABLE_CUDA_LAMBDA=ON \
  -DKokkos_ARCH_NATIVE=ON \
  -DKokkos_ARCH_AMPERE86=ON \
  -DKokkos_ENABLE_DEBUG=OFF \
  -DKokkos_ENABLE_DEBUG_BOUNDS_CHECK=OFF

```

*(Note: Change `AMPERE86` to match your specific GPU architecture, e.g., `ADA89` for RTX 40-series, `TURING75` for RTX 20-series).*



#### 3. Targeting Your Specific GPU Architecture

Kokkos relies on compile-time architecture flags to compile GPU device code directly into optimized machine instructions (SASS), avoiding heavy runtime JIT-compilation penalties. If you do not explicitly declare your GPU architecture, the build will likely generate a configuration error.

#### Step 3.1: Query Your GPU Compute Capability

To find your hardware's exact architectural generation, query the NVIDIA driver by running this command in your terminal:

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv

```

#### Understanding the Output:

* **Success Case:** The terminal will return your card name followed by its Compute Capability version decimal, for example:
```csv
name, compute_cap
NVIDIA GeForce RTX 3060 Laptop GPU, 8.6

```


* **Failure Case:** If the system returns `"command not found"` or cannot communicate with the driver, your NVIDIA kernel modules are not active or the drivers require a system update.

#### Step 3.2: Match Your Compute Capability to the Kokkos Flag

Find the major/minor compute version returned in Step 1 and append its corresponding CMake flag to your build configuration:

| Compute Capability | GPU Generation | Representative Hardware | Required Kokkos CMake Flag |
| --- | --- | --- | --- |
| **9.0** | Hopper | NVIDIA H100 | `-DKokkos_ARCH_HOPPER90=ON` |
| **8.9** | Ada Lovelace | RTX 4090, 4080, 4070, L40 | `-DKokkos_ARCH_ADA89=ON` |
| **8.6** | Ampere (Consumer) | RTX 3090, 3080, 3070, 3060, A6000 | `-DKokkos_ARCH_AMPERE86=ON` |
| **8.0** | Ampere (Enterprise) | NVIDIA A100, A30 | `-DKokkos_ARCH_AMPERE80=ON` |
| **7.5** | Turing | RTX 2080, 2070, T4, Quadro RTX | `-DKokkos_ARCH_TURING75=ON` |
| **7.0** | Volta | NVIDIA V100, Titan V | `-DKokkos_ARCH_VOLTA70=ON` |

> ⚠️ **Note on Legacy Architectures:** If you have an older card (such as Pascal 6.0/6.1 like the GTX 1080), be aware that CUDA 13.x has deprecated support for these chips. To compile for Pascal generation targets, you must downgrade your environment to the CUDA 12.x toolkit family.


### Step 4 (Recommended): Download, Build and Install YALM
k

k

k

k

k

k

k
k

k

kk

TO BE DONE

***

## 3. Install KOptions

### Step 1 : Download KOptions

### Step 2 : Build KOptions 

### Option A : Build without using Fetchcontent

### Option B : Build using Fetchcontent

### Step 3 : Install KOptions

### Recommended : Generate KOptions Documentation

***

## 5. Run KOptions

***



