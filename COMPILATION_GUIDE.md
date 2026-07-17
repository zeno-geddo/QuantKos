# KOptions: Compilation Guide

Welcome to the KOptions Compilation Guide.
This document provides a comprehensive blueprint explaining how to compile, install and launch the KOptions option pricing engine on modern multi-core CPU and GPU systems.

> ⚠️ **Important Note:** KOptions is designed to be portable across all platforms supported by **Kokkos**. In principle, it should compile and run on **Linux**, **macOS**, and **Windows**, using any CPU architecture, compiler, and GPU backend supported by Kokkos.
>
> Kokkos currently supports:
>
> * **Operating systems:** Linux, macOS, and Windows.
> * **CPU architectures:** x86-64 (Intel/AMD), ARM/AArch64 (including Apple Silicon and ARM-based HPC systems), IBM POWER, and other architectures supported by the selected compiler.
> * **Compilers:** GCC, Clang/LLVM, Intel oneAPI (icpx), NVIDIA HPC SDK (nvc++), Microsoft Visual Studio (MSVC), and other compilers officially supported by Kokkos.
> * **GPU backends:** NVIDIA GPUs (CUDA), AMD GPUs (HIP/ROCm), Intel GPUs (SYCL/oneAPI), and any additional accelerator backends supported by the installed Kokkos version (e.g., OpenMP Target where available).
>
> At present, however, KOptions has been developed and extensively tested only on a **Linux** system using **GCC** on an **Intel Core i7** CPU (using both OpemMP or procedural mode) and an **NVIDIA RTX** GPU (CUDA backend). We therefore expect this configuration to work reliably. Other supported platforms should also work in principle, but they have not yet been thoroughly tested and may still expose unknown bugs or portability issues.

---
## 1. Prerequisites

This section outlines the requirements and toolchains necessary to compile, install, and run the KOptions option pricing engine.

### Part I: Mandatory Requirements
These components are strictly required to build and execute the KOptions engine on target hardware.

#### A. Core Build Tools
* **CMake** (v3.20 or higher): The build orchestrator.
* **C++20 or C++23 Compiler** ([GCC 10+](https://gcc.gnu.org/) or [Clang 12+](https://clang.llvm.org/)).

#### B. Hardware Toolchains (Target-Dependent)
* **OpenMP**: Required if compiling for multi-core CPU thread-level parallelism.
* **NVIDIA CUDA Toolkit**: Provides the `nvcc` compiler, required if compiling for GPU-accelerated execution on NVIDIA hardware.

#### C. Dependencies
* **Kokkos**: The performance-portability framework.
* **yaml-cpp**: For parsing configuration files.

> ⚠️ **Note:** The CMake build system can automatically configure, download, and statically build Kokkos and yaml-cpp via `FetchContent` if they are not detected locally on your host. More will be said about that when addressing KOptions compilation and installation.


### Part II: Recommended Development & Profiling Tools
While not required to execute standard pricing runs, these tools are highly recommended for active development, codebase maintenance, and performance optimization.

#### A. Building Tool
* **ccmake**: terminal GUI for easier and more advanced build configuration.

#### B. Documentation Engine
* **Doxygen & Graphviz**: Automated tools to parse C++ source annotations and CMake configurations to generate hyperlinked HTML (and Latex) documentation with architecture graphs.

#### C. GPU Profiling Suite (NVIDIA Nsight)
Used to identify hardware-level bottlenecks in the code:

* **NVIDIA Nsight Systems (`nsys`)**: A system-wide profiler. Provides an interactive timeline of CPU thread activity, OS events, CUDA API calls, and PCIe bus memory transfers. Important for diagnosing Host-Device synchronization stalls (`Kokkos::fence`).
* **NVIDIA Nsight Compute (`ncu`)**: A deep-dive CUDA kernel profiler. Provides hardware metrics such as warp occupancy, register pressure, memory access patterns, and cache hit rates. Important for fine-tuning SDE integration loops.

***

## 🐧 2.  Installing the System level Prerequisites on Linux

This step-by-step guide walks you through setting up your environment on a fresh Linux instance (Ubuntu / Debian).
Although KOptions could work with different GPUs brands, we will here focus on setting up machines with NVIDIA GPUs only.

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
This is optimal to deploy KOptions and compare its performances under different hardware conditions. Of course, if you are interested in running KOptions only on a specific hardware, stick with it and avoid compiling for others.  

Anyway, despite your hardware choice, have to create a unique build directory for the target you want, execute the corresponding `cmake` command from the examples below, and then compile and install:

```bash
# Example for a specific build target
mkdir -p ~/software/build/kokkos-single
cd ~/software/build/kokkos-single

# [Run the appropriate CMake command from below]

# Build and Install
make -j$(nproc) # you can specify the number of cores used to build
make install

```

#### Option A: Procedural Single Core (Release)

From your target build directory, use this for procedural CPU compilation without multi-threading.

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
> ⚠️ **Note on CPU threads:** You can specify exactly how many CPU threads to use at runtime. Fon instance, running OMP_NUM_THREADS=8 before launching KOptions, will make it run in parallel with 8 threads. More about that will be said later.


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
> ⚠️ **Note on GPU Architectures:** Change `AMPERE86` to match your specific GPU architecture, e.g., `ADA89` for RTX 40-series, `TURING75` for RTX 20-series.



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

# 3. Install KOptions

This section explains how to build and install the KOptions engine itself. Two workflows are supported:

* **Option A (Recommended for developers):** Use pre-installed local versions of **Kokkos** and **yaml-cpp**. This provides full control over the hardware backends and avoids recompiling dependencies every time.
* **Option B (Recommended for first-time users):** Let CMake automatically download and build the required dependencies using `FetchContent`.

In both cases, KOptions is built using the standard CMake workflow.

---

## Step 1: Download KOptions

Clone the repository into your preferred source directory:

```bash
mkdir -p ~/software/src
cd ~/software/src

git clone https://github.com/zeno.geddo/KOptions.git
cd KOptions
```



As for Kokkos, it is recommended (though not required) to keep the **source**, **build**, and **installation** directories separated. Besides keeping the project organized, this approach allows you to generate multiple independent KOptions builds from the same source tree—for example, a procedural (Serial) version, an OpenMP version, a CUDA-enabled version, or separate Debug and Release builds—without the different configurations interfering with one another.

```text
~/software/
├── src/
│   └── KOptions/
├── build/
│   ├── KOptions-serial/
│   ├── KOptions-openmp/
│   ├── KOptions-cuda/
│   └── KOptions-debug/
└── installations/
    ├── KOptions-serial/
    ├── KOptions-openmp/
    ├── KOptions-cuda/
    └── KOptions-debug/
```

For example, if you want to build the CUDA version, simply create a dedicated build directory:

```bash
mkdir -p ~/software/build/KOptions-cuda
cd ~/software/build/KOptions-cuda
```

Later, you could similarly create `KOptions-openmp` or `KOptions-serial` build directories if you wish to compare performance across different hardware backends.

Of course, if you are just targeting a specific hardware and you do not need to run comparisons or benchmarks etc., just compile for that target.


## Step 2: Build KOptions

KOptions can either link against an existing installation of **Kokkos** and **yaml-cpp**, or automatically download and build these dependencies using `FetchContent`.

---

### Option A: Build without using FetchContent

This approach assumes that **Kokkos** and **yaml-cpp** have already been built and installed as described in Section 2.

First, enter the build directory corresponding to the version you want to generate. For example, to build the CUDA version:

```bash
cd ~/software/build/KOptions-cuda
```

Then configure the project by pointing CMake to the **KOptions source directory** and to the installation directories of the required dependencies:


```bash
cmake \
  $HOME/software/src/KOptions \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$HOME/software/installations/KOptions-cuda \
  -DKOPS_ENABLE_FETCHCONTENT=OFF \
  -DKOPS_BUILD_DOC=ON \
  -DKOPS_ENABLE_SINGLE_PRECISION=OFF \
  -DKOPS_ENABLE_TESTS=ON \
  -DKokkos_ROOT=$HOME/software/installations/kokkos-5/cuda/lib/cmake/Kokkos \
  -Dyaml-cpp_ROOT=$HOME/software/installations/yaml-cpp/lib/cmake/yaml-cpp
```

> **Additional important KOptions configuration options**
>
> Besides the standard CMake and dependency settings, KOptions provides several project-specific options:
>
> * **`KOPS_BUILD_TYPE=Release`** *(recommended for production)*: Compiles the project with compiler optimizations enabled and disables most debugging features. This produces the fastest executable and is the recommended configuration for production simulations and performance benchmarking. 
> Alternatively, you can use the flag `Debug`, which builds the project with full debugging information no optimization. This configuration is intended for development, debugging with tools such as gdb, and investigating runtime errors. 
> Otherwise, the flag `RelWithDebInfo` provides a compromise between performance and debuggability by enabling compiler optimizations while retaining debugging symbols. This is often the preferred choice when profiling or diagnosing problems that only appear in optimized builds.
>
> * **`KOPS_BUILD_DOC=ON`** *(recommended)*: Automatically generates the KOptions API documentation with **Doxygen** during the build process (provided Doxygen is installed on the system).
>
> * **`KOPS_ENABLE_SINGLE_PRECISION=ON`**: Compiles the engine using 32-bit `float` arithmetic instead of the default 64-bit `double`. This generally improves memory efficiency and can significantly increase performance—especially on GPUs—but at the cost of reduced numerical precision.
>
> * **`KOPS_ENABLE_TESTS=ON`**: Builds the KOptions test executables together with the main application. This option is recommended for development or when verifying a new installation, as it allows the built-in test suite to be executed after compilation.


The dependency paths deserve particular attention:

* **`Kokkos_ROOT`** must point to the directory containing the `KokkosConfig.cmake` file. This directory is typically located at

  ```text
  <kokkos-install-prefix>/lib/cmake/Kokkos
  ```

  For example:

  ```text
  ~/software/installations/kokkos-5/cuda/lib/cmake/Kokkos
  ```

* **`yaml-cpp_ROOT`** must similarly point to the directory containing `yaml-cpp-config.cmake` (or `yaml-cppConfig.cmake`), which is usually

  ```text
  <yaml-cpp-install-prefix>/lib/cmake/yaml-cpp
  ```

If CMake cannot locate either package, verify that these directories actually contain the corresponding `*.cmake` configuration files.

> **Note:** Because the selected Kokkos installation has already been compiled with its desired hardware backend (Serial, OpenMP, CUDA, HIP, etc.) and architecture, no additional Kokkos configuration flags are required when building KOptions. KOptions will simply link against the existing Kokkos installation.

Once the configuration completes successfully, compile the project from the same build directory:

```bash
cmake --build . -j$(nproc)
```

You may replace `$(nproc)` with any desired number of compilation threads. For example,

```bash
cmake --build . -j4
```
uses four CPU cores to build the project.




### Option B: Build using FetchContent

This approach is probably recommended if **Kokkos** and **yaml-cpp** are not already installed on your system and you just want to use KOptions with developing and performing benchmarks. Note that, in this case, during the configuration step, CMake automatically downloads, configures, builds, and links both libraries.

As for the previous option, first move to the build directory corresponding to the version you want to generate. For example, for a CUDA build:

```bash
cd ~/software/build/KOptions-cuda
```

Then configure the project by pointing CMake to the KOptions source directory and enabling `FetchContent`:

```bash
cmake \
  $HOME/software/src/KOptions \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$HOME/software/installations/KOptions-cuda \
  -DKOPS_ENABLE_FETCHCONTENT=ON \
  -DKOPS_BUILD_DOC=ON \
  -DKOPS_ENABLE_SINGLE_PRECISION=OFF \
  -DKOPS_ENABLE_TESTS=ON 
```

Unlike the previous approach, Kokkos is now built as part of the KOptions compilation. Consequently, all Kokkos configuration options become available and can be specified directly on the CMake command line.

The most important options affecting performance are:

* **Build profile**

    * `-DCMAKE_BUILD_TYPE=Release` (**recommended**) enables compiler optimizations and should be used for production runs and benchmarks.
    * `Debug` disables most optimizations and enables additional runtime checks, making it suitable only for development.
    * `RelWithDebInfo` provides nearly the same optimizations as `Release` while preserving debugging symbols.

* **Execution backend**

    * `-DKokkos_ENABLE_SERIAL=ON` enables procedural execution.
    * `-DKokkos_ENABLE_OPENMP=ON` enables multi-core CPU execution.
    * `-DKokkos_ENABLE_CUDA=ON` enables execution on NVIDIA GPUs.

* **Target architecture**

    * Select the architecture matching your hardware (see Section 2), for example
      `-DKokkos_ARCH_AMPERE86=ON` for an RTX 30-series GPU or
      `-DKokkos_ARCH_ADA89=ON` for an RTX 40-series GPU.
    * On CPU-only systems, enabling `-DKokkos_ARCH_NATIVE=ON` is generally recommended, as it allows the compiler to optimize for the local processor.

For example, an optimized OpenMP build can be configured with

```bash
cmake \
  $HOME/software/src/KOptions \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$HOME/software/installations/KOptions-openmp \
  -DKOPS_ENABLE_FETCHCONTENT=ON \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_OPENMP=ON \
  -DKokkos_ENABLE_CUDA=OFF \
  -DKokkos_ARCH_NATIVE=ON \
  -DKOPS_BUILD_DOC=ON \
  -DKOPS_ENABLE_SINGLE_PRECISION=OFF \
  -DKOPS_ENABLE_TESTS=ON 
```

while an optimized CUDA build for an RTX 30-series GPU becomes

```bash
cmake \
  $HOME/software/src/KOptions \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$HOME/software/installations/KOptions-cuda \
  -DKOPS_ENABLE_FETCHCONTENT=ON \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_OPENMP=OFF \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ENABLE_CUDA_LAMBDA=ON \
  -DKokkos_ARCH_AMPERE86=ON \
  -DKOPS_BUILD_DOC=ON \
  -DKOPS_ENABLE_SINGLE_PRECISION=OFF \
  -DKOPS_ENABLE_TESTS=ON 
  
```

The complete list of available Kokkos backends and architecture flags is described in **Section 2**.

Once the configuration completes successfully, compile the project from the same build directory:

```bash
cmake --build . -j$(nproc)
```

or specify any desired number of compilation threads, for example

```bash
cmake --build . -j4
```


---

## Step 3: Install KOptions

After compilation completes successfully, install the executable and the accompanying configuration files:

```bash
cmake --install .
```

If `CMAKE_INSTALL_PREFIX` was set to

```text
$HOME/software/installations/KOptions
```

the resulting installation will have the following layout:

```text
KOptions/
├── bin/
│   ├── KOptions
│   └── Tester
├── share/
│    ├── documentation
│    │        ├── html
│    │        └── latex
│    ├── ...
│    └── KOptionsConfig.yalm
└── ...
```

You can then copy the example configuration file into your working directory and launch the engine:

```bash
cd $HOME/software/installations/KOptions

./bin/KOptions share/KOptionsConfig.yaml
```


## About the Documentation Generated

If Doxygen and Graphviz are installed (Section 1), the project can generate a complete set of API and architecture documentation.

By enabling the documentation during configuration:

```bash
cmake \
  $HOME/software/src/KOptions \
  -DKOPS_BUILD_DOCUMENTATION=ON
```

Doxygen automatically extracts the documentation embedded in the source code and generates both **HTML** and **LaTeX** documentation. The documentation describes the software architecture, class hierarchies, namespaces, source file organization, and the public API of the library. When Graphviz is available, additional inheritance, collaboration, and call graphs are included.

The HTML documentation can be viewed by opening the `index.html` file located in the generated `html/` directory with any web browser, for example:

```bash
xdg-open html/index.html
```

(on Linux) or by opening the file directly from your preferred browser.

The LaTeX documentation is written to the generated `latex/` directory. It can be compiled into a PDF by running:

```bash
cd latex
make
```

which invokes `pdflatex` (and related tools) to produce the final PDF document, typically named `refman.pdf`.

***

## 5. Run KOptions

### Run a MonteCarlo Simulation to Price an Option


### Running the build-in tests 


### Specifying Cores/GPUs (Run Time)

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


***




