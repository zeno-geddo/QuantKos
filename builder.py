#  Copyright (C) 14/07/2026 Zeno GEDDO <zeno.geddo@gmail.com>
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

# !/usr/bin/python3

import os
import time
import subprocess
import sys
from typing import Dict, List, Optional
from enum import Enum


class QuantKosBuilder:
    """!
    @brief Builder class to configure, compile, and install the QuantKos option pricing engine.

    @details This class manages the CMake build process, allowing compilation against
             either pre-installed local dependencies or automatic downloads via FetchContent.
             It abstracts away the generation of CMake flags based on the desired hardware
             backend, precision, and math mode.
    @todo Should add the possibility of activate tests
    @todo Check and update gpu arch list
    """

    class ImplementedBackends(Enum):
        SERIAL = 1
        OPENMP = 2
        CUDA = 3
        HIP = 4  # Support to AMD GPUs

    class ImplementedPrecision(Enum):
        SINGLE = 1
        DOUBLE = 2

    class ImplementedMath(Enum):
        IEEE = 1
        FAST = 2

    class ImplementedBuildType(Enum):
        Release = 1
        Debug = 2
        RelWithDebInfo = 3

    class ImplementedNvidiaGPUArch(Enum):
        NONE = 0  # Do not pass any architecture flag
        AUTO = 1  # Automatically detect via nvidia-smi

        # --- Data Center & Supercomputing ---
        BLACKWELL100 = 2  # sm_100 / CC 10.0
        BLACKWELL103 = 3  # sm_103 / CC 10.3
        BLACKWELL120 = 4  # sm_120 / CC 12.0
        BLACKWELL121 = 5  # sm_121 / CC 12.1 # Compute 10.0 / 12.0: B200 / GB200 / B100
        HOPPER90 = 6  # Compute 9.0: H100 / H200 / GH200
        AMPERE80 = 7  # Compute 8.0: A100
        VOLTA70 = 8  # Compute 7.0: V100
        PASCAL60 = 9  # Compute 6.0: P100

        # --- Workstation & Consumer (Ada Lovelace, Ampere, Turing, Pascal) ---
        ADA89 = 10  # Compute 8.9: RTX 40-series / L40 / L4
        AMPERE86 = 11  # Compute 8.6: RTX 30-series / A6000 / A40
        TURING75 = 12  # Compute 7.5: RTX 20-series / T4 / Titan RTX
        PASCAL61 = 13  # Compute 6.1: GTX 10-series / P40 / P4

    class ImplementedAmdGPUArch(Enum):
        NONE = 0
        AUTO = 1  # Automatically detect via rocminfo
        # --- CDNA Series (Data Center & Supercomputing) ---
        MI300X = 2  # gfx942: AMD Instinct MI300X
        MI300A = 3  # gfx940: AMD Instinct MI300A APU
        VEGA90A = 4  # gfx90a: AMD Instinct MI200 Series (MI250X / MI250 / MI210)
        VEGA908 = 5  # gfx908: AMD Instinct MI100
        VEGA906 = 6  # gfx906: AMD Instinct MI50 / Radeon VII
        # --- RDNA3 Series (Gaming & Workstation) ---
        NAVI31 = 7  # gfx1100: Radeon RX 7900 XTX / 7900 XT / PRO W7900
        NAVI32 = 8  # gfx1101: Radeon RX 7700 XT / 7800 XT / PRO W7800
        NAVI33 = 9  # gfx1102: Radeon RX 7600
        # --- RDNA2 Series ---
        NAVI21 = 10  # gfx1030: Radeon RX 6800 / 6900 XT / PRO W6800
        NAVI22 = 11  # gfx1031: Radeon RX 6700 XT
        # --- Legacy / Workstation ---
        VEGA900 = 12  # gfx900: Vega 56 / 64

    def __init__(self,
                 QuantoKos_root_dir: str,
                 workspace_root_dir: str,
                 build_type: str = ImplementedBuildType.Release.name,
                 build_tests_also: bool = False,
                 use_fetchcontent: bool = False,
                 kokkos_paths: Optional[Dict[str, str]] = None,
                 yaml_cpp_path: Optional[str] = None,
                 gpu_arch: Dict[str, str] = {
                     ImplementedBackends.CUDA.name: ImplementedNvidiaGPUArch.AUTO.name,
                     ImplementedBackends.HIP.name: ImplementedAmdGPUArch.AUTO.name,
                 }
                 ) -> None:
        """!
        @brief Constructor for the QuantKosBuilder.

        @param QuantoKos_root_dir    Path to the root of the QuantKos root directory.
        @param workspace_root_dir    Path where 'build' and 'installations' directories will be created.
        @param build_tests_also      If True, compiles also the test suite.
        @param use_fetchcontent      If True, uses CMake FetchContent to download and build dependencies automatically.
                                     If False, links against local pre-installed dependencies.
        @param kokkos_paths          A dictionary mapping target_backends (e.g., 'SERIAL', 'OPENMP', 'CUDA') to their local Kokkos installation paths.
        @param yaml_cpp_path         Path to the local yaml-cpp installation directory.
        @param build_type            CMake build type: 'Release', 'Debug', or 'RelWithDebInfo'.
        @param gpu_arch              Kokkos GPU target architecture for different manufacturers (e.g., CUDA:'AMPERE86', HIP:'VEGA90A').
        """
        # Paths required
        self.src_root: str = os.path.abspath(QuantoKos_root_dir)
        self.workspace_root: str = os.path.abspath(workspace_root_dir)
        self.build_root: str = os.path.join(self.workspace_root, 'builds')
        self.install_root: str = os.path.join(self.workspace_root, 'installations')

        # Fetchcontent & Build parameters
        self.use_fetchcontent: bool = use_fetchcontent
        self.build_tests_also: str = build_tests_also
        self.build_type: str = build_type
        self.gpu_arch: Dict[str, str] = gpu_arch

        # Paths only needed if not using Fetchcontent
        self.yaml_cpp_path: Optional[str] = yaml_cpp_path  # Path to local yaml-cpp install
        self.kokkos_paths: Optional[
            Dict[str, str]] = kokkos_paths  # Map paradigms to their local pre-built Kokkos installation paths

        # String containing the outputs
        self.out_str = ""

    @classmethod
    def from_interactive(cls) -> "QuantKosBuilder":
        """!
        @brief Interactively prompts the user in the terminal with input validation and exit capabilities.
        @return A fully initialized instance of QuantKosBuilder.
        """
        inp_msg = (f" QuantKos Builder - Interactive Configuration\n"
                   f"{'-' * 60}\n"
                   f"   Author            : Zeno GEDDO\n"
                   f"   Build Year        : 2026\n"
                   f"   License           : GNU GENERAL PUBLIC LICENSE (V.3, 29/06/2007)\n"
                   f"   Contact           : zeno.geddo@gmail.com"
                   )
        print("\n\n")
        print("=" * 60)
        print(inp_msg)
        print("=" * 60)

        while True:
            print("\n\n(Type 'exit' or 'q' at any prompt to cancel)")
            print("(Press enter to chose the default. The default is printed inside [] in the query)")
            try:
                # QuantKos Root Directory (must exist)
                default_src_root = os.path.abspath(os.path.dirname(__file__))
                src_root = cls._get_valid_directory(
                    prompt_text="\n> QuantKos Root Directory",
                    default_path=default_src_root,
                    must_exist=True
                )

                # Workspace Directory (can be created but never destroyed)
                default_workspace = os.path.abspath(os.path.join(src_root, '..', 'TempBuildsAndInstalls'))
                workspace_root = cls._get_valid_directory(
                    prompt_text="\n> Workspace Directory",
                    default_path=default_workspace,
                    must_exist=False
                )

                # Build Type selection
                build_type = cls._get_enum_choice(
                    prompt_text="\n> CMake Build Type",
                    enum_cls=cls.ImplementedBuildType,
                    default_name=cls.ImplementedBuildType.Release.name
                )

                # FetchContent option
                build_tests_also = cls._get_boolean_choice(
                    prompt_text="\n> Compile and install also test suite?",
                    default_to_yes=False)

                # FetchContent option
                use_fetchcontent = cls._get_boolean_choice(
                    prompt_text="\n> Use FetchContent to automatically download dependencies?",
                    default_to_yes=True)

                # GPU Architecture (for FetchContent builds)
                gpu_arch = {}
                if use_fetchcontent:

                    # CUDA GPU
                    selected_arch : str = cls._get_enum_choice(
                        prompt_text="\n> Target NVIDIA GPU Architecture (Kokkos Flag)",
                        enum_cls=cls.ImplementedNvidiaGPUArch,
                        default_name=cls.ImplementedNvidiaGPUArch.AUTO.name
                    )
                    if selected_arch == cls.ImplementedNvidiaGPUArch.AUTO.name:
                        gpu_arch[cls.ImplementedBackends.CUDA.name] = cls._auto_detect_nvidia_arch()
                    elif selected_arch == cls.ImplementedNvidiaGPUArch.NONE.name:
                        gpu_arch[cls.ImplementedBackends.CUDA.name] = None
                    else:
                        gpu_arch[cls.ImplementedBackends.CUDA.name] = selected_arch

                    # AMD GPU
                    selected_arch : str = cls._get_enum_choice(
                        prompt_text="\n> Target AMD GPU Architecture (Kokkos Flag)",
                        enum_cls=cls.ImplementedAmdGPUArch,
                        default_name=cls.ImplementedAmdGPUArch.AUTO.name
                    )
                    if selected_arch == cls.ImplementedAmdGPUArch.AUTO.name:
                        gpu_arch[cls.ImplementedBackends.HIP.name] = cls._auto_detect_amd_arch()
                    elif selected_arch == cls.ImplementedAmdGPUArch.NONE.name:
                        gpu_arch[cls.ImplementedBackends.HIP.name] = None
                    else:
                        gpu_arch[cls.ImplementedBackends.HIP.name] = selected_arch

                # Dependency paths if not using FetchContent
                yaml_cpp_path = None
                kokkos_paths = None
                if not use_fetchcontent:
                    default_install = os.path.abspath(os.path.expanduser('~/installations'))

                    # yaml-cpp CMake Config Path
                    print("\n--- Validating Local Dependency Paths ---")
                    default_yaml_cpp_path = os.path.join(default_install, 'yaml-cpp', 'lib', 'cmake', 'yaml-cpp')
                    yaml_cpp_path = cls._get_valid_cmake_path_config(prompt_text="\n> yaml-cpp CMake Config Path",
                                                                     default_path=default_yaml_cpp_path)

                    # Kokkos paths per backend
                    kokkos_paths = {}
                    default_kokkos_paths_mapping = {
                        cls.ImplementedBackends.SERIAL.name: os.path.join(default_install, 'kokkos-5', 'single_core',
                                                                          'lib', 'cmake', 'Kokkos'),
                        cls.ImplementedBackends.OPENMP.name: os.path.join(default_install, 'kokkos-5', 'openmp', 'lib',
                                                                          'cmake', 'Kokkos'),
                        cls.ImplementedBackends.CUDA.name: os.path.join(default_install, 'kokkos-5', 'cuda', 'lib',
                                                                        'cmake', 'Kokkos'),
                        cls.ImplementedBackends.HIP.name: os.path.join(default_install, 'kokkos-5', 'hip', 'lib',
                                                                        'cmake', 'Kokkos'),
                    }

                    print("\nEnter Kokkos paths (type 'skip' or 'none' to omit a backend):")
                    for backend in cls.ImplementedBackends:
                        backend_name = backend.name
                        default_p = default_kokkos_paths_mapping.get(backend_name, "")
                        kokkos_paths[backend_name] = cls._get_valid_cmake_path_config(
                            prompt_text=f"\n> Kokkos [{backend_name}] Path", default_path=default_p)

                # Print summary
                print("\n\n" + "*" * 60)
                print(" CONFIGURATION SUMMARY")
                print("*" * 60)
                print(f"   QuantKos Root     : {src_root}")
                print(f"   Workspace Root    : {workspace_root}")
                print(f"   Build Type        : {build_type}")
                print(f"   Build Tests suite : {build_tests_also}")
                print(f"   Use FetchContent  : {use_fetchcontent}")
                if use_fetchcontent:
                    print(f"   GPU Arch     :")
                    for backend_name, arch in gpu_arch.items():
                        print(f"     - [{backend_name}] : {arch}")
                else:
                    print(f"   yaml-cpp Path     : {yaml_cpp_path}")
                    print(f"   Kokkos Paths      :")
                    for backend_name, banchend_path in kokkos_paths.items():
                        print(f"     - [{backend_name}] : {banchend_path}")
                print("=" * 60)

                # Ask for confirmation
                is_confirmed = cls._get_boolean_choice(
                    prompt_text="\nAre these settings correct?",
                    default_to_yes=True
                )

                # Create instance or repeat
                if is_confirmed:
                    return cls(
                        QuantoKos_root_dir=src_root,
                        workspace_root_dir=workspace_root,
                        build_type=build_type,
                        build_tests_also=build_tests_also,
                        use_fetchcontent=use_fetchcontent,
                        kokkos_paths=kokkos_paths,
                        yaml_cpp_path=yaml_cpp_path,
                    )
                else:
                    print("\n🔄 Restarting configuration from the beginning...\n")

            except SystemExit as e:
                print(e)
                return None

    @staticmethod
    def _prompt_input(prompt_text: str, default: str) -> str:
        """!
        @brief Prompts the user for input with a default value and handles exit signals.
        @exception SystemExit If the user types 'exit' or 'q'.
        """
        user_val = input(f"{prompt_text} [{default}]: ").strip()
        if user_val.lower() in ['exit', 'q', 'quit']:
            print("Exiting configuration setup.")
            raise SystemExit(0)
        return user_val if user_val else default

    @staticmethod
    def _get_valid_directory(prompt_text: str, default_path: str, must_exist: bool = False) -> str:
        """!
        @brief Continuously prompts until a valid directory path is provided.
        """
        while True:
            path = QuantKosBuilder._prompt_input(prompt_text, default_path)
            abs_path = os.path.abspath(path)

            if must_exist and not os.path.isdir(abs_path):
                print(f"  ❌ Error: Directory '{abs_path}' does not exist. Please try again (or type 'exit').")
                continue

            return abs_path

    @staticmethod
    def _get_boolean_choice(prompt_text: str, default_to_yes: bool = False) -> bool:
        """!
        @brief Prompts until a valid boolean decision (y/n) is provided.
        """
        default_str = "y" if default_to_yes else "n"
        while True:
            user_val = QuantKosBuilder._prompt_input(prompt_text, default_str)
            if user_val in ['y', 'yes', 'true', '1']:
                return True
            if user_val in ['n', 'no', 'false', '0']:
                return False
            print("  ❌ Invalid choice. Please enter 'y' for yes, 'n' for no, or 'exit'.")

    @staticmethod
    def _get_enum_choice(prompt_text: str, enum_cls: type, default_name: str) -> str:
        """!
        @brief Prompts the user to select an option from a given Enum class.
        """
        options = [item.name for item in enum_cls]
        opts_str = "\n\t- " + "\n\t- ".join(options)
        while True:
            val = QuantKosBuilder._prompt_input(f"{prompt_text} : {opts_str}\n\t ", default_name)
            for opt in options:
                if val.lower() == opt.lower():
                    return opt
            print(f"  ❌ Invalid choice '{val}'. Please select from: {opts_str}")

    @staticmethod
    def _get_valid_cmake_path_config(prompt_text: str, default_path: str) -> str:
        """!
        @brief Continuously prompts until a directory containing CMake config files is verified.
        @note Accepts 'none' or 'skip' or to omit a path.
        """
        while True:
            path = QuantKosBuilder._prompt_input(prompt_text, default_path)
            print(f"inp path : {path}")
            # Add the possibility to skip the path
            if path.lower() in ['none', 'skip', '']:
                print("  -> Skipped backed.")
                return None

            # Check if path exists
            abs_path = os.path.abspath(path)
            if not os.path.isdir(abs_path):
                print(f"  ❌ Error: Path '{abs_path}' does not exist. Please try again.")
                continue

            # Check if directory actually contains CMake configuration files
            has_cmake_file = any(f.endswith('.cmake') for f in os.listdir(abs_path)) if os.path.exists(
                abs_path) else False
            if not has_cmake_file:
                print(f"  ⚠️ Warning: No '.cmake' files found in '{abs_path}'. Double check the path.")
                continue_msg = "Use this path anyway? (y/n): "
                confirm = QuantKosBuilder._get_boolean_choice(continue_msg)
                if confirm:
                    return abs_path
                else:
                    continue

            return abs_path

    @classmethod
    def _auto_detect_nvidia_arch(cls) -> Optional[str]:
        """!
        @brief Queries the system via nvidia-smi to detect the GPU's Compute Capability.
               Returns None if no NVIDIA GPU is found.
        """
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                capture_output=True, text=True, check=True
            )

            compute_cap = result.stdout.strip().split('\n')[0].strip()

            mapping = {
                "12.1": cls.ImplementedNvidiaGPUArch.BLACKWELL121.name,
                "12.0": cls.ImplementedNvidiaGPUArch.BLACKWELL120.name,
                "10.3": cls.ImplementedNvidiaGPUArch.BLACKWELL103.name,
                "10.0": cls.ImplementedNvidiaGPUArch.BLACKWELL100.name,

                "9.0": cls.ImplementedNvidiaGPUArch.HOPPER90.name,
                "8.9": cls.ImplementedNvidiaGPUArch.ADA89.name,
                "8.6": cls.ImplementedNvidiaGPUArch.AMPERE86.name,
                "8.0": cls.ImplementedNvidiaGPUArch.AMPERE80.name,
                "7.5": cls.ImplementedNvidiaGPUArch.TURING75.name,
                "7.0": cls.ImplementedNvidiaGPUArch.VOLTA70.name,
                "6.1": cls.ImplementedNvidiaGPUArch.PASCAL61.name,
                "6.0": cls.ImplementedNvidiaGPUArch.PASCAL60.name,
            }

            arch = mapping.get(compute_cap)
            if arch:
                print(f"<\t\tAuto-detected NVIDIA Architecture: {arch} (Compute {compute_cap})")
                return arch
            else:
                print(f"<  WARNING: Unrecognized NVIDIA compute capability '{compute_cap}'.")
                return None

        except (subprocess.CalledProcessError, FileNotFoundError):
            print("<  WARNING: nvidia-smi not found or failed. No NVIDIA GPU detected.")
            return None

    @classmethod
    def _auto_detect_amd_arch(cls) -> Optional[None]:
        """!
        @brief Queries the system via rocminfo to detect the AMD GPU's gfx version.
               Returns None if no AMD GPU is found.
        """

        try:
            # rocminfo outputs detailed specs for all compute agents
            result = subprocess.run(
                ["rocminfo"],
                capture_output=True, text=True, check=True
            )

            # Look for the first GPU agent name, which always starts with 'gfx'
            # e.g., "Name:                    gfx90a"
            match = re.search(r'Name:\s+(gfx[0-9a-z]+)', result.stdout, re.IGNORECASE)

            if not match:
                print("  [WARNING] Could not parse a valid 'gfx' version from rocminfo.")
                return None

            gfx_version = match.group(1).lower()

            # Map AMD gfx versions to Kokkos CMake architecture flags
            # (Matches Kokkos 4.x architecture naming conventions)
            mapping = {
                # CDNA / Supercomputing
                "gfx942": cls.ImplementedAmdGPUArch.MI300X.name,
                "gfx940": cls.ImplementedAmdGPUArch.MI300A.name,
                "gfx90a": cls.ImplementedAmdGPUArch.VEGA90A.name,
                "gfx908": cls.ImplementedAmdGPUArch.VEGA908.name,
                "gfx906": cls.ImplementedAmdGPUArch.VEGA906.name,
                "gfx900": cls.ImplementedAmdGPUArch.VEGA900.name,

                # RDNA3 Series
                "gfx1100": cls.ImplementedAmdGPUArch.NAVI31.name,
                "gfx1101": cls.ImplementedAmdGPUArch.NAVI32.name,
                "gfx1102": cls.ImplementedAmdGPUArch.NAVI33.name,

                # RDNA2 Series
                "gfx1030": cls.ImplementedAmdGPUArch.NAVI21.name,
                "gfx1031": cls.ImplementedAmdGPUArch.NAVI22.name,
            }

            arch = mapping.get(gfx_version)
            if arch:
                print(f"<\t\tAuto-detected AMD Architecture: {arch} ({gfx_version})")
                return arch
            else:
                print(f"<  WARNING: Unrecognized AMD gfx version '{gfx_version}'.")
                return None

        except (subprocess.CalledProcessError, FileNotFoundError):
            print("<  WARNING: rocminfo not found or failed. No AMD ROCm stack detected.")
            return None

    def build_and_install(self,
                          precision: str,
                          math: str,
                          backend: str,
                          ) -> Optional[str]:
        """!
        @brief Configures, builds, and installs the QuantKos engine for a specific configuration,
        saving the build log to both the build and install directories.

        @param precision The floating-point precision to use (e.g., "SINGLE", "DOUBLE").
        @param math      The math mode to use (e.g., "IEEE", "FAST").
        @param backend   The hardware execution paradigm (e.g., "SERIAL", "OPENMP", "CUDA").

        @return The absolute path to the installed QuantKos executable on success, or None on failure.
        """
        # Reset log output string for this build run
        self.out_str = ""

        # Create build and install dir if do not exist
        target_suffix = f"{self.build_type.lower()}_{backend.lower()}_{precision.lower()}_{math.lower()}"
        build_dir = os.path.join(self.build_root, f"QuantKos-{target_suffix}")
        install_dir = os.path.join(self.install_root, f"QuantKos-{target_suffix}")  # create by cmake
        os.makedirs(build_dir, exist_ok=True)

        # 1. CMake Configure
        cmake_configure_cmd = (["cmake", self.src_root] +
                               self._get_cmake_flags(precision, math, backend, install_dir)
                               )
        print(f"\n\n" + f"{'.' * 60}")
        print(f"Configuring cmake : {precision} | {math} | {backend}")
        try:
            self._run_command(cmake_configure_cmd, working_dir=build_dir)
        except Exception as e:
            print(e)
            self._save_log(build_dir, "build_failure.log")
            return None

        # 2. Compile QuantKos given the specification
        build_cmd = ["cmake", "--build", ".", "-j", str(os.cpu_count() or 4)]
        print("\nBuilding...")
        try:
            self._run_command(build_cmd, working_dir=build_dir)
        except Exception as e:
            print(e)
            self._save_log(build_dir, "build_failure.log")
            return None

        # 3. Install QuantKos just compiled
        install_cmd = ["cmake", "--install", "."]
        print("\nInstalling...")
        try:
            self._run_command(install_cmd, working_dir=build_dir)
        except Exception as e:
            print(e)
            self._save_log(build_dir, "build_failure.log")
            return None

        # 4. Save build logs to both Build and Install directories
        print("\n\n")
        log_filename = f"python_builder_{target_suffix}.log"
        self._save_log(build_dir, log_filename)
        self._save_log(install_dir, log_filename)

        return os.path.join(install_dir, "bin", "QuantKos")

    def _run_command(self,
                     command: List[str],
                     working_dir: str) -> str:
        """!
        @brief Executes a shell command in a specified working directory, streaming output
               to stdout in real-time, accumulating logs, and timing execution.

        @param command     A list of command arguments (e.g., ["cmake", "--build", "."]).
        @param working_dir The directory where the command will be executed.

        @return The standard output (stdout) of the executed command.

        @exception RuntimeError Thrown if the command returns a non-zero exit code.
        """
        start_time = time.perf_counter()

        # Print and accumulate output
        command_str = f"\n\n>>> Executing: {' '.join(command)} (in {working_dir})\n" + "-" * 60 + "\n"
        print(command_str)
        full_output = command_str
        self.out_str += command_str

        process = subprocess.Popen(  # Popen to stream stdout in real-time, child process is created
            command,
            cwd=working_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Redirect stderr into stdout stream for unified real-time logs
            text=True,
            bufsize=1  # Line-buffered output
        )

        # Read and print lines as they are emitted
        stdout_lines: List[str] = []
        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                sys.stdout.write(line)
                sys.stdout.flush()
                stdout_lines.append(line)
        process.stdout.close()
        return_code = process.wait()  # Retrieving the exit status
        if return_code != 0:
            msg = f" ! Error executing {' '.join(command)}\n\n{result.stderr}"
            raise RuntimeError(msg)

        # Show elapsed time
        elapsed_time = time.perf_counter() - start_time
        time_msg = f"\nElapsed time  : {elapsed_time}s"
        print(time_msg)

        # return all the outputs that were produced
        full_output += "".join(stdout_lines) + time_msg
        self.out_str += full_output
        return full_output

    def _save_log(self, target_dir: str, filename: str) -> None:
        """!
        @brief Helper method to safely write accumulated build logs to a file.
        """
        try:
            os.makedirs(target_dir, exist_ok=True)
            log_path = os.path.join(target_dir, filename)
            with open(log_path, "w", encoding="utf-8") as log_file:
                log_file.write(self.out_str)
            print(f"  📝 Log saved to: {log_path}")
        except Exception as e:
            print(f"  ⚠️ Warning: Could not save build log to '{target_dir}': {e}")

    def _get_cmake_flags(self,
                         precision: str,
                         math: str,
                         backend: str,
                         install_dir: str,
                         ) -> List[str]:
        """!
        @brief Generates the required CMake configuration flags for the requested build strategy.

        @param precision   The floating-point precision to use ("SINGLE" or "DOUBLE").
        @param math        The math mode to use ("IEEE" or "FAST").
        @param backend     The execution backend paradigm ("SERIAL", "OPENMP", or "CUDA").
        @param install_dir The path where the compiled executable will be installed.

        @return A list of string flags to pass to the cmake configuration command.

        @exception FileNotFoundError Thrown if local dependencies are required but the path is not found.
        """

        if self.build_type not in [bt.name for bt in self.ImplementedBuildType]:
            msg = f" Built type {self.build_type} not in implemented types : {self.ImplementedBuildType}"
            raise Valueerror(msg)

        # Common CMake flags
        flags = [
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
            f"-DCMAKE_BUILD_TYPE={self.build_type}",
            "-DQKOS_BUILD_DOCS=OFF"
        ]

        # Precision Mapping
        flags.append("-DQKOS_ENABLE_SINGLE_PRECISION=" +
                     ("ON" if precision == self.ImplementedPrecision.SINGLE.name else "OFF")
                     )

        # Math Mode Mapping
        flags.append("-DQKOS_ENABLE_FAST_MATH=" + ("ON" if math == self.ImplementedMath.FAST.name else "OFF")
                     )

        # Buil Test Suite
        flags.append("-DQKOS_ENABLE_TESTS=" + ("ON" if self.build_tests_also else "OFF")
                     )

        # --- LOCAL PRE-INSTALLED DEPENDENCIES (FetchContent = OFF) ---
        if not self.use_fetchcontent:
            flags.append("-DQKOS_ENABLE_FETCHCONTENT=OFF")

            # Point to local yaml-cpp
            flags.append(f"-Dyaml-cpp_ROOT={self.yaml_cpp_path}")

            # Point to the local pre-built Kokkos matching the target paradigm
            kokkos_root = self.kokkos_paths.get(backend)
            if kokkos_root is None:
                raise FileNotFoundError(
                    f"Kokkos installation directory for '{backend}' was skipped. Aborting compilation."
                )
            if not os.path.exists(kokkos_root):
                raise FileNotFoundError(
                    f"Kokkos installation directory for '{backend}' not found at: {kokkos_root}"
                )
            flags.append(f"-DKokkos_ROOT={kokkos_root}")

            # NOTE: No Kokkos architecture or backend flags (-DKokkos_ENABLE_*) are needed
            # because QuantKos simply links against the pre-compiled Kokkos binaries.

        # --- FETCHCONTENT (AUTOMATIC DOWNLOADS) ---
        else:
            flags.append("-DQKOS_ENABLE_FETCHCONTENT=ON")

            if backend == self.ImplementedBackends.SERIAL.name:
                flags.extend([
                    "-DKokkos_ENABLE_SERIAL=ON",
                    "-DKokkos_ENABLE_OPENMP=OFF",
                    "-DKokkos_ENABLE_CUDA=OFF",
                    "-DKokkos_ARCH_NATIVE=ON"
                ])

            elif backend == self.ImplementedBackends.OPENMP.name:
                flags.extend([
                    "-DKokkos_ENABLE_SERIAL=ON",
                    "-DKokkos_ENABLE_OPENMP=ON",
                    "-DKokkos_ENABLE_CUDA=OFF",
                    "-DKokkos_ARCH_NATIVE=ON"
                ])

            elif backend == self.ImplementedBackends.CUDA.name:
                target_arch = self.gpu_arch.get(self.ImplementedBackends.CUDA.name)
                if target_arch is None:
                    raise ValueErrror(
                        "! ERROR ! Skipping CUDA Compilation because cuda architecture was not provided as input of the class."
                        f"Please provide one of the following architectures : {self.ImplementedNvidiaGPUArch}")
                if target_arch not in [bt.name for bt in self.ImplementedNvidiaGPUArch]:
                    msg = f" CUDA GPU Arch {target_arch} unknown. knowns are : {self.ImplementedNvidiaGPUArch}"
                    raise ValueError(msg)
                flags.extend([
                    "-DKokkos_ENABLE_SERIAL=ON",
                    "-DKokkos_ENABLE_OPENMP=ON",  # enabling openp in case of sorting the payoffs
                    "-DKokkos_ENABLE_CUDA=ON",
                    "-DKokkos_ENABLE_CUDA_LAMBDA=ON",
                    f"-DKokkos_ARCH_{target_arch.upper()}=ON"  # <-- Dynamic GPU architecture
                ])
            elif backend == self.ImplementedBackends.HIP.name:
                target_arch = self.gpu_arch.get(self.ImplementedBackends.HIP.name)
                if target_arch is None:
                    raise ValueError(
                        "! ERROR ! Skipping HIP Compilation because AMD architecture was not provided as input."
                        f" Please provide one of the following architectures : {self.ImplementedAmdGPUArch}")

                if target_arch not in [bt.name for bt in self.ImplementedAmdGPUArch]:
                    msg = f" HIP GPU Arch {target_arch} unknown. Knowns are : {self.ImplementedAmdGPUArch}"
                    raise ValueError(msg)

                # 2. Append Kokkos CMake flags for AMD ROCm/HIP
                flags.extend([
                    "-DKokkos_ENABLE_SERIAL=ON",
                    "-DKokkos_ENABLE_OPENMP=ON",  # Enabling OpenMP in case of sorting the payoffs
                    "-DKokkos_ENABLE_HIP=ON",  # <-- Enable AMD HIP Backend
                    f"-DKokkos_ARCH_{target_arch.upper()}=ON"  # <-- Dynamic AMD GPU architecture (e.g., VEGA90A)
                ])
            else:
                msg = (f"\n ! Error -> only the following backend are implemented so far : \n"
                       f"{[bc.name for bc in list(self.ImplementedBackends)]}")
                raise NotImplementedError(msg)

        return flags


if __name__ == "__main__":

    builder = QuantKosBuilder.from_interactive()
    if builder is None:
        print("Warning, builder was not created because interactive build was interrupted. ")
    else:
        # for backend in builder.ImplementedBackends:
        #     for precision in builder.ImplementedPrecision:
        #         for math in builder.ImplementedMath:
        #             builder.build_and_install(precision=precision.name,
        #                                       math=math.name,
        #                                       backend=backend.name)

        builder.build_and_install(precision=builder.ImplementedPrecision.DOUBLE.name,
                                  math=builder.ImplementedMath.IEEE.name,
                                  backend=builder.ImplementedBackends.OPENMP.name)

        # /home/zen/software/installations/kokkos-5/single_core/lib/cmake/Kokkos
        # /home/zen/software/installations/kokkos-5/openmp/lib/cmake/Kokkos
        # /home/zen/software/installations/kokkos-5/cuda/lib/cmake/Kokkos
