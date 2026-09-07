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

import sys
import os
import re
import subprocess
import datetime
from enum import Enum
from typing import List, Dict, Optional, Tuple

# import builder
current_dir = os.path.dirname(os.path.abspath(__file__))
builder_dir = os.path.abspath(os.path.join(current_dir, '..'))
if builder_dir not in sys.path:
    sys.path.append(builder_dir)
try:
    from builder import QuantKosBuilder
except ModuleNotFoundError as e:
    print(e)
    print(f"Looking the builder at the path {builder_dir}. Files found : {os.listdir(builder_dir)}")


class QuantKosBenchmark:
    """!
    @brief Orchestrates the building, execution, and metric extraction of QuantKos benchmarks.
    """

    class ImplementedBenchmarks(Enum):
        ALL = 0
        AsianNoGreeks = 1

    BenchmarksMap = {
        ImplementedBenchmarks.AsianNoGreeks.name: "bench_asian_option_price_no_greeks.yaml"
    }

    class ImplementedTimeRes(Enum):
        T_Total_Computation = 1
        T_Option = 2
        T_Greeks = 3

    ComputationTimePatterns = {
        ImplementedTimeRes.T_Total_Computation.name: r"Total time spent for the computations\s*:\s*([\d\.]+)s",
        ImplementedTimeRes.T_Option.name: r"Time spent evaluating the option price\s*:\s*([\d\.]+)s",
        ImplementedTimeRes.T_Greeks.name: r"Time spent evaluating the greeks\s*:\s*([\d\.]+)s",
    }

    class BenchmarkMetrics(Enum):
        BenchmarkID = 1
        Build = 2
        TotalTime = 3
        PricingTime = 4
        GreeksTime = 5

    def __init__(self, builder: QuantKosBuilder):
        self.builder = builder
        self.benchmarks_conf_filenames: Optional[List[str]] = None
        self.target_backends: Optional[List[str]] = None
        self.target_precisions: Optional[List[str]] = None
        self.target_math_modes: Optional[List[str]] = None
        self.executables: Dict[str, str] = {}  # Maps "BuildType_Backend_Precision_Math" -> "path/to/executable"
        self.results: List[Dict] = []  # Stores parsed metrics for each run

    def run_interactive_benchmark(self):
        """!
        @brief Main interactive loop to build, select configs, run, and report benchmarks.
        """
        inp_msg = (f" QuantKos Benchmark Orchestrator\n"
                   f"{'-' * 60}\n"
                   f"   Author            : Zeno GEDDO\n"
                   f"   Build Year        : 2026\n"
                   f"   License           : GNU GENERAL PUBLIC LICENSE (V.3, 29/06/2007)\n"
                   f"   Contact           : zeno.geddo@gmail.com"
                   )
        print("\n\n" + "=" * 60 + f"\n{inp_msg}\n" + "=" * 60)
        self._select_benchmark()
        self._ask_required_builds()
        self._build_binaries()
        self._run_benchmarks_and_parse_results()
        self._analyze_and_report_results()

    def _select_benchmark(self) -> List[str]:
        """!
        @brief Helper to let the user select a benchmark config using the builder's Enum prompter.
        @return A list containing the filenames of the selected benchmark configs.
        """
        print("\n--- Selecting Benchmark to Run ---")
        selected_choice = QuantKosBuilder._get_enum_choice(prompt_text="\n> Select Benchmark configuration",
                                                           enum_cls=self.ImplementedBenchmarks,
                                                           default_name=self.ImplementedBenchmarks.ALL.name
                                                           )

        if selected_choice == self.ImplementedBenchmarks.ALL.name:
            # all mapped filenames from the dictionary
            self.benchmarks_conf_filenames = list(self.BenchmarksMap.values())
        else:
            # the specific mapped filename for the chosen Enum
            self.benchmarks_conf_filenames = [self.BenchmarksMap[selected_choice]]

    def _ask_required_builds(self):
        self.target_backends = self._select_build_options("Backends", QuantKosBuilder.ImplementedBackends)
        self.target_precisions = self._select_build_options("Precisions", QuantKosBuilder.ImplementedPrecision)
        self.target_math_modes = self._select_build_options("Math Modes", QuantKosBuilder.ImplementedMath)

        # Filter out target_backends that were skipped during builder setup (when FetchContent = False)
        if not self.builder.use_fetchcontent and self.builder.kokkos_paths:
            valid_backends = []
            for b in self.target_backends:
                if self.builder.kokkos_paths.get(b) is None:
                    print(f"  ⚠️ Skipping backend '{b}' because its local Kokkos path was omitted.")
                else:
                    valid_backends.append(b)
            self.target_backends = valid_backends

        if not self.target_backends:
            raise ValueError("\n❌ No valid target_backends available to build. Aborting benchmark.")

    def _select_build_options(self, name: str, enum_cls) -> List[str]:
        """Helper to let the user select multiple comma-separated enum values."""
        valid_names = [e.name for e in enum_cls]
        prompt = f"> Select {name} to benchmark (comma-separated, or 'all') [{', '.join(valid_names)}]: "
        while True:
            choice = input(prompt).strip().upper()
            if choice in ['', 'ALL']:
                return valid_names

            selected = [x.strip() for x in choice.split(',')]
            invalid = [x for x in selected if x not in valid_names]
            if invalid:
                print(f"  ❌ Invalid options: {invalid}. Choose from: {valid_names}")
                continue
            return selected

    def _build_binaries(self):
        print("\n--- Starting Building ---")
        for backend in self.target_backends:
            for prec in self.target_precisions:
                for math in self.target_math_modes:
                    target_name = f"{self.builder.build_type}_{backend}_{prec}_{math}"
                    print(f"\n>>> Building Target Variant: {target_name}")

                    exe_path = self.builder.build_and_install(
                        precision=prec,
                        math=math,
                        backend=backend
                    )

                    if exe_path and os.path.exists(exe_path):
                        self.executables[target_name] = exe_path
                    else:
                        print(f"  ❌ Build failed for {target_name}. Skipping in benchmark matrix.")

        if not self.executables:
            print("\n❌ No executables were successfully built. Aborting benchmark.")
            return

    def _run_benchmarks_and_parse_results(self):
        print("\n--- Executing Benchmark Matrix ---")
        path_curr_dir = os.path.abspath(os.path.dirname(__file__))
        for benchmark_name in self.benchmarks_conf_filenames:
            config_path = os.path.abspath(os.path.join(path_curr_dir, "configs", benchmark_name))
            for build_name, exe_path in self.executables.items():
                print(f" > Running {benchmark_name} on [{build_name}]...")
                metrics = self._run_quantkos_and_parse_results(exe_path, config_path)
                print(f" > Done")
                if metrics:
                    self.results.append({
                        self.BenchmarkMetrics.BenchmarkID.name: benchmark_name,
                        self.BenchmarkMetrics.Build.name: build_name,
                        self.BenchmarkMetrics.TotalTime.name: metrics[0],
                        self.BenchmarkMetrics.PricingTime.name: metrics[1],
                        self.BenchmarkMetrics.GreeksTime.name: metrics[2]
                    })

    def _run_quantkos_and_parse_results(self,
                                        exe_path: str,
                                        config_path: str,
                                        print_results: bool = False) -> Optional[Tuple[float, float, float]]:
        """Executes QuantKos and parses the timing stdout via regex."""
        try:
            # Assuming the executable takes the config file via CLI like: `./QuantKos --config config.yaml`
            result = subprocess.run([exe_path, config_path],
                                    capture_output=True, text=True, check=True)
            output = result.stdout
            if print_results:
                print(output)

            # Regex patterns strictly matching your engine's output format
            tot_match = re.search(self.ComputationTimePatterns.get(self.ImplementedTimeRes.T_Total_Computation.name),
                                  output)
            prc_match = re.search(self.ComputationTimePatterns.get(self.ImplementedTimeRes.T_Option.name),
                                  output)
            grk_match = re.search(self.ComputationTimePatterns.get(self.ImplementedTimeRes.T_Greeks.name),
                                  output)

            if tot_match and prc_match and grk_match:
                return float(tot_match.group(1)), float(prc_match.group(1)), float(grk_match.group(1))
            else:
                print(f"  ❌ ERROR: Could not parse timing metrics for {exe_path}.")
                return None

        except subprocess.CalledProcessError as e:
            print(f"  ❌ Execution crashed: {e}")
            return None

    def _analyze_and_report_results(self):
        """!
        @brief Analyzes raw execution metrics, computes speedups,
               and prints a benchmark table grouped by configuration.
        @note Benchmarks are exported in CSV.
        """
        if not self.results:
            print("\n❌ No benchmark results to analyze.")
            return

        print("\n" + "=" * 95)
        print(
            f"{'BenchmarkID':<20} | {'Build Variant':<28} | {'Total(s)':<10} | {'Pricing(s)':<10} | {'Greeks(s)':<10} | {'Speedup':<8}")
        print("-" * 95)

        # Group results by the configuration file used
        id_benchmarks = set(r[self.BenchmarkMetrics.BenchmarkID.name] for r in self.results)

        for banch_id in id_benchmarks:
            runs = [r for r in self.results if r[self.BenchmarkMetrics.BenchmarkID.name] == banch_id]

            # Determine the baseline (Preferably SERIAL_DOUBLE, else the slowest total time)
            baseline_run = next((r for r in runs if "SERIAL_DOUBLE" in r[self.BenchmarkMetrics.Build.name].upper()),
                                None)
            if not baseline_run:
                baseline_run = max(runs, key=lambda x: x[self.BenchmarkMetrics.TotalTime.name])
            baseline_time = baseline_run[self.BenchmarkMetrics.TotalTime.name]

            # Sort runs by Total Time (fastest first)
            runs.sort(key=lambda x: x[self.BenchmarkMetrics.TotalTime.name])

            for run in runs:
                build = run[self.BenchmarkMetrics.Build.name]
                tot = run[self.BenchmarkMetrics.TotalTime.name]
                prc = run[self.BenchmarkMetrics.PricingTime.name]
                grk = run[self.BenchmarkMetrics.GreeksTime.name]

                # Derived Metrics
                speedup = baseline_time / tot if tot > 0 else 0.0

                speedup_str = f"{speedup:.2f}x"
                if speedup == 1.0 and build == baseline_run[self.BenchmarkMetrics.Build.name]:
                    speedup_str = "BASE"

                print(f"{banch_id:<20} | {build:<28} | {tot:<10.4f} | {prc:<10.4f} | {grk:<10.4f} | {speedup_str:<8}")

            print("-" * 95)

        self._export_to_csv()

    def _export_to_csv(self):
        """!
        @brief Saves the raw and derived metrics to a CSV file for external analysis.
        """

        workspace = self.builder.workspace_root
        reports_dir = os.path.join(workspace, "QuantKos_benchmark_reports")
        os.makedirs(reports_dir, exist_ok=True)


        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(reports_dir, f"benchmark_{timestamp}.csv")

        with open(csv_path, "w") as f:
            # CSV Header
            f.write("BenchmarkID,Build,TotalTime_s,PricingTime_s,GreeksTime_s,Speedup\n")

            bench_ids = set(r[self.BenchmarkMetrics.BenchmarkID.name] for r in self.results)
            for config in bench_ids:
                runs = [r for r in self.results if r[self.BenchmarkMetrics.BenchmarkID.name] == config]
                runs.sort(key=lambda x: x[self.BenchmarkMetrics.TotalTime.name])

                # Match baseline selection logic with the terminal table
                baseline_run = next((r for r in runs if "SERIAL_DOUBLE" in r[self.BenchmarkMetrics.Build.name].upper()), None)
                if not baseline_run:
                    baseline_run = max(runs, key=lambda x: x[self.BenchmarkMetrics.TotalTime.name])
                baseline_time = baseline_run[self.BenchmarkMetrics.TotalTime.name]

                for run in runs:
                    tot = run[self.BenchmarkMetrics.TotalTime.name]
                    prc = run[self.BenchmarkMetrics.PricingTime.name]
                    grk = run[self.BenchmarkMetrics.GreeksTime.name]
                    speedup = baseline_time / tot if tot > 0 else 0.0

                    f.write(f"{run[self.BenchmarkMetrics.BenchmarkID.name]},"
                            f"{run[self.BenchmarkMetrics.Build.name]},"
                            f"{tot:.6f},{prc:.6f},{grk:.6f},{speedup:.4f}\n")

        print(f"\n📊 CSV Report saved to: {csv_path}")


if __name__ == "__main__":
    builder = QuantKosBuilder.from_interactive()  # collect info but does not build yet
    benchmarker = QuantKosBenchmark(builder)
    benchmarker.run_interactive_benchmark()  # build with the builder collected informations
