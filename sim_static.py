import numpy as np
from scipy.optimize import minimize
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Create data storage directory
data_dir = "simulation_data"
os.makedirs(data_dir, exist_ok=True)

# =============================================================================
# Scenario and Topology Configuration
# =============================================================================

# Define scenario abbreviation names
scenario_names = {
    # Baseline scenarios
    "B4": "Baseline 4-path",
    "B6": "Baseline 6-path",
    "B3": "Baseline 3-path",
    "B8": "Baseline 8-path",

    # High-load scenarios
    "H4": "High-load 4-path",
    "H6": "High-load 6-path",
    "H3": "High-load 3-path",
    "H8": "High-load 8-path",

    # Asymmetric capacity scenarios
    "A4": "Asymmetric 4-path",
    "A6": "Asymmetric 6-path",
    "A3": "Asymmetric 3-path",
    "A8": "Asymmetric 8-path"
}


# Define three traffic models
def get_traffic_scenario(scenario_type):
    """Get configuration for different traffic models"""
    scenarios = {
        "baseline": {
            "latency": 3.8,
            "throughput": 7.2,
            "medium": 2.0
        },
        "high_load": {
            "latency": 4.0,
            "throughput": 10.0,
            "medium": 3.0
        },
        "asymmetric": {
            "latency": 5.0,
            "throughput": 12.0,
            "medium": 3.0
        }
    }
    return scenarios.get(scenario_type, scenarios["baseline"])


# Define four topologies
def get_topology_config(topology_type):
    """Get configuration for different topologies"""
    topologies = {
        # Standard 4-path Fat-Tree
        "4path": {
            "paths": ['Path1', 'Path2', 'Path3', 'Path4'],
            "capacity": {'Path1': 10, 'Path2': 10, 'Path3': 5, 'Path4': 5},
            "delay": {'Path1': 20, 'Path2': 25, 'Path3': 40, 'Path4': 40}
        },
        # 6-path Fat-Tree
        "6path": {
            "paths": ['Core1-ToR1', 'Core1-ToR2', 'Core2-ToR1', 'Core2-ToR2', 'Core3-ToR1', 'Core3-ToR2'],
            "capacity": {'Core1-ToR1': 10, 'Core1-ToR2': 10, 'Core2-ToR1': 10, 'Core2-ToR2': 10,
                         'Core3-ToR1': 5, 'Core3-ToR2': 5},
            "delay": {'Core1-ToR1': 20, 'Core1-ToR2': 25, 'Core2-ToR1': 22, 'Core2-ToR2': 27,
                      'Core3-ToR1': 40, 'Core3-ToR2': 45}
        },
        # Simplified 3-path topology
        "3path": {
            "paths": ['Primary', 'Backup1', 'Backup2'],
            "capacity": {'Primary': 15, 'Backup1': 8, 'Backup2': 5},
            "delay": {'Primary': 15, 'Backup1': 35, 'Backup2': 50}
        },
        # 8-path Clos network
        "8path": {
            "paths": ['Spine1-Leaf1', 'Spine1-Leaf2', 'Spine1-Leaf3', 'Spine1-Leaf4',
                      'Spine2-Leaf1', 'Spine2-Leaf2', 'Spine2-Leaf3', 'Spine2-Leaf4'],
            "capacity": {'Spine1-Leaf1': 10, 'Spine1-Leaf2': 10, 'Spine1-Leaf3': 10, 'Spine1-Leaf4': 10,
                         'Spine2-Leaf1': 10, 'Spine2-Leaf2': 10, 'Spine2-Leaf3': 10, 'Spine2-Leaf4': 10},
            "delay": {'Spine1-Leaf1': 18, 'Spine1-Leaf2': 20, 'Spine1-Leaf3': 22, 'Spine1-Leaf4': 24,
                      'Spine2-Leaf1': 20, 'Spine2-Leaf2': 22, 'Spine2-Leaf3': 24, 'Spine2-Leaf4': 26}
        }
    }
    return topologies.get(topology_type, topologies["4path"])


# =============================================================================
# Utility Function Configuration (Modified Version)
# =============================================================================

def get_utility_config(utility_type):
    """Get configuration for different utility function variants (corrected version)"""
    utility_configs = {
        "baseline": {
            "name": "Baseline",
            "latency_k": 0.01,  # Reduced coefficient to balance weights
            "medium_k": 0.005,  # Reduced coefficient to balance weights
            "throughput_func": lambda x: np.log(1 + x),
            "normalization_factor": 10.0  # New normalization factor
        },
        "sensitive": {
            "name": "Sensitive Latency",
            "latency_k": 0.02,
            "medium_k": 0.01,
            "throughput_func": lambda x: np.log(1 + x),
            "normalization_factor": 10.0
        },
        "linear": {
            "name": "Linear Throughput",
            "latency_k": 0.01,
            "medium_k": 0.005,
            "throughput_func": lambda x: x * 0.5,  # Reduced linear weight
            "normalization_factor": 15.0  # Increased normalization factor
        }
    }
    return utility_configs.get(utility_type, utility_configs["baseline"])


# =============================================================================
# Core Calculation Functions (Corrected Version)
# =============================================================================

def calculate_delay(base_delay, utilization):
    """Calculate path delay (corrected M/M/1 queue model)"""
    if utilization >= 0.99:
        return base_delay * 100  # Penalty high delay when congested
    elif utilization <= 0.01:
        return base_delay  # Base delay at low utilization
    else:
        # Standard queuing model: total delay = transmission delay + queuing delay
        transmission_delay = base_delay
        queue_delay = (utilization / (1 - utilization)) * (base_delay / 2)  # Standardized queuing delay
        return transmission_delay + queue_delay


def calculate_utility(allocations, topology, utility_config):
    """Calculate utility value for given allocation scheme (corrected version)"""
    paths = topology["paths"]
    path_capacity = topology["capacity"]
    base_delay = topology["delay"]

    num_paths = len(paths)
    num_traffic_types = len(allocations) // num_paths

    # Reconstruct allocation matrix
    traffic_alloc = {}
    traffic_types = ["latency", "throughput", "medium"][:num_traffic_types]

    for i, t_type in enumerate(traffic_types):
        traffic_alloc[t_type] = allocations[i * num_paths:(i + 1) * num_paths]

    # Calculate path utilization and delay
    utilizations = []
    delays = []
    for i, path in enumerate(paths):
        total_traffic = sum(traffic_alloc[t_type][i] for t_type in traffic_types)
        utilization = total_traffic / path_capacity[path]
        utilizations.append(utilization)
        delays.append(calculate_delay(base_delay[path], utilization))

    # Calculate utility value (using linear penalty model instead of exponential model)
    U_lat = sum(traffic_alloc["latency"][i] * max(0, 1 - utility_config["latency_k"] * delays[i])
                for i in range(num_paths))

    total_throughput = sum(traffic_alloc["throughput"])
    U_bw = utility_config["throughput_func"](total_throughput)

    U_med = sum(traffic_alloc["medium"][i] * max(0, 1 - utility_config["medium_k"] * delays[i])
                for i in range(num_paths))

    # Normalize total utility
    normalized_utility = (U_lat + U_bw + U_med) / utility_config["normalization_factor"]

    return normalized_utility, utilizations, delays


def calculate_metrics(allocations, topology):
    """Calculate performance metrics (corrected version)"""
    paths = topology["paths"]
    path_capacity = topology["capacity"]
    base_delay = topology["delay"]

    num_paths = len(paths)
    num_traffic_types = len(allocations) // num_paths
    traffic_types = ["latency", "throughput", "medium"][:num_traffic_types]

    # Reconstruct allocation matrix
    traffic_alloc = {}
    for i, t_type in enumerate(traffic_types):
        traffic_alloc[t_type] = allocations[i * num_paths:(i + 1) * num_paths]

    # Calculate path utilization and delay (using corrected delay calculation)
    utilizations = []
    delays = []
    for i, path in enumerate(paths):
        total_traffic = sum(traffic_alloc[t_type][i] for t_type in traffic_types)
        utilization = total_traffic / path_capacity[path]
        utilizations.append(utilization)
        delays.append(calculate_delay(base_delay[path], utilization))

    # Calculate metrics
    avg_delay = np.mean(delays)
    max_utilization = np.max(utilizations)
    actual_throughput = 0
    for i, path in enumerate(paths):
        path_traffic = sum(traffic_alloc[t_type][i] for t_type in traffic_types)
        actual_throughput += min(path_traffic, path_capacity[path] * 0.9)  # Consider capacity constraints
    std_delay = np.std(delays)

    return avg_delay, max_utilization, actual_throughput, std_delay


def UtiliNet_optimization(topology, traffic_demand, utility_config):
    """UtiliNet optimization solution (corrected version)"""
    paths = topology["paths"]
    path_capacity = topology["capacity"]
    base_delay = topology["delay"]

    num_paths = len(paths)
    traffic_types = ["latency", "throughput", "medium"]
    num_traffic_types = len(traffic_types)

    # Objective function
    def objective(x):
        utility, _, _ = calculate_utility(x, topology, utility_config)
        return -utility  # Minimize negative utility

    # Constraints
    constraints = []

    # Flow conservation constraints
    for i in range(num_traffic_types):
        def constraint_func(x, i=i):
            return sum(x[i * num_paths:(i + 1) * num_paths]) - traffic_demand[traffic_types[i]]

        constraints.append({'type': 'eq', 'fun': constraint_func})

    # Path capacity constraints (corrected version: add margin and ensure strict constraints)
    for j in range(num_paths):
        def constraint_func(x, j=j):
            total_load = sum(x[i * num_paths + j] for i in range(num_traffic_types))
            # Reserve 10% capacity margin to avoid high delay
            return (path_capacity[paths[j]] * 0.9) - total_load

        constraints.append({'type': 'ineq', 'fun': constraint_func})

    # Non-negative constraints
    bounds = [(0, None) for _ in range(num_traffic_types * num_paths)]

    # Initial guess (use ECMP allocation as starting point)
    x0 = []
    for t_type in traffic_types:
        x0.extend([traffic_demand[t_type] / num_paths] * num_paths)

    # Solve optimization problem (increase optimizer robustness)
    try:
        # Try multiple optimization methods
        for method in ['SLSQP', 'COBYLA']:
            result = minimize(
                objective,
                x0,
                method=method,
                constraints=constraints,
                bounds=bounds,
                options={'maxiter': 5000, 'ftol': 1e-6, 'disp': True}
            )

            if result.success:
                print(f"Optimization successful with {method}")
                utility, utilizations, delays = calculate_utility(result.x, topology, utility_config)
                return utility, result.x, utilizations, delays
            else:
                print(f"Optimization failed with {method}: {result.message}")

        # Fall back to ECMP if all methods fail
        print("All optimization methods failed, falling back to ECMP")
        utility, utilizations, delays = calculate_utility(x0, topology, utility_config)
        return utility, x0, utilizations, delays

    except Exception as e:
        print("Optimization error:", e)
        utility, utilizations, delays = calculate_utility(x0, topology, utility_config)
        return utility, x0, utilizations, delays


def ecmp_allocation(topology, traffic_demand):
    """ECMP allocation strategy (unchanged)"""
    paths = topology["paths"]
    num_paths = len(paths)

    allocations = []
    for t_type in ["latency", "throughput", "medium"]:
        allocations.extend([traffic_demand[t_type] / num_paths] * num_paths)

    return allocations


def spf_allocation(topology, traffic_demand):
    """SPF allocation strategy (unchanged)"""
    paths = topology["paths"]
    path_capacity = topology["capacity"]
    base_delay = topology["delay"]

    num_paths = len(paths)

    # Find the path with the lowest delay
    min_delay_path_idx = np.argmin([base_delay[path] for path in paths])
    min_delay_path = paths[min_delay_path_idx]
    min_path_capacity = path_capacity[min_delay_path]

    # Total demand
    total_demand = sum(traffic_demand.values())

    # If the lowest delay path has sufficient capacity, allocate all traffic to it
    if min_path_capacity >= total_demand:
        allocations = []
        for t_type in ["latency", "throughput", "medium"]:
            for i in range(num_paths):
                if i == min_delay_path_idx:
                    allocations.append(traffic_demand[t_type])
                else:
                    allocations.append(0)
        return allocations
    else:
        # Insufficient capacity, scale down proportionally
        scale_factor = min_path_capacity / total_demand
        allocations = []
        for t_type in ["latency", "throughput", "medium"]:
            for i in range(num_paths):
                if i == min_delay_path_idx:
                    allocations.append(traffic_demand[t_type] * scale_factor)
                else:
                    allocations.append(0)
        return allocations


# =============================================================================
# Experiment Execution and Result Analysis
# =============================================================================

def run_experiment(scenario_abbr, topology_type, traffic_scenario, utility_type, strategy):
    """Run a single experiment and return results"""
    # Get scenario configuration
    traffic_demand = get_traffic_scenario(traffic_scenario)
    topology_config = get_topology_config(topology_type)
    utility_config = get_utility_config(utility_type)
    scenario_name = scenario_names[scenario_abbr]

    print(f"Running: {scenario_name}, {utility_config['name']}, {strategy}")

    # Calculate allocation and utility based on strategy
    if strategy == "UtiliNet":
        utility, allocations, utilizations, delays = UtiliNet_optimization(
            topology_config, traffic_demand, utility_config)
    elif strategy == "ECMP":
        allocations = ecmp_allocation(topology_config, traffic_demand)
        utility, utilizations, delays = calculate_utility(allocations, topology_config, utility_config)
    elif strategy == "SPF":
        allocations = spf_allocation(topology_config, traffic_demand)
        utility, utilizations, delays = calculate_utility(allocations, topology_config, utility_config)

    # Calculate performance metrics
    avg_delay, max_utilization, total_throughput, std_delay = calculate_metrics(
        allocations, topology_config)

    # Return results
    return {
        "Scenario": scenario_abbr,
        "Scenario_Name": scenario_name,
        "Utility_Type": utility_config["name"],
        "Strategy": strategy,
        "Utility": utility,
        "Avg_Delay": avg_delay,
        "Max_Utilization": max_utilization,
        "Total_Throughput": total_throughput,
        "Std_Delay": std_delay,
        "Allocations": allocations,
        "Utilizations": utilizations,
        "Delays": delays
    }


def run_all_experiments():
    """Run all experiment combinations"""
    results = []

    # Set random seed for reproducible results
    np.random.seed(42)

    print("Starting experiments with improved models...")

    # Define all experiment parameters
    experiment_configs = [
        # Baseline scenarios
        {"scenario_abbr": "B4", "topology": "4path", "traffic": "baseline"},
        {"scenario_abbr": "B6", "topology": "6path", "traffic": "baseline"},
        {"scenario_abbr": "B3", "topology": "3path", "traffic": "baseline"},
        {"scenario_abbr": "B8", "topology": "8path", "traffic": "baseline"},

        # High-load scenarios
        {"scenario_abbr": "H4", "topology": "4path", "traffic": "high_load"},
        {"scenario_abbr": "H6", "topology": "6path", "traffic": "high_load"},
        {"scenario_abbr": "H3", "topology": "3path", "traffic": "high_load"},
        {"scenario_abbr": "H8", "topology": "8path", "traffic": "high_load"},

        # Asymmetric scenarios
        {"scenario_abbr": "A4", "topology": "4path", "traffic": "asymmetric"},
        {"scenario_abbr": "A6", "topology": "6path", "traffic": "asymmetric"},
        {"scenario_abbr": "A3", "topology": "3path", "traffic": "asymmetric"},
        {"scenario_abbr": "A8", "topology": "8path", "traffic": "asymmetric"}
    ]

    utility_types = ["baseline", "sensitive", "linear"]
    strategies = ["UtiliNet", "ECMP", "SPF"]

    # Iterate through all combinations
    for config in experiment_configs:
        for utility_type in utility_types:
            for strategy in strategies:
                result = run_experiment(
                    config["scenario_abbr"],
                    config["topology"],
                    config["traffic"],
                    utility_type,
                    strategy
                )
                results.append(result)

    return results


def analyze_results(results):
    """Analyze experiment results and generate visualizations"""
    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Save complete results
    df.to_csv(os.path.join(data_dir, "all_experiment_results_with_metrics.csv"), index=False)

    # Create pivot table
    pivot_table = df.pivot_table(
        values="Utility",
        index=["Scenario", "Scenario_Name", "Utility_Type"],
        columns="Strategy",
        aggfunc=np.mean
    )
    pivot_table.to_csv(os.path.join(data_dir, "utility_results_pivot.csv"))

    # Create comparison charts
    plt.figure(figsize=(15, 10))
    sns.barplot(x="Scenario", y="Utility", hue="Strategy", data=df)
    plt.title("Utility Comparison Across Scenarios")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "utility_comparison.png"))

    plt.figure(figsize=(15, 10))
    sns.barplot(x="Scenario", y="Avg_Delay", hue="Strategy", data=df)
    plt.title("Average Delay Comparison Across Scenarios")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "delay_comparison.png"))

    # Strategy comparison analysis
    strategy_comparison = df.groupby("Strategy").agg({
        "Utility": ["mean", "std"],
        "Avg_Delay": ["mean", "std"],
        "Max_Utilization": ["mean", "std"],
        "Total_Throughput": ["mean", "std"]
    })
    strategy_comparison.to_csv(os.path.join(data_dir, "strategy_comparison.csv"))

    # Scenario comparison analysis
    scenario_comparison = df.groupby(["Scenario", "Strategy"]).agg({
        "Utility": "mean",
        "Avg_Delay": "mean"
    }).unstack()
    scenario_comparison.to_csv(os.path.join(data_dir, "scenario_comparison.csv"))

    return df


# =============================================================================
# Execute Experiments and Analyze Results
# =============================================================================

if __name__ == "__main__":
    print("Starting all experiments...")
    results = run_all_experiments()
    df = analyze_results(results)

    print("\nExperiments completed successfully!")
    print(f"Total results: {len(results)}")
    print(f"Results saved to {data_dir}")

    # Print summary statistics
    print("\nStrategy Performance Summary:")
    print(df.groupby("Strategy")[["Utility", "Avg_Delay"]].mean())

    # Print best strategy
    best_strategy = df.groupby("Strategy")["Utility"].mean().idxmax()
    print(f"\nOverall best strategy: {best_strategy}")

    # Print worst strategy
    worst_strategy = df.groupby("Strategy")["Utility"].mean().idxmin()
    print(f"Overall worst strategy: {worst_strategy}")

    # Print scenarios where UtiliNet has advantage
    UtiliNet_advantage = df[df["Strategy"] == "UtiliNet"].groupby("Scenario")["Utility"].mean()
    ecmp_utility = df[df["Strategy"] == "ECMP"].groupby("Scenario")["Utility"].mean()
    advantage_scenarios = UtiliNet_advantage[UtiliNet_advantage > ecmp_utility].index.tolist()
    print(f"\nScenarios where UtiliNet outperforms ECMP: {', '.join(advantage_scenarios)}")