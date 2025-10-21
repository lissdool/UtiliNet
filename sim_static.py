import numpy as np
from scipy.optimize import minimize
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 创建数据保存目录
data_dir = "simulation_data"
os.makedirs(data_dir, exist_ok=True)

# =============================================================================
# 场景和拓扑配置
# =============================================================================

# 定义场景缩写名称
scenario_names = {
    # 基准场景
    "B4": "Baseline 4-path",
    "B6": "Baseline 6-path",
    "B3": "Baseline 3-path",
    "B8": "Baseline 8-path",

    # 高负载场景
    "H4": "High-load 4-path",
    "H6": "High-load 6-path",
    "H3": "High-load 3-path",
    "H8": "High-load 8-path",

    # 非对称容量场景
    "A4": "Asymmetric 4-path",
    "A6": "Asymmetric 6-path",
    "A3": "Asymmetric 3-path",
    "A8": "Asymmetric 8-path"
}


# 定义三种流量模型
def get_traffic_scenario(scenario_type):
    """获取不同流量模型的配置"""
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


# 定义四种拓扑
def get_topology_config(topology_type):
    """获取不同拓扑的配置"""
    topologies = {
        # 标准4路径Fat-Tree
        "4path": {
            "paths": ['Path1', 'Path2', 'Path3', 'Path4'],
            "capacity": {'Path1': 10, 'Path2': 10, 'Path3': 5, 'Path4': 5},
            "delay": {'Path1': 20, 'Path2': 25, 'Path3': 40, 'Path4': 40}
        },
        # 6路径Fat-Tree
        "6path": {
            "paths": ['Core1-ToR1', 'Core1-ToR2', 'Core2-ToR1', 'Core2-ToR2', 'Core3-ToR1', 'Core3-ToR2'],
            "capacity": {'Core1-ToR1': 10, 'Core1-ToR2': 10, 'Core2-ToR1': 10, 'Core2-ToR2': 10,
                         'Core3-ToR1': 5, 'Core3-ToR2': 5},
            "delay": {'Core1-ToR1': 20, 'Core1-ToR2': 25, 'Core2-ToR1': 22, 'Core2-ToR2': 27,
                      'Core3-ToR1': 40, 'Core3-ToR2': 45}
        },
        # 3路径简化拓扑
        "3path": {
            "paths": ['Primary', 'Backup1', 'Backup2'],
            "capacity": {'Primary': 15, 'Backup1': 8, 'Backup2': 5},
            "delay": {'Primary': 15, 'Backup1': 35, 'Backup2': 50}
        },
        # 8路径Clos网络
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
# 效用函数配置 (修改版本)
# =============================================================================

def get_utility_config(utility_type):
    """获取不同效用函数变体的配置（修正版）"""
    utility_configs = {
        "baseline": {
            "name": "Baseline",
            "latency_k": 0.01,  # 降低系数以平衡权重
            "medium_k": 0.005,  # 降低系数以平衡权重
            "throughput_func": lambda x: np.log(1 + x),
            "normalization_factor": 10.0  # 新增归一化因子
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
            "throughput_func": lambda x: x * 0.5,  # 降低线性权重
            "normalization_factor": 15.0  # 提高归一化因子
        }
    }
    return utility_configs.get(utility_type, utility_configs["baseline"])


# =============================================================================
# 核心计算函数 (修正版本)
# =============================================================================

def calculate_delay(base_delay, utilization):
    """计算路径延迟 (修正的M/M/1队列模型)"""
    if utilization >= 0.99:
        return base_delay * 100  # 拥塞时惩罚性高时延
    elif utilization <= 0.01:
        return base_delay  # 低利用率时基础时延
    else:
        # 标准排队模型：总时延 = 传输时延 + 排队时延
        # 更合理的计算方式：传输时延 + 排队时延
        transmission_delay = base_delay
        queue_delay = (utilization / (1 - utilization)) * (base_delay / 2)  # 标准化排队时延
        return transmission_delay + queue_delay


def calculate_utility(allocations, topology, utility_config):
    """计算给定分配方案的效用值 (修正版本)"""
    paths = topology["paths"]
    path_capacity = topology["capacity"]
    base_delay = topology["delay"]

    num_paths = len(paths)
    num_traffic_types = len(allocations) // num_paths

    # 重组分配矩阵
    traffic_alloc = {}
    traffic_types = ["latency", "throughput", "medium"][:num_traffic_types]

    for i, t_type in enumerate(traffic_types):
        traffic_alloc[t_type] = allocations[i * num_paths:(i + 1) * num_paths]

    # 计算路径利用率和延迟
    utilizations = []
    delays = []
    for i, path in enumerate(paths):
        total_traffic = sum(traffic_alloc[t_type][i] for t_type in traffic_types)
        utilization = total_traffic / path_capacity[path]
        utilizations.append(utilization)
        delays.append(calculate_delay(base_delay[path], utilization))

    # 计算效用值 (使用线性惩罚模型替代指数模型)
    U_lat = sum(traffic_alloc["latency"][i] * max(0, 1 - utility_config["latency_k"] * delays[i])
                for i in range(num_paths))

    total_throughput = sum(traffic_alloc["throughput"])
    U_bw = utility_config["throughput_func"](total_throughput)

    U_med = sum(traffic_alloc["medium"][i] * max(0, 1 - utility_config["medium_k"] * delays[i])
                for i in range(num_paths))

    # 归一化总效用
    normalized_utility = (U_lat + U_bw + U_med) / utility_config["normalization_factor"]

    return normalized_utility, utilizations, delays


def calculate_metrics(allocations, topology):
    """计算性能指标 (修正版本)"""
    paths = topology["paths"]
    path_capacity = topology["capacity"]
    base_delay = topology["delay"]

    num_paths = len(paths)
    num_traffic_types = len(allocations) // num_paths
    traffic_types = ["latency", "throughput", "medium"][:num_traffic_types]

    # 重组分配矩阵
    traffic_alloc = {}
    for i, t_type in enumerate(traffic_types):
        traffic_alloc[t_type] = allocations[i * num_paths:(i + 1) * num_paths]

    # 计算路径利用率和延迟 (使用修正的时延计算)
    utilizations = []
    delays = []
    for i, path in enumerate(paths):
        total_traffic = sum(traffic_alloc[t_type][i] for t_type in traffic_types)
        utilization = total_traffic / path_capacity[path]
        utilizations.append(utilization)
        delays.append(calculate_delay(base_delay[path], utilization))

    # 计算指标
    avg_delay = np.mean(delays)
    max_utilization = np.max(utilizations)
    actual_throughput = 0
    for i, path in enumerate(paths):
        path_traffic = sum(traffic_alloc[t_type][i] for t_type in traffic_types)
        actual_throughput += min(path_traffic, path_capacity[path] * 0.9)  # 考虑容量约束
    std_delay = np.std(delays)

    return avg_delay, max_utilization, actual_throughput, std_delay


def UtiliNet_optimization(topology, traffic_demand, utility_config):
    """UtiliNet优化求解 (修正版本)"""
    paths = topology["paths"]
    path_capacity = topology["capacity"]
    base_delay = topology["delay"]

    num_paths = len(paths)
    traffic_types = ["latency", "throughput", "medium"]
    num_traffic_types = len(traffic_types)

    # 目标函数
    def objective(x):
        utility, _, _ = calculate_utility(x, topology, utility_config)
        return -utility  # 最小化负效用

    # 约束条件
    constraints = []

    # 流量守恒约束
    for i in range(num_traffic_types):
        def constraint_func(x, i=i):
            return sum(x[i * num_paths:(i + 1) * num_paths]) - traffic_demand[traffic_types[i]]

        constraints.append({'type': 'eq', 'fun': constraint_func})

    # 路径容量约束 (修正版：添加余量并确保严格约束)
    for j in range(num_paths):
        def constraint_func(x, j=j):
            total_load = sum(x[i * num_paths + j] for i in range(num_traffic_types))
            # 保留10%容量余量以避免高时延
            return (path_capacity[paths[j]] * 0.9) - total_load

        constraints.append({'type': 'ineq', 'fun': constraint_func})

    # 非负约束
    bounds = [(0, None) for _ in range(num_traffic_types * num_paths)]

    # 初始猜测 (使用ECMP分配作为起点)
    x0 = []
    for t_type in traffic_types:
        x0.extend([traffic_demand[t_type] / num_paths] * num_paths)

    # 求解优化问题 (增加优化器鲁棒性)
    try:
        # 尝试多种优化方法
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

        # 所有方法都失败时回退到ECMP
        print("All optimization methods failed, falling back to ECMP")
        utility, utilizations, delays = calculate_utility(x0, topology, utility_config)
        return utility, x0, utilizations, delays

    except Exception as e:
        print("Optimization error:", e)
        utility, utilizations, delays = calculate_utility(x0, topology, utility_config)
        return utility, x0, utilizations, delays


def ecmp_allocation(topology, traffic_demand):
    """ECMP分配策略 (不变)"""
    paths = topology["paths"]
    num_paths = len(paths)

    allocations = []
    for t_type in ["latency", "throughput", "medium"]:
        allocations.extend([traffic_demand[t_type] / num_paths] * num_paths)

    return allocations


def spf_allocation(topology, traffic_demand):
    """SPF分配策略 (不变)"""
    paths = topology["paths"]
    path_capacity = topology["capacity"]
    base_delay = topology["delay"]

    num_paths = len(paths)

    # 找到延迟最低的路径
    min_delay_path_idx = np.argmin([base_delay[path] for path in paths])
    min_delay_path = paths[min_delay_path_idx]
    min_path_capacity = path_capacity[min_delay_path]

    # 总需求
    total_demand = sum(traffic_demand.values())

    # 如果最低延迟路径容量足够，将所有流量分配给它
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
        # 容量不足，按比例缩减
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
# 实验执行和结果分析
# =============================================================================

def run_experiment(scenario_abbr, topology_type, traffic_scenario, utility_type, strategy):
    """运行单个实验并返回结果"""
    # 获取场景配置
    traffic_demand = get_traffic_scenario(traffic_scenario)
    topology_config = get_topology_config(topology_type)
    utility_config = get_utility_config(utility_type)
    scenario_name = scenario_names[scenario_abbr]

    print(f"Running: {scenario_name}, {utility_config['name']}, {strategy}")

    # 根据策略计算分配和效用
    if strategy == "UtiliNet":
        utility, allocations, utilizations, delays = UtiliNet_optimization(
            topology_config, traffic_demand, utility_config)
    elif strategy == "ECMP":
        allocations = ecmp_allocation(topology_config, traffic_demand)
        utility, utilizations, delays = calculate_utility(allocations, topology_config, utility_config)
    elif strategy == "SPF":
        allocations = spf_allocation(topology_config, traffic_demand)
        utility, utilizations, delays = calculate_utility(allocations, topology_config, utility_config)

    # 计算性能指标
    avg_delay, max_utilization, total_throughput, std_delay = calculate_metrics(
        allocations, topology_config)

    # 返回结果
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
    """运行所有实验组合"""
    results = []

    # 设置随机种子保证结果可复现
    np.random.seed(42)

    print("Starting experiments with improved models...")

    # 定义所有实验参数
    experiment_configs = [
        # Baseline场景
        {"scenario_abbr": "B4", "topology": "4path", "traffic": "baseline"},
        {"scenario_abbr": "B6", "topology": "6path", "traffic": "baseline"},
        {"scenario_abbr": "B3", "topology": "3path", "traffic": "baseline"},
        {"scenario_abbr": "B8", "topology": "8path", "traffic": "baseline"},

        # High-load场景
        {"scenario_abbr": "H4", "topology": "4path", "traffic": "high_load"},
        {"scenario_abbr": "H6", "topology": "6path", "traffic": "high_load"},
        {"scenario_abbr": "H3", "topology": "3path", "traffic": "high_load"},
        {"scenario_abbr": "H8", "topology": "8path", "traffic": "high_load"},

        # Asymmetric场景
        {"scenario_abbr": "A4", "topology": "4path", "traffic": "asymmetric"},
        {"scenario_abbr": "A6", "topology": "6path", "traffic": "asymmetric"},
        {"scenario_abbr": "A3", "topology": "3path", "traffic": "asymmetric"},
        {"scenario_abbr": "A8", "topology": "8path", "traffic": "asymmetric"}
    ]

    utility_types = ["baseline", "sensitive", "linear"]
    strategies = ["UtiliNet", "ECMP", "SPF"]

    # 遍历所有组合
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
    """分析实验结果并生成可视化"""
    # 转换为DataFrame
    df = pd.DataFrame(results)

    # 保存完整结果
    df.to_csv(os.path.join(data_dir, "all_experiment_results_with_metrics.csv"), index=False)

    # 创建透视表
    pivot_table = df.pivot_table(
        values="Utility",
        index=["Scenario", "Scenario_Name", "Utility_Type"],
        columns="Strategy",
        aggfunc=np.mean
    )
    pivot_table.to_csv(os.path.join(data_dir, "utility_results_pivot.csv"))

    # 创建对比图表
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

    # 策略对比分析
    strategy_comparison = df.groupby("Strategy").agg({
        "Utility": ["mean", "std"],
        "Avg_Delay": ["mean", "std"],
        "Max_Utilization": ["mean", "std"],
        "Total_Throughput": ["mean", "std"]
    })
    strategy_comparison.to_csv(os.path.join(data_dir, "strategy_comparison.csv"))

    # 场景对比分析
    scenario_comparison = df.groupby(["Scenario", "Strategy"]).agg({
        "Utility": "mean",
        "Avg_Delay": "mean"
    }).unstack()
    scenario_comparison.to_csv(os.path.join(data_dir, "scenario_comparison.csv"))

    return df


# =============================================================================
# 执行实验并分析结果
# =============================================================================

if __name__ == "__main__":
    print("Starting all experiments...")
    results = run_all_experiments()
    df = analyze_results(results)

    print("\nExperiments completed successfully!")
    print(f"Total results: {len(results)}")
    print(f"Results saved to {data_dir}")

    # 打印摘要统计
    print("\nStrategy Performance Summary:")
    print(df.groupby("Strategy")[["Utility", "Avg_Delay"]].mean())

    # 打印最佳策略
    best_strategy = df.groupby("Strategy")["Utility"].mean().idxmax()
    print(f"\nOverall best strategy: {best_strategy}")

    # 打印最差策略
    worst_strategy = df.groupby("Strategy")["Utility"].mean().idxmin()
    print(f"Overall worst strategy: {worst_strategy}")

    # 打印UtiliNet优势场景
    UtiliNet_advantage = df[df["Strategy"] == "UtiliNet"].groupby("Scenario")["Utility"].mean()
    ecmp_utility = df[df["Strategy"] == "ECMP"].groupby("Scenario")["Utility"].mean()
    advantage_scenarios = UtiliNet_advantage[UtiliNet_advantage > ecmp_utility].index.tolist()
    print(f"\nScenarios where UtiliNet outperforms ECMP: {', '.join(advantage_scenarios)}")