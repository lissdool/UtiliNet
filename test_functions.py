import numpy as np
import sys
import os

# 添加当前目录到Python路径，以便导入sim_static模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入函数
from sim_static import calculate_delay, calculate_utility, calculate_metrics, get_topology_config, get_utility_config

def test_calculate_delay():
    """测试calculate_delay函数"""
    print("=" * 60)
    print("测试 calculate_delay 函数")
    print("=" * 60)
    
    # 测试用例
    test_cases = [
        (20, 0.0, "零利用率"),
        (20, 0.1, "低利用率"),
        (20, 0.5, "中等利用率"),
        (20, 0.8, "高利用率"),
        (20, 0.95, "接近拥塞"),
        (20, 0.99, "拥塞状态"),
        (20, 1.0, "完全拥塞")
    ]
    
    for base_delay, utilization, description in test_cases:
        result = calculate_delay(base_delay, utilization)
        print(f"基础延迟: {base_delay}ms, 利用率: {utilization:.2f} ({description})")
        print(f"  计算延迟: {result:.2f}ms")
        print(f"  延迟倍数: {result/base_delay:.2f}x")
        print()

def test_calculate_utility():
    """测试calculate_utility函数"""
    print("=" * 60)
    print("测试 calculate_utility 函数")
    print("=" * 60)
    
    # 获取拓扑和效用配置
    topology = get_topology_config("4path")
    utility_config = get_utility_config("baseline")
    
    print("拓扑配置:")
    print(f"  路径: {topology['paths']}")
    print(f"  容量: {topology['capacity']}")
    print(f"  基础延迟: {topology['delay']}")
    print()
    
    print("效用配置:")
    print(f"  名称: {utility_config['name']}")
    print(f"  延迟系数: {utility_config['latency_k']}")
    print(f"  中等流量系数: {utility_config['medium_k']}")
    print(f"  归一化因子: {utility_config['normalization_factor']}")
    print()
    
    # 测试用例1: 均衡分配
    print("测试用例1: 均衡分配")
    allocations_balanced = [1.0, 1.0, 1.0, 1.0,  # 延迟敏感流量
                          2.0, 2.0, 2.0, 2.0,  # 吞吐量敏感流量
                          0.5, 0.5, 0.5, 0.5]   # 中等流量
    
    utility, utilizations, delays = calculate_utility(allocations_balanced, topology, utility_config)
    print(f"分配方案: {allocations_balanced}")
    print(f"效用值: {utility:.4f}")
    print(f"路径利用率: {[f'{u:.3f}' for u in utilizations]}")
    print(f"路径延迟: {[f'{d:.2f}ms' for d in delays]}")
    print()
    
    # 测试用例2: 集中分配（可能拥塞）
    print("测试用例2: 集中分配")
    allocations_concentrated = [3.0, 0.0, 0.0, 0.0,  # 延迟敏感流量集中在第一条路径
                              5.0, 0.0, 0.0, 0.0,  # 吞吐量敏感流量集中在第一条路径
                              1.0, 0.0, 0.0, 0.0]   # 中等流量集中在第一条路径
    
    utility, utilizations, delays = calculate_utility(allocations_concentrated, topology, utility_config)
    print(f"分配方案: {allocations_concentrated}")
    print(f"效用值: {utility:.4f}")
    print(f"路径利用率: {[f'{u:.3f}' for u in utilizations]}")
    print(f"路径延迟: {[f'{d:.2f}ms' for d in delays]}")
    print()
    
    # 测试用例3: 过度分配（测试拥塞惩罚）
    print("测试用例3: 过度分配")
    allocations_overload = [8.0, 0.0, 0.0, 0.0,  # 超过第一条路径容量
                          8.0, 0.0, 0.0, 0.0,  # 超过第一条路径容量
                          2.0, 0.0, 0.0, 0.0]   # 超过第一条路径容量
    
    utility, utilizations, delays = calculate_utility(allocations_overload, topology, utility_config)
    print(f"分配方案: {allocations_overload}")
    print(f"效用值: {utility:.4f}")
    print(f"路径利用率: {[f'{u:.3f}' for u in utilizations]}")
    print(f"路径延迟: {[f'{d:.2f}ms' for d in delays]}")
    print()

def test_calculate_metrics():
    """测试calculate_metrics函数"""
    print("=" * 60)
    print("测试 calculate_metrics 函数")
    print("=" * 60)
    
    # 获取拓扑配置
    topology = get_topology_config("4path")
    
    # 测试用例1: 均衡分配
    print("测试用例1: 均衡分配")
    allocations_balanced = [1.0, 1.0, 1.0, 1.0,  # 延迟敏感流量
                          2.0, 2.0, 2.0, 2.0,  # 吞吐量敏感流量
                          0.5, 0.5, 0.5, 0.5]   # 中等流量
    
    avg_delay, max_util, actual_throughput, delay_std = calculate_metrics(allocations_balanced, topology)
    print(f"分配方案: {allocations_balanced}")
    print(f"平均延迟: {avg_delay:.2f}ms")
    print(f"最大链路利用率: {max_util:.3f}")
    print(f"实际吞吐量: {actual_throughput:.2f}")
    print(f"延迟标准差: {delay_std:.2f}ms")
    print()
    
    # 测试用例2: 集中分配
    print("测试用例2: 集中分配")
    allocations_concentrated = [3.0, 0.0, 0.0, 0.0,  # 延迟敏感流量集中在第一条路径
                              5.0, 0.0, 0.0, 0.0,  # 吞吐量敏感流量集中在第一条路径
                              1.0, 0.0, 0.0, 0.0]   # 中等流量集中在第一条路径
    
    avg_delay, max_util, actual_throughput, delay_std = calculate_metrics(allocations_concentrated, topology)
    print(f"分配方案: {allocations_concentrated}")
    print(f"平均延迟: {avg_delay:.2f}ms")
    print(f"最大链路利用率: {max_util:.3f}")
    print(f"实际吞吐量: {actual_throughput:.2f}")
    print(f"延迟标准差: {delay_std:.2f}ms")
    print()
    
    # 测试用例3: 不同拓扑测试
    print("测试用例3: 不同拓扑测试")
    topologies_to_test = ["4path", "6path", "3path", "8path"]
    
    for topo_name in topologies_to_test:
        topology = get_topology_config(topo_name)
        num_paths = len(topology["paths"])
        
        # 创建简单的均衡分配方案
        allocations = []
        for _ in range(3):  # 三种流量类型
            allocations.extend([1.0] * num_paths)
        
        avg_delay, max_util, actual_throughput, delay_std = calculate_metrics(allocations, topology)
        print(f"拓扑: {topo_name} ({num_paths}路径)")
        print(f"  平均延迟: {avg_delay:.2f}ms")
        print(f"  最大链路利用率: {max_util:.3f}")
        print(f"  实际吞吐量: {actual_throughput:.2f}")
        print(f"  延迟标准差: {delay_std:.2f}ms")
        print()

def run_all_tests():
    """运行所有测试"""
    print("开始测试核心计算函数...")
    print()
    
    test_calculate_delay()
    test_calculate_utility()
    test_calculate_metrics()
    
    print("=" * 60)
    print("所有测试完成!")
    print("=" * 60)

if __name__ == "__main__":
    run_all_tests()