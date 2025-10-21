import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add current directory to Python path to import sim_static module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import functions
from sim_static import calculate_delay, calculate_utility, calculate_metrics, get_topology_config, get_utility_config

# Set font and chart style
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def visualize_delay_function():
    """Visualize delay calculation function"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Test delay curves for different base delays
    base_delays = [10, 20, 30, 50]
    utilizations = np.linspace(0, 1, 100)
    
    # Subplot 1: Delay curves for different base delays
    for base_delay in base_delays:
        delays = [calculate_delay(base_delay, u) for u in utilizations]
        axes[0].plot(utilizations, delays, label=f'Base Delay={base_delay}ms', linewidth=2)
    
    axes[0].set_xlabel('Link Utilization')
    axes[0].set_ylabel('Delay (ms)')
    axes[0].set_title('Delay-Utilization Relationship for Different Base Delays')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Subplot 2: Delay ratio (relative to base delay)
    base_delay = 20
    delays = [calculate_delay(base_delay, u) for u in utilizations]
    delay_ratios = [d / base_delay for d in delays]
    
    axes[1].plot(utilizations, delay_ratios, 'r-', linewidth=2)
    axes[1].axvline(x=0.99, color='red', linestyle='--', alpha=0.7, label='Congestion Threshold (99%)')
    axes[1].set_xlabel('Link Utilization')
    axes[1].set_ylabel('Delay Ratio (Relative to Base Delay)')
    axes[1].set_title('Delay Ratio vs Utilization Relationship')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('delay_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_utility_comparison():
    """Visualize utility comparison of different allocation strategies"""
    topology = get_topology_config("4path")
    utility_config = get_utility_config("baseline")
    
    # Create different allocation strategies
    strategies = {
        'Balanced': [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.5],
        'Concentrated': [3.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        'Overloaded': [8.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
        'Smart': [2.0, 1.5, 1.0, 0.5, 3.0, 2.5, 2.0, 1.5, 0.8, 0.6, 0.4, 0.2]
    }
    
    results = {}
    for strategy_name, allocations in strategies.items():
        utility, utilizations, delays = calculate_utility(allocations, topology, utility_config)
        avg_delay, max_util, actual_throughput, delay_std = calculate_metrics(allocations, topology)
        
        results[strategy_name] = {
            'utility': utility,
            'avg_delay': avg_delay,
            'max_util': max_util,
            'throughput': actual_throughput,
            'delay_std': delay_std,
            'utilizations': utilizations,
            'delays': delays
        }
    
    # Create comparison charts
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Subplot 1: Utility value comparison
    utilities = [results[s]['utility'] for s in strategies]
    axes[0,0].bar(strategies.keys(), utilities, color=['blue', 'orange', 'red', 'green'])
    axes[0,0].set_title('Utility Value Comparison of Different Allocation Strategies')
    axes[0,0].set_ylabel('Utility Value')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Subplot 2: Average delay comparison
    avg_delays = [results[s]['avg_delay'] for s in strategies]
    axes[0,1].bar(strategies.keys(), avg_delays, color=['blue', 'orange', 'red', 'green'])
    axes[0,1].set_title('Average Delay Comparison of Different Allocation Strategies')
    axes[0,1].set_ylabel('Average Delay (ms)')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Subplot 3: Maximum link utilization comparison
    max_utils = [results[s]['max_util'] for s in strategies]
    axes[1,0].bar(strategies.keys(), max_utils, color=['blue', 'orange', 'red', 'green'])
    axes[1,0].set_title('Maximum Link Utilization Comparison of Different Allocation Strategies')
    axes[1,0].set_ylabel('Maximum Link Utilization')
    axes[1,0].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Capacity Limit')
    axes[1,0].legend()
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Subplot 4: Actual throughput comparison
    throughputs = [results[s]['throughput'] for s in strategies]
    axes[1,1].bar(strategies.keys(), throughputs, color=['blue', 'orange', 'red', 'green'])
    axes[1,1].set_title('Actual Throughput Comparison of Different Allocation Strategies')
    axes[1,1].set_ylabel('Actual Throughput')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('utility_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed results
    print("=" * 60)
    print("Detailed Comparison of Allocation Strategies")
    print("=" * 60)
    for strategy_name in strategies:
        result = results[strategy_name]
        print(f"\nStrategy: {strategy_name}")
        print(f"  Utility Value: {result['utility']:.4f}")
        print(f"  Average Delay: {result['avg_delay']:.2f}ms")
        print(f"  Max Link Utilization: {result['max_util']:.3f}")
        print(f"  Actual Throughput: {result['throughput']:.2f}")
        print(f"  Delay Standard Deviation: {result['delay_std']:.2f}ms")
        print(f"  Path Utilizations: {[f'{u:.3f}' for u in result['utilizations']]}")
        print(f"  Path Delays: {[f'{d:.2f}ms' for d in result['delays']]}")

def visualize_topology_comparison():
    """Visualize performance comparison of different topologies"""
    topologies = ['4path', '6path', '3path', '8path']
    utility_config = get_utility_config("baseline")
    
    results = {}
    for topo_name in topologies:
        topology = get_topology_config(topo_name)
        num_paths = len(topology["paths"])
        
        # Create balanced allocation scheme
        allocations = []
        for _ in range(3):  # Three traffic types
            allocations.extend([2.0] * num_paths)
        
        utility, utilizations, delays = calculate_utility(allocations, topology, utility_config)
        avg_delay, max_util, actual_throughput, delay_std = calculate_metrics(allocations, topology)
        
        results[topo_name] = {
            'utility': utility,
            'avg_delay': avg_delay,
            'max_util': max_util,
            'throughput': actual_throughput,
            'delay_std': delay_std,
            'num_paths': num_paths
        }
    
    # Create topology comparison charts
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Subplot 1: Utility vs number of paths
    utilities = [results[t]['utility'] for t in topologies]
    num_paths = [results[t]['num_paths'] for t in topologies]
    axes[0,0].scatter(num_paths, utilities, s=100, alpha=0.7)
    for i, topo in enumerate(topologies):
        axes[0,0].annotate(topo, (num_paths[i], utilities[i]), xytext=(5, 5), textcoords='offset points')
    axes[0,0].set_xlabel('Number of Paths')
    axes[0,0].set_ylabel('Utility Value')
    axes[0,0].set_title('Utility Value vs Number of Paths in Topology')
    axes[0,0].grid(True, alpha=0.3)
    
    # Subplot 2: Average delay comparison
    avg_delays = [results[t]['avg_delay'] for t in topologies]
    axes[0,1].bar(topologies, avg_delays, color=['blue', 'orange', 'green', 'red'])
    axes[0,1].set_title('Average Delay Comparison Across Topologies')
    axes[0,1].set_ylabel('Average Delay (ms)')
    
    # Subplot 3: Maximum link utilization comparison
    max_utils = [results[t]['max_util'] for t in topologies]
    axes[1,0].bar(topologies, max_utils, color=['blue', 'orange', 'green', 'red'])
    axes[1,0].set_title('Maximum Link Utilization Comparison Across Topologies')
    axes[1,0].set_ylabel('Maximum Link Utilization')
    axes[1,0].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Capacity Limit')
    axes[1,0].legend()
    
    # Subplot 4: Actual throughput comparison
    throughputs = [results[t]['throughput'] for t in topologies]
    axes[1,1].bar(topologies, throughputs, color=['blue', 'orange', 'green', 'red'])
    axes[1,1].set_title('Actual Throughput Comparison Across Topologies')
    axes[1,1].set_ylabel('Actual Throughput')
    
    plt.tight_layout()
    plt.savefig('topology_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed results
    print("\n" + "=" * 60)
    print("Detailed Topology Performance Comparison")
    print("=" * 60)
    for topo_name in topologies:
        result = results[topo_name]
        print(f"\nTopology: {topo_name} ({result['num_paths']} paths)")
        print(f"  Utility Value: {result['utility']:.4f}")
        print(f"  Average Delay: {result['avg_delay']:.2f}ms")
        print(f"  Max Link Utilization: {result['max_util']:.3f}")
        print(f"  Actual Throughput: {result['throughput']:.2f}")
        print(f"  Delay Standard Deviation: {result['delay_std']:.2f}ms")

def visualize_utility_function_variants():
    """Visualize comparison of different utility function variants"""
    topology = get_topology_config("4path")
    utility_types = ['baseline', 'sensitive', 'linear']
    
    # Test different allocation schemes
    allocations = [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.5]
    
    results = {}
    for util_type in utility_types:
        utility_config = get_utility_config(util_type)
        utility, utilizations, delays = calculate_utility(allocations, topology, utility_config)
        
        results[util_type] = {
            'utility': utility,
            'config': utility_config
        }
    
    # Create utility function comparison chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    utility_values = [results[t]['utility'] for t in utility_types]
    config_names = [results[t]['config']['name'] for t in utility_types]
    
    bars = ax.bar(config_names, utility_values, color=['blue', 'orange', 'green'])
    ax.set_title('Utility Value Comparison of Different Utility Function Variants')
    ax.set_ylabel('Utility Value')
    
    # Add value labels on bars
    for bar, value in zip(bars, utility_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('utility_variants_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed configurations
    print("\n" + "=" * 60)
    print("Detailed Configuration of Utility Function Variants")
    print("=" * 60)
    for util_type in utility_types:
        config = results[util_type]['config']
        utility = results[util_type]['utility']
        print(f"\nUtility Function: {config['name']}")
        print(f"  Latency Coefficient: {config['latency_k']}")
        print(f"  Medium Traffic Coefficient: {config['medium_k']}")
        print(f"  Normalization Factor: {config['normalization_factor']}")
        print(f"  Calculated Utility Value: {utility:.4f}")

def run_all_visualizations():
    """Run all visualizations"""
    print("Starting visualization generation...")
    print()
    
    # 1. Delay function visualization
    print("1. Generating delay function visualization...")
    visualize_delay_function()
    
    # 2. Utility comparison visualization
    print("\n2. Generating allocation strategy comparison visualization...")
    visualize_utility_comparison()
    
    # 3. Topology comparison visualization
    print("\n3. Generating topology performance comparison visualization...")
    visualize_topology_comparison()
    
    # 4. Utility function variants comparison
    print("\n4. Generating utility function variants comparison visualization...")
    visualize_utility_function_variants()
    
    print("\n" + "=" * 60)
    print("All visualizations completed!")
    print("Charts saved as PNG files:")
    print("- delay_visualization.png")
    print("- utility_comparison.png") 
    print("- topology_comparison.png")
    print("- utility_variants_comparison.png")
    print("=" * 60)

if __name__ == "__main__":
    run_all_visualizations()