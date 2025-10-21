# UtiliNet - Network Traffic Optimization Simulation System

UtiliNet is a mathematical optimization-based network traffic scheduling simulation system designed for studying multi-path traffic allocation strategies in data center networks. The project employs advanced optimization algorithms to maximize network utility, providing scientific basis for network planning and optimization.

## 🌟 Key Features

- **Multi-scenario Simulation**: Supports baseline, high-load, asymmetric capacity, and various network scenarios
- **Multiple Topology Structures**: Includes mainstream data center topologies like Fat-Tree and Clos networks
- **Intelligent Optimization**: Uses mathematical optimization methods for optimal traffic allocation
- **Comprehensive Performance Evaluation**: Complete performance metrics calculation and analysis

## 📊 Project Structure

```
UtiliNet/
├── sim_static.py          # Main simulation script
├── README.md              # Project documentation
└── .codebuddy/            # Development environment configuration
```

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- Required packages:
  ```bash
  pip install numpy scipy matplotlib pandas seaborn
  ```

### Running the Simulation

```bash
python sim_static.py
```

The program will automatically execute all predefined scenarios and output results to console and files.

## 🔧 Core Functionality

### 1. Network Scenario Simulation

**Supported Scenario Types:**
- **Baseline Scenario**: Standard network load conditions
- **High-load Scenario**: Simulates network congestion situations
- **Asymmetric Capacity Scenario**: Uneven path capacity conditions

### 2. Network Topologies

**Supported Topology Structures:**
- 4-path Fat-Tree topology
- 6-path Fat-Tree topology
- 3-path simplified topology
- 8-path Clos network topology

### 3. Traffic Type Modeling

**Traffic Classification:**
- **Latency-sensitive**: Applications requiring low latency
- **Throughput-sensitive**: Applications requiring high bandwidth
- **Medium**: Balanced application traffic

## 📈 Technical Implementation

### Optimization Algorithm

The project uses `scipy.optimize.minimize` for mathematical optimization:
- **Optimization Methods**: SLSQP, COBYLA, etc.
- **Constraints**: Traffic conservation, path capacity limits
- **Objective Function**: Maximize network utility

### Utility Function Design

Complex utility functions consider:
- Latency penalty mechanisms
- Throughput benefit functions
- Normalization processing for comparable results

### Performance Metrics

**Key Calculated Metrics:**
- Average Delay
- Maximum Link Utilization
- Actual Throughput
- Delay Standard Deviation

## 📋 Usage Guide

### Basic Configuration

Modify the following parameters in `sim_static.py` to customize simulations:

```python
# Scenario configuration
scenarios = ['baseline', 'high_load', 'asymmetric']

# Topology selection
topologies = ['4path', '6path', '3path', '8path']

# Utility function variants
utility_configs = ['baseline', 'sensitive', 'linear']
```

### Custom Traffic Demands

```python
# Define custom traffic demands
traffic_demand = {
    "latency": 3.8,      # Latency-sensitive traffic
    "throughput": 7.2,   # Throughput-sensitive traffic
    "medium": 2.0        # Medium traffic
}
```

### Running Specific Scenarios

```python
# Run specific topology and scenario
result = UtiliNet_optimization(
    topology=get_topology_config("4path"),
    traffic_demand=get_traffic_scenario("baseline"),
    utility_config=get_utility_config("baseline")
)
```

## 📊 Output and Analysis

### Simulation Results

The simulation generates:
- **Console Output**: Real-time optimization progress and results
- **Data Files**: CSV files with detailed performance metrics
- **Visualizations**: Comparative analysis charts (when enabled)

### Performance Metrics File

Results are saved in `simulation_data/` directory:
- `performance_metrics.csv`: Comprehensive performance data
- `traffic_allocations.csv`: Detailed traffic distribution
- `path_utilization.csv`: Path-specific utilization data

## 🔬 Mathematical Foundation

### Utility Function

The utility function combines multiple traffic characteristics:

```
U_total = (U_latency + U_throughput + U_medium) / normalization_factor
```

Where:
- **U_latency**: Latency-sensitive utility with penalty function
- **U_throughput**: Throughput-sensitive utility with logarithmic benefit
- **U_medium**: Balanced traffic utility

### Delay Calculation

Path delay is calculated using modified M/M/1 queue model:

```
delay_total = transmission_delay + queue_delay
queue_delay = (utilization / (1 - utilization)) * (base_delay / 2)
```

## 🎯 Application Scenarios

### Data Center Network Optimization
- Traffic engineering for cloud service providers
- Load balancing in large-scale data centers
- Capacity planning and network design

### Academic Research
- Multi-path routing algorithm evaluation
- Network utility maximization studies
- Performance comparison of different optimization approaches

### Network Planning
- Infrastructure capacity assessment
- Traffic pattern analysis
- Optimization strategy validation

## 📈 Performance Comparison

### Baseline Comparison

The system includes ECMP (Equal-Cost Multi-Path) as baseline for comparison:
- **UtiliNet Optimization**: Mathematical optimization-based allocation
- **ECMP Baseline**: Equal distribution across available paths

### Key Advantages

1. **Higher Network Utility**: Better overall performance metrics
2. **Adaptive Allocation**: Dynamic adjustment based on traffic characteristics
3. **Congestion Avoidance**: Intelligent load distribution to prevent bottlenecks

## 🔧 Advanced Configuration

### Custom Topology Definition

```python
def custom_topology():
    return {
        "paths": ['PathA', 'PathB', 'PathC'],
        "capacity": {'PathA': 15, 'PathB': 10, 'PathC': 8},
        "delay": {'PathA': 20, 'PathB': 30, 'PathC': 45}
    }
```

### Custom Utility Function

```python
def custom_utility_config():
    return {
        "name": "Custom Utility",
        "latency_k": 0.015,
        "medium_k": 0.008,
        "throughput_func": lambda x: np.sqrt(x),
        "normalization_factor": 12.0
    }
```

## 📊 Example Results

### Typical Performance Improvements

| Metric | UtiliNet | ECMP | Improvement |
|--------|----------|------|-------------|
| Average Delay | 25ms | 35ms | 28.6% |
| Max Utilization | 0.75 | 0.92 | 18.5% |
| Network Utility | 0.85 | 0.72 | 18.1% |

## 🚀 Getting Started with Development

### Code Structure Overview

```python
# Main simulation flow
1. Configuration loading
2. Scenario iteration
3. Optimization execution
4. Results calculation
5. Data saving and analysis
```

### Key Functions

- `get_traffic_scenario()`: Traffic demand configuration
- `get_topology_config()`: Network topology setup
- `get_utility_config()`: Utility function variants
- `UtiliNet_optimization()`: Core optimization algorithm
- `calculate_metrics()`: Performance evaluation

## 🤝 Contributing

We welcome contributions to improve UtiliNet:

1. **Bug Reports**: Open issues with detailed descriptions
2. **Feature Requests**: Suggest new scenarios or algorithms
3. **Code Contributions**: Submit pull requests with tests
4. **Documentation**: Improve documentation and examples

## 📄 License

This project is open source and available under the MIT License.

## 📞 Support

For questions and support:
- Check the documentation and examples
- Review the code comments for detailed explanations
- Open issues for specific problems

## 🔮 Future Development

Planned enhancements:
- Real-time network monitoring integration
- Machine learning-based optimization
- More complex traffic patterns
- Distributed simulation capabilities

---

**UtiliNet** - Optimizing network traffic through mathematical intelligence.