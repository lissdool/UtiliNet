from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim_static import calculate_delay

app = Flask(__name__)
CORS(app)  # 允许跨域请求

@app.route('/api/calculate_delay', methods=['POST'])
def calculate_delay_api():
    """计算延迟的API接口"""
    try:
        # 获取前端发送的JSON数据
        data = request.get_json()
        
        # 验证参数
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        base_delay = data.get('base_delay')
        utilization = data.get('utilization')
        
        if base_delay is None or utilization is None:
            return jsonify({'error': 'Missing required parameters: base_delay and utilization'}), 400
        
        # 验证参数类型和范围
        try:
            base_delay = float(base_delay)
            utilization = float(utilization)
        except ValueError:
            return jsonify({'error': 'Parameters must be numbers'}), 400
        
        if base_delay <= 0:
            return jsonify({'error': 'Base delay must be positive'}), 400
            
        if utilization < 0 or utilization > 1:
            return jsonify({'error': 'Utilization must be between 0 and 1'}), 400
        
        # 调用calculate_delay函数
        calculated_delay = calculate_delay(base_delay, utilization)
        
        # 计算延迟倍数
        delay_ratio = calculated_delay / base_delay
        
        # 返回计算结果
        return jsonify({
            'success': True,
            'base_delay': base_delay,
            'utilization': utilization,
            'calculated_delay': round(calculated_delay, 2),
            'delay_ratio': round(delay_ratio, 2),
            'message': f'Delay calculated successfully: {calculated_delay:.2f}ms (Ratio: {delay_ratio:.2f}x)'
        })
        
    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/generate_curve', methods=['POST'])
def generate_delay_curve():
    """生成延迟曲线的API接口"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        base_delay = data.get('base_delay')
        
        if base_delay is None:
            return jsonify({'error': 'Missing required parameter: base_delay'}), 400
        
        try:
            base_delay = float(base_delay)
        except ValueError:
            return jsonify({'error': 'Base delay must be a number'}), 400
        
        if base_delay <= 0:
            return jsonify({'error': 'Base delay must be positive'}), 400
        
        # 生成0到1之间的利用率点
        utilizations = [i * 0.01 for i in range(101)]  # 0.00, 0.01, 0.02, ..., 1.00
        delays = []
        
        for util in utilizations:
            delay = calculate_delay(base_delay, util)
            delays.append(round(delay, 2))
        
        return jsonify({
            'success': True,
            'base_delay': base_delay,
            'utilizations': utilizations,
            'delays': delays
        })
        
    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({'status': 'healthy', 'message': 'Delay Calculator API is running'})

if __name__ == '__main__':
    print("Starting Delay Calculator API server...")
    print("API endpoints:")
    print("- POST /api/calculate_delay - Calculate single delay value")
    print("- POST /api/generate_curve - Generate delay curve")
    print("- GET /api/health - Health check")
    print("\nServer will run on http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)