class DelayCalculator {
    constructor() {
        // 云函数API端点
        this.apiBaseUrl = '/api';
        this.chart = null;
        this.init();
    }

    init() {
        this.bindEvents();
        this.checkApiHealth();
    }

    bindEvents() {
        document.getElementById('calculateBtn').addEventListener('click', () => {
            this.calculateDelay();
        });

        document.getElementById('generateCurveBtn').addEventListener('click', () => {
            this.generateDelayCurve();
        });

        document.getElementById('clearBtn').addEventListener('click', () => {
            this.clearResults();
        });

        // 实时计算（输入时自动计算）
        document.getElementById('baseDelay').addEventListener('input', () => {
            this.debounce(() => this.calculateDelay(), 500);
        });

        document.getElementById('utilization').addEventListener('input', () => {
            this.debounce(() => this.calculateDelay(), 500);
        });
    }

    debounce(func, wait) {
        clearTimeout(this.debounceTimer);
        this.debounceTimer = setTimeout(func, wait);
    }

    async checkApiHealth() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/health`);
            if (response.ok) {
                this.showStatus('云函数API已连接并准备就绪', 'success');
            } else {
                this.showStatus('云函数API响应异常', 'warning');
            }
        } catch (error) {
            this.showStatus('无法连接到云函数API。请确保网络连接正常。', 'error');
        }
    }

    // 延迟计算函数（基于M/M/1队列模型）
    calculateDelayValue(baseDelay, utilization) {
        if (utilization >= 0.99) {
            return baseDelay * 100; // 拥塞时高延迟惩罚
        } else if (utilization <= 0.01) {
            return baseDelay; // 低利用率时的基础延迟
        } else {
            // 标准排队模型：总延迟 = 传输延迟 + 排队延迟
            const transmissionDelay = baseDelay;
            const queueDelay = (utilization / (1 - utilization)) * (baseDelay / 2);
            return transmissionDelay + queueDelay;
        }
    }

    async calculateDelay() {
        const baseDelay = parseFloat(document.getElementById('baseDelay').value);
        const utilization = parseFloat(document.getElementById('utilization').value);

        // 验证输入
        if (!this.validateInputs(baseDelay, utilization)) {
            return;
        }

        this.showLoading('calculateBtn', '计算中...');

        try {
            const response = await fetch(`${this.apiBaseUrl}/calculate_delay`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    base_delay: baseDelay,
                    utilization: utilization
                })
            });

            const data = await response.json();

            if (response.ok && data.success) {
                this.displayResults(data);
                this.showStatus('延迟计算成功！', 'success');
            } else {
                this.showError(data.error || '计算失败');
            }
        } catch (error) {
            this.showError('网络错误: ' + error.message);
        } finally {
            this.hideLoading('calculateBtn', '计算延迟');
        }
    }

    async generateDelayCurve() {
        const baseDelay = parseFloat(document.getElementById('baseDelay').value);

        if (!baseDelay || baseDelay <= 0) {
            this.showError('请输入有效的基础延迟');
            return;
        }

        this.showLoading('generateCurveBtn', '生成中...');

        try {
            const response = await fetch(`${this.apiBaseUrl}/generate_curve`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    base_delay: baseDelay
                })
            });

            const data = await response.json();

            if (response.ok && data.success) {
                this.displayChart(data);
                this.showStatus('延迟曲线生成成功！', 'success');
            } else {
                this.showError(data.error || '生成曲线失败');
            }
        } catch (error) {
            this.showError('网络错误: ' + error.message);
        } finally {
            this.hideLoading('generateCurveBtn', '生成延迟曲线');
        }
    }

    validateInputs(baseDelay, utilization) {
        if (!baseDelay || baseDelay <= 0) {
            this.showError('基础延迟必须是正数');
            return false;
        }

        if (utilization === null || utilization === undefined || utilization < 0 || utilization > 1) {
            this.showError('链路利用率必须在0到1之间');
            return false;
        }

        return true;
    }

    displayResults(data) {
        // 显示结果区域
        document.getElementById('resultsSection').style.display = 'block';

        // 更新结果值
        document.getElementById('resultBaseDelay').textContent = data.base_delay + ' ms';
        document.getElementById('resultUtilization').textContent = (data.utilization * 100).toFixed(1) + '%';
        document.getElementById('resultDelay').textContent = data.calculated_delay + ' ms';
        document.getElementById('resultRatio').textContent = data.delay_ratio + 'x';
    }

    displayChart(data) {
        // 显示图表区域
        document.getElementById('chartSection').style.display = 'block';

        const ctx = document.getElementById('delayChart').getContext('2d');

        // 销毁现有图表
        if (this.chart) {
            this.chart.destroy();
        }

        // 创建新图表
        this.chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.utilizations.map(u => (u * 100).toFixed(0) + '%'),
                datasets: [{
                    label: `Delay (Base: ${data.base_delay}ms)`,
                    data: data.delays,
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 3,
                    pointHoverRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Delay vs Utilization Curve',
                        font: {
                            size: 16
                        }
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        callbacks: {
                            label: function(context) {
                                return `Utilization: ${(context.parsed.x).toFixed(0)}%, Delay: ${context.parsed.y}ms`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Link Utilization (%)'
                        },
                        grid: {
                            color: 'rgba(0, 0, 0, 0.1)'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Delay (ms)'
                        },
                        grid: {
                            color: 'rgba(0, 0, 0, 0.1)'
                        },
                        beginAtZero: true
                    }
                },
                interaction: {
                    mode: 'nearest',
                    axis: 'x',
                    intersect: false
                }
            }
        });
    }

    clearResults() {
        // 隐藏结果和图表区域
        document.getElementById('resultsSection').style.display = 'none';
        document.getElementById('chartSection').style.display = 'none';

        // 清除图表
        if (this.chart) {
            this.chart.destroy();
            this.chart = null;
        }

        // 清除状态消息
        this.clearMessages();

        this.showStatus('结果已清除', 'info');
    }

    showLoading(buttonId, loadingText) {
        const button = document.getElementById(buttonId);
        button.disabled = true;
        button.innerHTML = `<span class="loading"></span>${loadingText}`;
    }

    hideLoading(buttonId, originalText) {
        const button = document.getElementById(buttonId);
        button.disabled = false;
        button.textContent = originalText;
    }

    showStatus(message, type = 'info') {
        this.clearMessages();
        
        const statusElement = document.getElementById('statusMessage');
        statusElement.textContent = message;
        statusElement.style.display = 'block';
        
        // 根据类型设置样式
        statusElement.className = 'status-message';
        if (type === 'error') {
            statusElement.style.background = '#fed7d7';
            statusElement.style.borderColor = '#feb2b2';
            statusElement.style.color = '#742a2a';
        } else if (type === 'warning') {
            statusElement.style.background = '#feebc8';
            statusElement.style.borderColor = '#fbd38d';
            statusElement.style.color = '#744210';
        } else if (type === 'success') {
            statusElement.style.background = '#f0fff4';
            statusElement.style.borderColor = '#9ae6b4';
            statusElement.style.color = '#22543d';
        }
    }

    showError(message) {
        this.clearMessages();
        
        const errorElement = document.getElementById('errorMessage');
        errorElement.textContent = message;
        errorElement.style.display = 'block';
    }

    clearMessages() {
        document.getElementById('statusMessage').style.display = 'none';
        document.getElementById('errorMessage').style.display = 'none';
    }
}

// 页面加载完成后初始化应用
document.addEventListener('DOMContentLoaded', () => {
    new DelayCalculator();
});

// 添加键盘快捷键支持
document.addEventListener('keydown', (e) => {
    if (e.ctrlKey || e.metaKey) {
        switch(e.key) {
            case 'Enter':
                e.preventDefault();
                document.getElementById('calculateBtn').click();
                break;
            case 'c':
                e.preventDefault();
                document.getElementById('clearBtn').click();
                break;
            case 'g':
                e.preventDefault();
                document.getElementById('generateCurveBtn').click();
                break;
        }
    }
});