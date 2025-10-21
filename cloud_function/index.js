// 云函数入口文件
const calculateDelay = (baseDelay, utilization) => {
    if (utilization >= 0.99) {
        return baseDelay * 100; // 拥塞时高延迟惩罚
    } else if (utilization <= 0.01) {
        return baseDelay; // 低利用率时的基础延迟
    } else {
        // 标准排队模型：总延迟 = 传输延迟 + 排队延迟
        const transmissionDelay = baseDelay;
        const queueDelay = (utilization / (1 - utilization)) * (baseDelay / 2); // 标准化排队延迟
        return transmissionDelay + queueDelay;
    }
};

// 云函数主入口
exports.main_handler = async (event, context) => {
    console.log('Event:', JSON.stringify(event, null, 2));
    
    try {
        const { path, httpMethod, queryString, body } = event;
        
        // 设置CORS头
        const headers = {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization'
        };
        
        // 处理预检请求
        if (httpMethod === 'OPTIONS') {
            return {
                statusCode: 200,
                headers,
                body: JSON.stringify({ message: 'CORS preflight successful' })
            };
        }
        
        // 解析请求体
        let requestBody;
        if (body) {
            try {
                requestBody = typeof body === 'string' ? JSON.parse(body) : body;
            } catch (error) {
                return {
                    statusCode: 400,
                    headers,
                    body: JSON.stringify({ 
                        success: false, 
                        error: 'Invalid JSON format in request body' 
                    })
                };
            }
        }
        
        // 路由处理
        if (path === '/api/calculate_delay' && httpMethod === 'POST') {
            return handleCalculateDelay(requestBody, headers);
        } else if (path === '/api/generate_curve' && httpMethod === 'POST') {
            return handleGenerateCurve(requestBody, headers);
        } else if (path === '/api/health' && httpMethod === 'GET') {
            return handleHealthCheck(headers);
        } else {
            return {
                statusCode: 404,
                headers,
                body: JSON.stringify({ 
                    success: false, 
                    error: 'Endpoint not found' 
                })
            };
        }
        
    } catch (error) {
        console.error('Error:', error);
        return {
            statusCode: 500,
            headers: {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            body: JSON.stringify({ 
                success: false, 
                error: 'Internal server error: ' + error.message 
            })
        };
    }
};

// 处理延迟计算请求
function handleCalculateDelay(requestBody, headers) {
    if (!requestBody) {
        return {
            statusCode: 400,
            headers,
            body: JSON.stringify({ 
                success: false, 
                error: 'No data provided' 
            })
        };
    }
    
    const baseDelay = requestBody.base_delay;
    const utilization = requestBody.utilization;
    
    if (baseDelay === undefined || utilization === undefined) {
        return {
            statusCode: 400,
            headers,
            body: JSON.stringify({ 
                success: false, 
                error: 'Missing required parameters: base_delay and utilization' 
            })
        };
    }
    
    // 验证参数类型和范围
    const baseDelayNum = parseFloat(baseDelay);
    const utilizationNum = parseFloat(utilization);
    
    if (isNaN(baseDelayNum) || isNaN(utilizationNum)) {
        return {
            statusCode: 400,
            headers,
            body: JSON.stringify({ 
                success: false, 
                error: 'Parameters must be numbers' 
            })
        };
    }
    
    if (baseDelayNum <= 0) {
        return {
            statusCode: 400,
            headers,
            body: JSON.stringify({ 
                success: false, 
                error: 'Base delay must be positive' 
            })
        };
    }
    
    if (utilizationNum < 0 || utilizationNum > 1) {
        return {
            statusCode: 400,
            headers,
            body: JSON.stringify({ 
                success: false, 
                error: 'Utilization must be between 0 and 1' 
            })
        };
    }
    
    // 计算延迟
    const calculatedDelay = calculateDelay(baseDelayNum, utilizationNum);
    const delayRatio = calculatedDelay / baseDelayNum;
    
    return {
        statusCode: 200,
        headers,
        body: JSON.stringify({
            success: true,
            base_delay: baseDelayNum,
            utilization: utilizationNum,
            calculated_delay: Math.round(calculatedDelay * 100) / 100,
            delay_ratio: Math.round(delayRatio * 100) / 100,
            message: `Delay calculated successfully: ${calculatedDelay.toFixed(2)}ms (Ratio: ${delayRatio.toFixed(2)}x)`
        })
    };
}

// 处理延迟曲线生成请求
function handleGenerateCurve(requestBody, headers) {
    if (!requestBody) {
        return {
            statusCode: 400,
            headers,
            body: JSON.stringify({ 
                success: false, 
                error: 'No data provided' 
            })
        };
    }
    
    const baseDelay = requestBody.base_delay;
    
    if (baseDelay === undefined) {
        return {
            statusCode: 400,
            headers,
            body: JSON.stringify({ 
                success: false, 
                error: 'Missing required parameter: base_delay' 
            })
        };
    }
    
    const baseDelayNum = parseFloat(baseDelay);
    
    if (isNaN(baseDelayNum)) {
        return {
            statusCode: 400,
            headers,
            body: JSON.stringify({ 
                success: false, 
                error: 'Base delay must be a number' 
            })
        };
    }
    
    if (baseDelayNum <= 0) {
        return {
            statusCode: 400,
            headers,
            body: JSON.stringify({ 
                success: false, 
                error: 'Base delay must be positive' 
            })
        };
    }
    
    // 生成0到1之间的利用率点
    const utilizations = [];
    const delays = [];
    
    for (let i = 0; i <= 100; i++) {
        const util = i * 0.01;
        const delay = calculateDelay(baseDelayNum, util);
        utilizations.push(util);
        delays.push(Math.round(delay * 100) / 100);
    }
    
    return {
        statusCode: 200,
        headers,
        body: JSON.stringify({
            success: true,
            base_delay: baseDelayNum,
            utilizations: utilizations,
            delays: delays
        })
    };
}

// 处理健康检查请求
function handleHealthCheck(headers) {
    return {
        statusCode: 200,
        headers,
        body: JSON.stringify({ 
            status: 'healthy', 
            message: 'Delay Calculator Cloud Function is running' 
        })
    };
}