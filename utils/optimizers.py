"""
优化器实现
包含SGD和Adam优化器
"""

import numpy as np


class Optimizer:
    """优化器基类"""
    
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    def update(self, params, grads):
        """更新参数"""
        raise NotImplementedError


class SGD(Optimizer):
    """随机梯度下降优化器"""
    
    def __init__(self, learning_rate=0.01, momentum=0.0, weight_decay=0.0):
        """
        初始化SGD优化器
        
        Args:
            learning_rate: 学习率
            momentum: 动量参数
            weight_decay: 权重衰减参数
        """
        super().__init__(learning_rate)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = {}
    
    def update(self, param_name, param, grad):
        """
        更新参数
        
        Args:
            param_name: 参数名称
            param: 参数值
            grad: 梯度值
            
        Returns:
            numpy.ndarray: 更新后的参数
        """
        # 添加权重衰减
        if self.weight_decay > 0:
            grad = grad + self.weight_decay * param
        
        # 动量更新
        if self.momentum > 0:
            if param_name not in self.velocity:
                self.velocity[param_name] = np.zeros_like(param)
            
            self.velocity[param_name] = self.momentum * self.velocity[param_name] - self.learning_rate * grad
            return param + self.velocity[param_name]
        else:
            # 标准SGD更新
            return param - self.learning_rate * grad


class Adam(Optimizer):
    """Adam优化器"""
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0):
        """
        初始化Adam优化器
        
        Args:
            learning_rate: 学习率
            beta1: 一阶矩估计的衰减率
            beta2: 二阶矩估计的衰减率
            epsilon: 防止除零的小常数
            weight_decay: 权重衰减参数
        """
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        
        # 动量缓存
        self.m = {}  # 一阶矩估计
        self.v = {}  # 二阶矩估计
        self.t = 0   # 时间步
    
    def update(self, param_name, param, grad):
        """
        更新参数
        
        Args:
            param_name: 参数名称
            param: 参数值
            grad: 梯度值
            
        Returns:
            numpy.ndarray: 更新后的参数
        """
        # 添加权重衰减
        if self.weight_decay > 0:
            grad = grad + self.weight_decay * param
        
        # 初始化动量
        if param_name not in self.m:
            self.m[param_name] = np.zeros_like(param)
            self.v[param_name] = np.zeros_like(param)
        
        # 更新时间步
        self.t += 1
        
        # 更新一阶和二阶矩估计
        self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * grad
        self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * (grad ** 2)
        
        # 偏置修正
        m_hat = self.m[param_name] / (1 - self.beta1 ** self.t)
        v_hat = self.v[param_name] / (1 - self.beta2 ** self.t)
        
        # 参数更新
        return param - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)


class RMSprop(Optimizer):
    """RMSprop优化器"""
    
    def __init__(self, learning_rate=0.001, alpha=0.99, epsilon=1e-8, weight_decay=0.0):
        """
        初始化RMSprop优化器
        
        Args:
            learning_rate: 学习率
            alpha: 衰减率
            epsilon: 防止除零的小常数
            weight_decay: 权重衰减参数
        """
        super().__init__(learning_rate)
        self.alpha = alpha
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        
        # 缓存
        self.cache = {}
    
    def update(self, param_name, param, grad):
        """
        更新参数
        
        Args:
            param_name: 参数名称
            param: 参数值
            grad: 梯度值
            
        Returns:
            numpy.ndarray: 更新后的参数
        """
        # 添加权重衰减
        if self.weight_decay > 0:
            grad = grad + self.weight_decay * param
        
        # 初始化缓存
        if param_name not in self.cache:
            self.cache[param_name] = np.zeros_like(param)
        
        # 更新缓存
        self.cache[param_name] = self.alpha * self.cache[param_name] + (1 - self.alpha) * (grad ** 2)
        
        # 参数更新
        return param - self.learning_rate * grad / (np.sqrt(self.cache[param_name]) + self.epsilon)


class LearningRateScheduler:
    """学习率调度器"""
    
    def __init__(self, optimizer, schedule_type='step', **kwargs):
        """
        初始化学习率调度器
        
        Args:
            optimizer: 优化器
            schedule_type: 调度类型 ('step', 'exponential', 'cosine')
            **kwargs: 调度参数
        """
        self.optimizer = optimizer
        self.schedule_type = schedule_type
        self.initial_lr = optimizer.learning_rate
        self.current_step = 0
        
        # 调度参数
        if schedule_type == 'step':
            self.step_size = kwargs.get('step_size', 100)
            self.gamma = kwargs.get('gamma', 0.1)
        elif schedule_type == 'exponential':
            self.gamma = kwargs.get('gamma', 0.95)
        elif schedule_type == 'cosine':
            self.T_max = kwargs.get('T_max', 1000)
            self.eta_min = kwargs.get('eta_min', 0)
    
    def step(self):
        """更新学习率"""
        self.current_step += 1
        
        if self.schedule_type == 'step':
            if self.current_step % self.step_size == 0:
                self.optimizer.learning_rate *= self.gamma
        elif self.schedule_type == 'exponential':
            self.optimizer.learning_rate = self.initial_lr * (self.gamma ** self.current_step)
        elif self.schedule_type == 'cosine':
            self.optimizer.learning_rate = self.eta_min + (self.initial_lr - self.eta_min) * \
                                         (1 + np.cos(np.pi * self.current_step / self.T_max)) / 2
    
    def get_lr(self):
        """获取当前学习率"""
        return self.optimizer.learning_rate


if __name__ == "__main__":
    # 测试优化器
    print("优化器测试")
    
    # 创建模拟参数和梯度
    np.random.seed(42)
    param = np.random.randn(10, 5)
    grad = np.random.randn(10, 5)
    
    # 测试SGD
    print("\n测试SGD:")
    sgd = SGD(learning_rate=0.01, momentum=0.9)
    updated_param = sgd.update('test_param', param, grad)
    print(f"原始参数均值: {param.mean():.4f}")
    print(f"更新后参数均值: {updated_param.mean():.4f}")
    
    # 测试Adam
    print("\n测试Adam:")
    adam = Adam(learning_rate=0.001)
    updated_param = adam.update('test_param', param, grad)
    print(f"原始参数均值: {param.mean():.4f}")
    print(f"更新后参数均值: {updated_param.mean():.4f}")
    
    # 测试RMSprop
    print("\n测试RMSprop:")
    rmsprop = RMSprop(learning_rate=0.001)
    updated_param = rmsprop.update('test_param', param, grad)
    print(f"原始参数均值: {param.mean():.4f}")
    print(f"更新后参数均值: {updated_param.mean():.4f}")
    
    # 测试学习率调度器
    print("\n测试学习率调度器:")
    scheduler = LearningRateScheduler(sgd, schedule_type='step', step_size=5, gamma=0.1)
    print(f"初始学习率: {scheduler.get_lr():.4f}")
    
    for i in range(10):
        scheduler.step()
        print(f"Step {i+1}, 学习率: {scheduler.get_lr():.4f}")
    
    print("\n测试完成！") 