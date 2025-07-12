"""
激活函数和损失函数实现
包含深度学习中常用的函数
"""

import numpy as np


def sigmoid(x):
    """
    Sigmoid激活函数
    
    Args:
        x: 输入数据
        
    Returns:
        numpy.ndarray: 激活后的输出
    """
    # 防止数值溢出
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """
    Sigmoid函数的导数
    
    Args:
        x: 输入数据
        
    Returns:
        numpy.ndarray: 导数值
    """
    s = sigmoid(x)
    return s * (1 - s)


def relu(x):
    """
    ReLU激活函数
    
    Args:
        x: 输入数据
        
    Returns:
        numpy.ndarray: 激活后的输出
    """
    return np.maximum(0, x)


def relu_derivative(x):
    """
    ReLU函数的导数
    
    Args:
        x: 输入数据
        
    Returns:
        numpy.ndarray: 导数值
    """
    return (x > 0).astype(float)


def softmax(x):
    """
    Softmax激活函数
    
    Args:
        x: 输入数据
        
    Returns:
        numpy.ndarray: 激活后的输出
    """
    # 防止数值溢出
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def tanh(x):
    """
    Tanh激活函数
    
    Args:
        x: 输入数据
        
    Returns:
        numpy.ndarray: 激活后的输出
    """
    return np.tanh(x)


def tanh_derivative(x):
    """
    Tanh函数的导数
    
    Args:
        x: 输入数据
        
    Returns:
        numpy.ndarray: 导数值
    """
    return 1 - np.tanh(x) ** 2


def cross_entropy_loss(y_true, y_pred):
    """
    交叉熵损失函数
    
    Args:
        y_true: 真实标签（one-hot编码）
        y_pred: 预测概率
        
    Returns:
        float: 损失值
    """
    # 防止log(0)
    y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=-1))


def mean_squared_error(y_true, y_pred):
    """
    均方误差损失函数
    
    Args:
        y_true: 真实标签
        y_pred: 预测值
        
    Returns:
        float: 损失值
    """
    return np.mean((y_true - y_pred) ** 2)


def accuracy_score(y_true, y_pred):
    """
    计算准确率
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        
    Returns:
        float: 准确率
    """
    return np.mean(y_true == y_pred)


def one_hot_to_labels(one_hot):
    """
    将one-hot编码转换为标签
    
    Args:
        one_hot: one-hot编码的标签
        
    Returns:
        numpy.ndarray: 标签数组
    """
    return np.argmax(one_hot, axis=-1)


def labels_to_one_hot(labels, num_classes):
    """
    将标签转换为one-hot编码
    
    Args:
        labels: 标签数组
        num_classes: 类别数量
        
    Returns:
        numpy.ndarray: one-hot编码
    """
    one_hot = np.zeros((len(labels), num_classes))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot


def init_weights(shape, method='xavier'):
    """
    权重初始化
    
    Args:
        shape: 权重形状
        method: 初始化方法 ('xavier', 'he', 'random')
        
    Returns:
        numpy.ndarray: 初始化的权重
    """
    if method == 'xavier':
        # Xavier初始化
        fan_in, fan_out = shape[0], shape[1]
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, shape)
    elif method == 'he':
        # He初始化
        fan_in = shape[0]
        return np.random.normal(0, np.sqrt(2.0 / fan_in), shape)
    elif method == 'random':
        # 随机初始化
        return np.random.normal(0, 0.1, shape)
    else:
        raise ValueError(f"未知的初始化方法: {method}")


def euclidean_distance(x1, x2):
    """
    计算欧几里得距离
    
    Args:
        x1: 第一个向量
        x2: 第二个向量
        
    Returns:
        float: 欧几里得距离
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))


def cosine_similarity(x1, x2):
    """
    计算余弦相似度
    
    Args:
        x1: 第一个向量
        x2: 第二个向量
        
    Returns:
        float: 余弦相似度
    """
    dot_product = np.dot(x1, x2)
    norm_x1 = np.linalg.norm(x1)
    norm_x2 = np.linalg.norm(x2)
    
    if norm_x1 == 0 or norm_x2 == 0:
        return 0
    
    return dot_product / (norm_x1 * norm_x2)


if __name__ == "__main__":
    # 测试激活函数
    x = np.array([-2, -1, 0, 1, 2])
    print("输入:", x)
    print("Sigmoid:", sigmoid(x))
    print("ReLU:", relu(x))
    print("Tanh:", tanh(x))
    
    # 测试softmax
    logits = np.array([[1, 2, 3], [4, 5, 6]])
    print("\nLogits:", logits)
    print("Softmax:", softmax(logits))
    
    # 测试损失函数
    y_true = np.array([[0, 1, 0], [1, 0, 0]])
    y_pred = np.array([[0.1, 0.8, 0.1], [0.9, 0.05, 0.05]])
    print("\n真实标签:", y_true)
    print("预测概率:", y_pred)
    print("交叉熵损失:", cross_entropy_loss(y_true, y_pred))
    
    # 测试准确率
    y_true_labels = np.array([1, 0, 2, 1])
    y_pred_labels = np.array([1, 0, 1, 1])
    print("\n准确率:", accuracy_score(y_true_labels, y_pred_labels)) 