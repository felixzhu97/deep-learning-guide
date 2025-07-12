# 深度学习入门指南

基于斋藤康毅《深度学习入门》（ゼロから作る Deep Learning）一书，从零开始构建深度学习项目。

## 项目概述

本项目通过 MNIST 手写数字识别任务，演示深度学习的核心概念和实现方法：

1. **像素相似性方法** - 基础图像识别方法
2. **多层感知机** - 简单神经网络实现
3. **卷积神经网络** - CNN 的完整实现
4. **训练优化** - 反向传播、梯度下降等

## 项目结构

```
deep-learning-guide/
├── data/                   # 数据处理模块
│   ├── __init__.py
│   ├── mnist_loader.py     # MNIST数据加载器
│   └── preprocessor.py     # 数据预处理
├── models/                 # 模型实现
│   ├── __init__.py
│   ├── pixel_similarity.py # 像素相似性方法
│   ├── neural_network.py   # 简单神经网络
│   └── cnn.py             # 卷积神经网络
├── utils/                  # 工具函数
│   ├── __init__.py
│   ├── functions.py       # 激活函数、损失函数
│   ├── optimizers.py      # 优化器
│   └── visualization.py   # 可视化工具
├── examples/              # 示例脚本
│   ├── train_pixel_similarity.py
│   ├── train_neural_network.py
│   └── train_cnn.py
├── tests/                 # 测试文件
└── requirements.txt       # 依赖包
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 像素相似性方法

```bash
python examples/train_pixel_similarity.py
```

### 2. 简单神经网络

```bash
python examples/train_neural_network.py
```

### 3. 卷积神经网络

```bash
python examples/train_cnn.py
```

## 学习路径

1. **数据理解** - 了解 MNIST 数据集的结构和特征
2. **基础方法** - 从像素相似性开始，理解图像识别的基本原理
3. **神经网络** - 学习多层感知机的结构和训练过程
4. **深度学习** - 掌握 CNN 的卷积层、池化层等核心概念
5. **优化技巧** - 学习各种训练优化技术

## 特色功能

- 🔄 **渐进式学习** - 从简单到复杂，循序渐进
- 📊 **可视化展示** - 训练过程和结果的图表展示
- 🔧 **模块化设计** - 代码结构清晰，便于理解和修改
- 📚 **中文注释** - 详细的中文代码注释和说明
- 🎯 **实践导向** - 重点关注实际编程实现

## 性能对比

| 方法         | 准确率 | 训练时间   | 复杂度 |
| ------------ | ------ | ---------- | ------ |
| 像素相似性   | ~50%   | 几分钟     | 低     |
| 简单神经网络 | ~90%   | 10-20 分钟 | 中     |
| 卷积神经网络 | ~99%   | 30-60 分钟 | 高     |

## 参考资料

- 《深度学习入门》- 斋藤康毅著
- MNIST 数据集官方文档
- 深度学习相关论文和教程
