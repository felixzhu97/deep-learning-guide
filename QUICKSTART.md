# 🚀 快速开始指南

欢迎使用深度学习入门项目！本指南将帮助您快速上手。

## 📦 安装依赖

```bash
# 安装Python依赖包
pip install -r requirements.txt

# 或者使用pip3
pip3 install -r requirements.txt
```

## 🎯 快速体验

### 1. 运行演示脚本（推荐）

```bash
python3 demo.py
```

选择"1. 🚀 快速演示"，大约 5-10 分钟即可看到各种方法的效果对比。

### 2. 逐步体验

#### 基础方法：像素相似性

```bash
python3 examples/train_pixel_similarity.py
```

#### 进阶方法：神经网络

```bash
python3 examples/train_neural_network.py
```

### 3. 测试项目功能

```bash
python3 tests/test_basic.py
```

## 📊 预期结果

| 方法         | 准确率 | 训练时间   | 特点                       |
| ------------ | ------ | ---------- | -------------------------- |
| 像素相似性   | ~50%   | 几分钟     | 简单直观，理解图像识别基础 |
| 简单神经网络 | ~90%   | 10-20 分钟 | 掌握反向传播和梯度下降     |

## 🔧 自定义实验

### 修改网络架构

```python
# 在examples/train_neural_network.py中修改
nn = NeuralNetwork(
    input_size=784,
    hidden_sizes=[256, 128, 64],  # 改变隐藏层结构
    output_size=10,
    activation='relu'
)
```

### 调整超参数

```python
# 修改训练参数
nn.fit(
    X_train, y_train,
    epochs=50,           # 增加训练轮数
    batch_size=64,       # 调整批次大小
    learning_rate=0.001, # 调整学习率
    verbose=True
)
```

## 🎓 学习路径建议

1. **理解数据** - 运行 `python3 demo.py` 选择"3. 🔍 数据探索"
2. **基础方法** - 先运行像素相似性方法，理解最基本的图像识别原理
3. **深入神经网络** - 运行神经网络训练，观察训练过程
4. **实验对比** - 尝试不同的网络结构和参数设置
5. **可视化分析** - 查看生成的训练图表和混淆矩阵

## 📈 输出文件说明

运行后会生成以下可视化文件：

- `mnist_samples.png` - MNIST 数据集样本
- `neural_network_training_history.png` - 训练过程
- `neural_network_confusion_matrix.png` - 混淆矩阵
- `*_comparison.png` - 各种方法对比图

## ⚠️ 常见问题

**Q: 提示缺少依赖包？**
A: 运行 `pip3 install -r requirements.txt`

**Q: 训练很慢？**
A: 这是正常的，可以减少数据量或降低 epoch 数量

**Q: 想要更高的准确率？**
A: 可以尝试：

- 增加训练轮数
- 使用更多训练数据
- 调整网络结构
- 优化超参数

**Q: 如何理解代码？**
A: 建议按以下顺序阅读：

1. `data/` - 数据处理
2. `utils/functions.py` - 基础函数
3. `models/neural_network.py` - 神经网络核心
4. `examples/` - 完整示例

## 🌟 进阶学习

完成基础项目后，您可以尝试：

1. **添加新的激活函数**
2. **实现更多优化器**
3. **添加正则化技术**
4. **实现卷积神经网络**
5. **尝试其他数据集**

## 🤝 获取帮助

- 查看 `README.md` 了解项目详情
- 阅读代码中的详细注释
- 运行 `python3 demo.py` 选择"4. 📋 项目信息"

祝您深度学习之旅愉快！🎉
