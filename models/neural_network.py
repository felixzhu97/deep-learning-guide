"""
简单神经网络实现
包含多层感知机和反向传播算法
"""

import numpy as np
from tqdm import tqdm
from utils.functions import sigmoid, relu, softmax, cross_entropy_loss, init_weights


class NeuralNetwork:
    """
    简单的多层感知机神经网络
    
    实现了前向传播、反向传播和梯度下降训练
    """
    
    def __init__(self, input_size, hidden_sizes, output_size, activation='relu'):
        """
        初始化神经网络
        
        Args:
            input_size: 输入层大小
            hidden_sizes: 隐藏层大小列表，如[100, 50]表示两个隐藏层
            output_size: 输出层大小
            activation: 激活函数类型 ('relu', 'sigmoid')
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation = activation
        
        # 构建网络结构
        self.layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.num_layers = len(self.layer_sizes) - 1
        
        # 初始化权重和偏置
        self.weights = []
        self.biases = []
        
        for i in range(self.num_layers):
            # 权重初始化
            weight_shape = (self.layer_sizes[i], self.layer_sizes[i + 1])
            if activation == 'relu':
                w = init_weights(weight_shape, method='he')
            else:
                w = init_weights(weight_shape, method='xavier')
            self.weights.append(w)
            
            # 偏置初始化
            b = np.zeros((1, self.layer_sizes[i + 1]))
            self.biases.append(b)
        
        # 训练历史
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
    
    def _activate(self, x, layer_idx):
        """
        激活函数
        
        Args:
            x: 输入数据
            layer_idx: 层索引
            
        Returns:
            激活后的输出
        """
        if layer_idx == self.num_layers - 1:
            # 输出层使用softmax
            return softmax(x)
        else:
            # 隐藏层使用指定的激活函数
            if self.activation == 'relu':
                return relu(x)
            elif self.activation == 'sigmoid':
                return sigmoid(x)
            else:
                raise ValueError(f"不支持的激活函数: {self.activation}")
    
    def _activate_derivative(self, x):
        """
        激活函数的导数
        
        Args:
            x: 输入数据
            
        Returns:
            激活函数的导数
        """
        if self.activation == 'relu':
            return (x > 0).astype(float)
        elif self.activation == 'sigmoid':
            s = sigmoid(x)
            return s * (1 - s)
        else:
            raise ValueError(f"不支持的激活函数: {self.activation}")
    
    def forward(self, X):
        """
        前向传播
        
        Args:
            X: 输入数据 (batch_size, input_size)
            
        Returns:
            tuple: (输出, 各层激活值, 各层输入值)
        """
        activations = [X]  # 存储每层的激活值
        z_values = []      # 存储每层的加权输入值
        
        current_input = X
        
        for i in range(self.num_layers):
            # 计算加权输入
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            z_values.append(z)
            
            # 激活函数
            activation = self._activate(z, i)
            activations.append(activation)
            
            current_input = activation
        
        return activations[-1], activations, z_values
    
    def backward(self, X, y, activations, z_values):
        """
        反向传播
        
        Args:
            X: 输入数据
            y: 真实标签
            activations: 前向传播的激活值
            z_values: 前向传播的加权输入值
            
        Returns:
            tuple: (权重梯度, 偏置梯度)
        """
        m = X.shape[0]  # 批次大小
        
        # 初始化梯度
        weight_gradients = []
        bias_gradients = []
        
        # 输出层误差
        output_error = activations[-1] - y
        
        # 从输出层开始反向传播
        error = output_error
        
        for i in range(self.num_layers - 1, -1, -1):
            # 计算权重梯度
            if i == 0:
                weight_grad = np.dot(activations[i].T, error) / m
            else:
                weight_grad = np.dot(activations[i].T, error) / m
            weight_gradients.insert(0, weight_grad)
            
            # 计算偏置梯度
            bias_grad = np.sum(error, axis=0, keepdims=True) / m
            bias_gradients.insert(0, bias_grad)
            
            # 计算下一层的误差（除了输入层）
            if i > 0:
                error = np.dot(error, self.weights[i].T) * self._activate_derivative(z_values[i-1])
        
        return weight_gradients, bias_gradients
    
    def train_batch(self, X, y, learning_rate=0.01):
        """
        训练一个批次
        
        Args:
            X: 输入数据
            y: 真实标签
            learning_rate: 学习率
            
        Returns:
            float: 批次损失
        """
        # 前向传播
        output, activations, z_values = self.forward(X)
        
        # 计算损失
        loss = cross_entropy_loss(y, output)
        
        # 反向传播
        weight_grads, bias_grads = self.backward(X, y, activations, z_values)
        
        # 更新参数
        for i in range(self.num_layers):
            self.weights[i] -= learning_rate * weight_grads[i]
            self.biases[i] -= learning_rate * bias_grads[i]
        
        return loss
    
    def predict(self, X):
        """
        预测
        
        Args:
            X: 输入数据
            
        Returns:
            numpy.ndarray: 预测概率
        """
        output, _, _ = self.forward(X)
        return output
    
    def predict_classes(self, X):
        """
        预测类别
        
        Args:
            X: 输入数据
            
        Returns:
            numpy.ndarray: 预测类别
        """
        predictions = self.predict(X)
        return np.argmax(predictions, axis=1)
    
    def evaluate(self, X, y):
        """
        评估模型
        
        Args:
            X: 输入数据
            y: 真实标签
            
        Returns:
            dict: 评估结果
        """
        # 预测
        predictions = self.predict(X)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # 计算损失
        loss = cross_entropy_loss(y, predictions)
        
        # 计算准确率
        if y.ndim == 2:  # one-hot编码
            true_classes = np.argmax(y, axis=1)
        else:
            true_classes = y
        
        accuracy = np.mean(predicted_classes == true_classes)
        
        # 计算各类别准确率
        class_accuracies = {}
        for class_label in range(self.output_size):
            class_mask = (true_classes == class_label)
            if np.sum(class_mask) > 0:
                class_accuracy = np.mean(predicted_classes[class_mask] == true_classes[class_mask])
                class_accuracies[class_label] = class_accuracy
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'class_accuracies': class_accuracies,
            'predictions': predictions,
            'predicted_classes': predicted_classes
        }
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, 
            epochs=100, batch_size=32, learning_rate=0.01, 
            verbose=True, early_stopping=False, patience=10):
        """
        训练神经网络
        
        Args:
            X_train: 训练数据
            y_train: 训练标签
            X_val: 验证数据
            y_val: 验证标签
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            verbose: 是否显示训练过程
            early_stopping: 是否启用早停
            patience: 早停耐心值
        """
        # 清空训练历史
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        num_batches = len(X_train) // batch_size
        
        for epoch in range(epochs):
            # 随机打乱数据
            indices = np.random.permutation(len(X_train))
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            
            # 批次训练
            epoch_losses = []
            
            if verbose:
                pbar = tqdm(range(num_batches), desc=f'Epoch {epoch+1}/{epochs}')
            else:
                pbar = range(num_batches)
            
            for batch_idx in pbar:
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                
                X_batch = X_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]
                
                # 训练批次
                loss = self.train_batch(X_batch, y_batch, learning_rate)
                epoch_losses.append(loss)
                
                if verbose and isinstance(pbar, tqdm):
                    pbar.set_postfix({'loss': f'{loss:.4f}'})
            
            # 计算epoch平均损失
            avg_train_loss = np.mean(epoch_losses)
            self.train_losses.append(avg_train_loss)
            
            # 计算训练准确率
            train_results = self.evaluate(X_train, y_train)
            self.train_accuracies.append(train_results['accuracy'])
            
            # 验证集评估
            if X_val is not None and y_val is not None:
                val_results = self.evaluate(X_val, y_val)
                self.val_losses.append(val_results['loss'])
                self.val_accuracies.append(val_results['accuracy'])
                
                # 早停检查
                if early_stopping:
                    if val_results['loss'] < best_val_loss:
                        best_val_loss = val_results['loss']
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            if verbose:
                                print(f"早停在epoch {epoch+1}，最佳验证损失: {best_val_loss:.4f}")
                            break
            
            # 打印进度
            if verbose:
                if X_val is not None and y_val is not None:
                    print(f'Epoch {epoch+1}/{epochs}: '
                          f'Train Loss: {avg_train_loss:.4f}, '
                          f'Train Acc: {train_results["accuracy"]:.4f}, '
                          f'Val Loss: {val_results["loss"]:.4f}, '
                          f'Val Acc: {val_results["accuracy"]:.4f}')
                else:
                    print(f'Epoch {epoch+1}/{epochs}: '
                          f'Train Loss: {avg_train_loss:.4f}, '
                          f'Train Acc: {train_results["accuracy"]:.4f}')
    
    def get_architecture_summary(self):
        """获取网络架构摘要"""
        total_params = 0
        summary = []
        
        for i in range(self.num_layers):
            layer_params = self.weights[i].size + self.biases[i].size
            total_params += layer_params
            
            if i == 0:
                layer_type = "Input → Hidden"
            elif i == self.num_layers - 1:
                layer_type = "Hidden → Output"
            else:
                layer_type = "Hidden → Hidden"
            
            summary.append({
                'layer': i + 1,
                'type': layer_type,
                'input_size': self.layer_sizes[i],
                'output_size': self.layer_sizes[i + 1],
                'parameters': layer_params,
                'activation': 'softmax' if i == self.num_layers - 1 else self.activation
            })
        
        return {
            'summary': summary,
            'total_parameters': total_params
        }


if __name__ == "__main__":
    # 测试神经网络
    print("神经网络测试")
    
    # 创建模拟数据
    np.random.seed(42)
    X_train = np.random.randn(1000, 784)
    y_train = np.random.randint(0, 10, (1000, 10))  # one-hot编码
    
    X_test = np.random.randn(200, 784)
    y_test = np.random.randint(0, 10, (200, 10))
    
    # 创建神经网络
    nn = NeuralNetwork(
        input_size=784,
        hidden_sizes=[128, 64],
        output_size=10,
        activation='relu'
    )
    
    # 打印网络架构
    arch_summary = nn.get_architecture_summary()
    print("\n网络架构:")
    for layer_info in arch_summary['summary']:
        print(f"Layer {layer_info['layer']}: {layer_info['type']}")
        print(f"  输入: {layer_info['input_size']}, 输出: {layer_info['output_size']}")
        print(f"  激活函数: {layer_info['activation']}")
        print(f"  参数数量: {layer_info['parameters']}")
    print(f"总参数数量: {arch_summary['total_parameters']}")
    
    # 训练网络
    print("\n开始训练...")
    nn.fit(X_train, y_train, X_test, y_test, epochs=5, batch_size=32, verbose=True)
    
    # 评估网络
    print("\n评估结果:")
    results = nn.evaluate(X_test, y_test)
    print(f"测试损失: {results['loss']:.4f}")
    print(f"测试准确率: {results['accuracy']:.4f}")
    
    print("\n测试完成！") 