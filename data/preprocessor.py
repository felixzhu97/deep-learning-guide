"""
数据预处理器
提供MNIST数据的各种预处理功能
"""

import numpy as np
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    """数据预处理器类"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False
    
    def normalize_images(self, images):
        """
        归一化图像数据到[0,1]范围
        
        Args:
            images: 图像数据数组
            
        Returns:
            numpy.ndarray: 归一化后的图像数据
        """
        return images.astype(np.float32) / 255.0
    
    def flatten_images(self, images):
        """
        将图像数据展平为一维向量
        
        Args:
            images: 图像数据数组 (N, H, W)
            
        Returns:
            numpy.ndarray: 展平后的图像数据 (N, H*W)
        """
        return images.reshape(images.shape[0], -1)
    
    def one_hot_encode(self, labels, num_classes=10):
        """
        将标签转换为one-hot编码
        
        Args:
            labels: 标签数组
            num_classes: 类别数量
            
        Returns:
            numpy.ndarray: one-hot编码后的标签
        """
        one_hot = np.zeros((len(labels), num_classes))
        one_hot[np.arange(len(labels)), labels] = 1
        return one_hot
    
    def standardize_features(self, X_train, X_test=None):
        """
        标准化特征（零均值，单位方差）
        
        Args:
            X_train: 训练数据
            X_test: 测试数据（可选）
            
        Returns:
            tuple: 标准化后的训练和测试数据
        """
        if not self.fitted:
            X_train_scaled = self.scaler.fit_transform(X_train)
            self.fitted = True
        else:
            X_train_scaled = self.scaler.transform(X_train)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def add_noise(self, images, noise_level=0.1):
        """
        为图像添加噪声（数据增强）
        
        Args:
            images: 图像数据
            noise_level: 噪声强度
            
        Returns:
            numpy.ndarray: 添加噪声后的图像
        """
        noise = np.random.normal(0, noise_level, images.shape)
        noisy_images = images + noise
        return np.clip(noisy_images, 0, 1)
    
    def preprocess_for_pixel_similarity(self, train_images, train_labels, test_images, test_labels):
        """
        为像素相似性方法预处理数据
        
        Args:
            train_images: 训练图像
            train_labels: 训练标签
            test_images: 测试图像
            test_labels: 测试标签
            
        Returns:
            tuple: 预处理后的数据
        """
        # 归一化
        train_images = self.normalize_images(train_images)
        test_images = self.normalize_images(test_images)
        
        # 展平
        train_images = self.flatten_images(train_images)
        test_images = self.flatten_images(test_images)
        
        return train_images, train_labels, test_images, test_labels
    
    def preprocess_for_neural_network(self, train_images, train_labels, test_images, test_labels):
        """
        为神经网络预处理数据
        
        Args:
            train_images: 训练图像
            train_labels: 训练标签
            test_images: 测试图像
            test_labels: 测试标签
            
        Returns:
            tuple: 预处理后的数据
        """
        # 归一化
        train_images = self.normalize_images(train_images)
        test_images = self.normalize_images(test_images)
        
        # 展平
        train_images = self.flatten_images(train_images)
        test_images = self.flatten_images(test_images)
        
        # one-hot编码标签
        train_labels_onehot = self.one_hot_encode(train_labels)
        test_labels_onehot = self.one_hot_encode(test_labels)
        
        return train_images, train_labels_onehot, test_images, test_labels_onehot
    
    def preprocess_for_cnn(self, train_images, train_labels, test_images, test_labels):
        """
        为CNN预处理数据
        
        Args:
            train_images: 训练图像
            train_labels: 训练标签
            test_images: 测试图像
            test_labels: 测试标签
            
        Returns:
            tuple: 预处理后的数据
        """
        # 归一化
        train_images = self.normalize_images(train_images)
        test_images = self.normalize_images(test_images)
        
        # 添加通道维度 (N, H, W) -> (N, 1, H, W)
        train_images = train_images[:, np.newaxis, :, :]
        test_images = test_images[:, np.newaxis, :, :]
        
        # one-hot编码标签
        train_labels_onehot = self.one_hot_encode(train_labels)
        test_labels_onehot = self.one_hot_encode(test_labels)
        
        return train_images, train_labels_onehot, test_images, test_labels_onehot
    
    def create_batches(self, X, y, batch_size=32):
        """
        创建批次数据
        
        Args:
            X: 特征数据
            y: 标签数据
            batch_size: 批次大小
            
        Yields:
            tuple: (X_batch, y_batch)
        """
        num_samples = len(X)
        indices = np.random.permutation(num_samples)
        
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            yield X[batch_indices], y[batch_indices]


def preprocess_data(train_images, train_labels, test_images, test_labels, method='neural_network'):
    """
    便捷函数：根据指定方法预处理数据
    
    Args:
        train_images: 训练图像
        train_labels: 训练标签
        test_images: 测试图像
        test_labels: 测试标签
        method: 预处理方法 ('pixel_similarity', 'neural_network', 'cnn')
        
    Returns:
        tuple: 预处理后的数据
    """
    preprocessor = DataPreprocessor()
    
    if method == 'pixel_similarity':
        return preprocessor.preprocess_for_pixel_similarity(
            train_images, train_labels, test_images, test_labels
        )
    elif method == 'neural_network':
        return preprocessor.preprocess_for_neural_network(
            train_images, train_labels, test_images, test_labels
        )
    elif method == 'cnn':
        return preprocessor.preprocess_for_cnn(
            train_images, train_labels, test_images, test_labels
        )
    else:
        raise ValueError(f"未知的预处理方法: {method}")


if __name__ == "__main__":
    # 测试预处理功能
    from mnist_loader import load_mnist
    
    # 加载数据
    train_images, train_labels, test_images, test_labels = load_mnist()
    
    # 测试不同的预处理方法
    print("测试像素相似性预处理:")
    X_train, y_train, X_test, y_test = preprocess_data(
        train_images, train_labels, test_images, test_labels, 
        method='pixel_similarity'
    )
    print(f"训练集形状: {X_train.shape}, 标签形状: {y_train.shape}")
    print(f"测试集形状: {X_test.shape}, 标签形状: {y_test.shape}")
    
    print("\n测试神经网络预处理:")
    X_train, y_train, X_test, y_test = preprocess_data(
        train_images, train_labels, test_images, test_labels, 
        method='neural_network'
    )
    print(f"训练集形状: {X_train.shape}, 标签形状: {y_train.shape}")
    print(f"测试集形状: {X_test.shape}, 标签形状: {y_test.shape}")
    
    print("\n测试CNN预处理:")
    X_train, y_train, X_test, y_test = preprocess_data(
        train_images, train_labels, test_images, test_labels, 
        method='cnn'
    )
    print(f"训练集形状: {X_train.shape}, 标签形状: {y_train.shape}")
    print(f"测试集形状: {X_test.shape}, 标签形状: {y_test.shape}") 