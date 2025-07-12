"""
MNIST数据加载器
提供下载和加载MNIST数据集的功能
"""

import os
import gzip
import numpy as np
import requests
from tqdm import tqdm


class MNISTLoader:
    """MNIST数据加载器类"""
    
    def __init__(self, data_dir='./mnist_data'):
        """
        初始化MNIST数据加载器
        
        Args:
            data_dir: 数据存储目录
        """
        self.data_dir = data_dir
        self.urls = {
            'train_images': 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
            'train_labels': 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
            'test_images': 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
            'test_labels': 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
        }
        
        # 创建数据目录
        os.makedirs(self.data_dir, exist_ok=True)
    
    def download_file(self, url, filename):
        """
        下载文件
        
        Args:
            url: 文件URL
            filename: 保存的文件名
        """
        filepath = os.path.join(self.data_dir, filename)
        
        # 如果文件已存在，跳过下载
        if os.path.exists(filepath):
            print(f"文件 {filename} 已存在，跳过下载")
            return filepath
        
        print(f"正在下载 {filename}...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        print(f"下载完成: {filename}")
        return filepath
    
    def load_images(self, filename):
        """
        加载图像数据
        
        Args:
            filename: 图像文件名
            
        Returns:
            numpy.ndarray: 图像数据数组
        """
        filepath = os.path.join(self.data_dir, filename)
        
        with gzip.open(filepath, 'rb') as f:
            # 读取魔数和维度信息
            magic = int.from_bytes(f.read(4), 'big')
            num_images = int.from_bytes(f.read(4), 'big')
            num_rows = int.from_bytes(f.read(4), 'big')
            num_cols = int.from_bytes(f.read(4), 'big')
            
            # 读取图像数据
            images = np.frombuffer(f.read(), dtype=np.uint8)
            images = images.reshape(num_images, num_rows, num_cols)
            
        return images
    
    def load_labels(self, filename):
        """
        加载标签数据
        
        Args:
            filename: 标签文件名
            
        Returns:
            numpy.ndarray: 标签数据数组
        """
        filepath = os.path.join(self.data_dir, filename)
        
        with gzip.open(filepath, 'rb') as f:
            # 读取魔数和维度信息
            magic = int.from_bytes(f.read(4), 'big')
            num_labels = int.from_bytes(f.read(4), 'big')
            
            # 读取标签数据
            labels = np.frombuffer(f.read(), dtype=np.uint8)
            
        return labels
    
    def download_all(self):
        """下载所有MNIST数据文件"""
        print("开始下载MNIST数据集...")
        
        for name, url in self.urls.items():
            filename = url.split('/')[-1]
            self.download_file(url, filename)
        
        print("MNIST数据集下载完成！")
    
    def load_data(self):
        """
        加载所有MNIST数据
        
        Returns:
            tuple: (train_images, train_labels, test_images, test_labels)
        """
        print("正在加载MNIST数据...")
        
        # 确保数据文件存在
        self.download_all()
        
        # 加载训练数据
        train_images = self.load_images('train-images-idx3-ubyte.gz')
        train_labels = self.load_labels('train-labels-idx1-ubyte.gz')
        
        # 加载测试数据
        test_images = self.load_images('t10k-images-idx3-ubyte.gz')
        test_labels = self.load_labels('t10k-labels-idx1-ubyte.gz')
        
        print(f"训练集: {train_images.shape[0]} 个样本")
        print(f"测试集: {test_images.shape[0]} 个样本")
        print(f"图像尺寸: {train_images.shape[1]}x{train_images.shape[2]}")
        
        return train_images, train_labels, test_images, test_labels


def load_mnist(data_dir='./mnist_data'):
    """
    便捷函数：加载MNIST数据集
    
    Args:
        data_dir: 数据存储目录
        
    Returns:
        tuple: (train_images, train_labels, test_images, test_labels)
    """
    loader = MNISTLoader(data_dir)
    return loader.load_data()


if __name__ == "__main__":
    # 测试数据加载
    train_images, train_labels, test_images, test_labels = load_mnist()
    
    print(f"训练集图像形状: {train_images.shape}")
    print(f"训练集标签形状: {train_labels.shape}")
    print(f"测试集图像形状: {test_images.shape}")
    print(f"测试集标签形状: {test_labels.shape}")
    
    # 显示一些基本统计信息
    print(f"像素值范围: [{train_images.min()}, {train_images.max()}]")
    print(f"标签范围: [{train_labels.min()}, {train_labels.max()}]") 