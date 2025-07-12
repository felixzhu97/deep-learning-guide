"""
像素相似性分类器
基于像素相似度的简单图像分类方法
"""

import numpy as np
from tqdm import tqdm
from utils.functions import euclidean_distance, cosine_similarity


class PixelSimilarityClassifier:
    """
    像素相似性分类器
    
    通过计算测试图像与训练图像的像素相似度来进行分类
    这是最基础的图像分类方法，用于理解图像识别的基本原理
    """
    
    def __init__(self, similarity_metric='euclidean', k=5):
        """
        初始化像素相似性分类器
        
        Args:
            similarity_metric: 相似度度量方法 ('euclidean', 'cosine')
            k: 使用k个最相似的样本进行投票
        """
        self.similarity_metric = similarity_metric
        self.k = k
        self.X_train = None
        self.y_train = None
        self.is_trained = False
    
    def fit(self, X_train, y_train):
        """
        训练分类器（实际上是存储训练数据）
        
        Args:
            X_train: 训练图像数据 (N, 784)
            y_train: 训练标签 (N,)
        """
        self.X_train = X_train
        self.y_train = y_train
        self.is_trained = True
        
        print(f"训练完成！存储了 {len(X_train)} 个训练样本")
    
    def _calculate_similarity(self, x1, x2):
        """
        计算两个图像的相似度
        
        Args:
            x1: 第一个图像向量
            x2: 第二个图像向量
            
        Returns:
            float: 相似度值
        """
        if self.similarity_metric == 'euclidean':
            # 欧几里得距离（距离越小越相似）
            return -euclidean_distance(x1, x2)
        elif self.similarity_metric == 'cosine':
            # 余弦相似度（相似度越大越相似）
            return cosine_similarity(x1, x2)
        else:
            raise ValueError(f"不支持的相似度度量方法: {self.similarity_metric}")
    
    def predict_single(self, x):
        """
        预测单个样本
        
        Args:
            x: 单个图像向量
            
        Returns:
            int: 预测标签
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 计算与所有训练样本的相似度
        similarities = []
        for i in range(len(self.X_train)):
            sim = self._calculate_similarity(x, self.X_train[i])
            similarities.append((sim, self.y_train[i]))
        
        # 按相似度排序，获取前k个最相似的样本
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_k_labels = [similarities[i][1] for i in range(min(self.k, len(similarities)))]
        
        # 投票决定最终预测结果
        prediction = np.bincount(top_k_labels).argmax()
        return prediction
    
    def predict(self, X_test):
        """
        预测测试数据
        
        Args:
            X_test: 测试图像数据 (N, 784)
            
        Returns:
            numpy.ndarray: 预测标签数组
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        predictions = []
        print(f"正在预测 {len(X_test)} 个测试样本...")
        
        for i in tqdm(range(len(X_test)), desc="预测进度"):
            pred = self.predict_single(X_test[i])
            predictions.append(pred)
        
        return np.array(predictions)
    
    def evaluate(self, X_test, y_test):
        """
        评估模型性能
        
        Args:
            X_test: 测试图像数据
            y_test: 测试标签
            
        Returns:
            dict: 包含准确率等评估指标的字典
        """
        predictions = self.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        
        # 计算每个类别的准确率
        class_accuracies = {}
        for class_label in range(10):  # MNIST有10个类别
            class_mask = (y_test == class_label)
            if np.sum(class_mask) > 0:
                class_accuracy = np.mean(predictions[class_mask] == y_test[class_mask])
                class_accuracies[class_label] = class_accuracy
        
        return {
            'accuracy': accuracy,
            'class_accuracies': class_accuracies,
            'predictions': predictions
        }
    
    def get_most_similar_images(self, test_image, n=5):
        """
        获取与测试图像最相似的n个训练图像
        
        Args:
            test_image: 测试图像向量
            n: 返回的相似图像数量
            
        Returns:
            list: [(相似度, 图像索引, 标签), ...]
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        similarities = []
        for i in range(len(self.X_train)):
            sim = self._calculate_similarity(test_image, self.X_train[i])
            similarities.append((sim, i, self.y_train[i]))
        
        # 按相似度排序，返回前n个
        similarities.sort(key=lambda x: x[0], reverse=True)
        return similarities[:n]


class PixelSimilarityBenchmark:
    """像素相似性方法的性能基准测试"""
    
    def __init__(self):
        self.results = {}
    
    def run_benchmark(self, X_train, y_train, X_test, y_test, test_size=1000):
        """
        运行基准测试
        
        Args:
            X_train: 训练数据
            y_train: 训练标签
            X_test: 测试数据
            y_test: 测试标签
            test_size: 测试样本数量（为了加速测试）
        """
        # 限制测试样本数量以加速测试
        if test_size < len(X_test):
            indices = np.random.choice(len(X_test), test_size, replace=False)
            X_test = X_test[indices]
            y_test = y_test[indices]
        
        # 测试不同的相似度度量方法
        metrics = ['euclidean', 'cosine']
        k_values = [1, 3, 5, 10]
        
        for metric in metrics:
            for k in k_values:
                print(f"\n测试配置: {metric} 距离, k={k}")
                
                # 创建分类器
                classifier = PixelSimilarityClassifier(
                    similarity_metric=metric,
                    k=k
                )
                
                # 训练和评估
                classifier.fit(X_train, y_train)
                results = classifier.evaluate(X_test, y_test)
                
                # 存储结果
                config_name = f"{metric}_k{k}"
                self.results[config_name] = results
                
                print(f"准确率: {results['accuracy']:.4f}")
    
    def print_results(self):
        """打印基准测试结果"""
        print("\n" + "="*50)
        print("像素相似性方法性能基准测试结果")
        print("="*50)
        
        for config_name, results in self.results.items():
            print(f"\n配置: {config_name}")
            print(f"总体准确率: {results['accuracy']:.4f}")
            print("各类别准确率:")
            for class_label, accuracy in results['class_accuracies'].items():
                print(f"  数字 {class_label}: {accuracy:.4f}")
    
    def get_best_config(self):
        """获取最佳配置"""
        if not self.results:
            return None
        
        best_config = max(self.results.items(), key=lambda x: x[1]['accuracy'])
        return best_config[0], best_config[1]


if __name__ == "__main__":
    # 测试像素相似性分类器
    print("像素相似性分类器测试")
    print("注意：此方法计算量较大，建议使用小样本进行测试")
    
    # 创建模拟数据进行测试
    np.random.seed(42)
    X_train = np.random.rand(100, 784)  # 100个训练样本
    y_train = np.random.randint(0, 10, 100)  # 10个类别
    X_test = np.random.rand(20, 784)  # 20个测试样本
    y_test = np.random.randint(0, 10, 20)
    
    # 测试基本功能
    classifier = PixelSimilarityClassifier(similarity_metric='euclidean', k=3)
    classifier.fit(X_train, y_train)
    
    # 预测单个样本
    prediction = classifier.predict_single(X_test[0])
    print(f"单个样本预测: {prediction}")
    
    # 预测所有测试样本
    predictions = classifier.predict(X_test)
    print(f"所有预测: {predictions}")
    
    # 评估模型
    results = classifier.evaluate(X_test, y_test)
    print(f"准确率: {results['accuracy']:.4f}")
    
    # 获取最相似的图像
    similar_images = classifier.get_most_similar_images(X_test[0], n=3)
    print(f"最相似的3个图像: {similar_images}")
    
    print("\n测试完成！") 