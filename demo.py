#!/usr/bin/env python3
"""
深度学习入门项目演示脚本
项目主要入口点，提供各种演示和测试功能
"""

import os
import sys
import time
import numpy as np

# 确保项目根目录在Python路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.mnist_loader import load_mnist
from data.preprocessor import preprocess_data
from models.pixel_similarity import PixelSimilarityClassifier
from models.neural_network import NeuralNetwork
from utils.visualization import plot_model_comparison


def print_header():
    """打印项目头部信息"""
    print("\n" + "="*70)
    print("🚀 深度学习入门项目演示")
    print("📚 基于《深度学习入门》（斋藤康毅著）")
    print("🎯 MNIST手写数字识别 - 从基础到深度学习")
    print("="*70)


def check_dependencies():
    """检查项目依赖"""
    print("📋 检查项目依赖...")
    
    required_packages = [
        'numpy', 'matplotlib', 'scikit-learn', 
        'requests', 'tqdm', 'Pillow'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - 缺失")
    
    if missing_packages:
        print(f"\n⚠️  缺少以下依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    print("✅ 所有依赖都已安装")
    return True


def quick_demo():
    """快速演示"""
    print("\n🎬 快速演示开始...")
    print("使用少量数据展示各种方法的效果")
    
    # 加载数据
    print("\n📥 加载MNIST数据...")
    train_images, train_labels, test_images, test_labels = load_mnist()
    
    # 使用少量数据进行演示
    print("🔧 准备演示数据...")
    train_sample = 1000
    test_sample = 200
    
    X_train = train_images[:train_sample]
    y_train = train_labels[:train_sample]
    X_test = test_images[:test_sample]
    y_test = test_labels[:test_sample]
    
    print(f"训练样本: {train_sample}, 测试样本: {test_sample}")
    
    results = {}
    
    # 1. 像素相似性方法
    print("\n🔍 测试像素相似性方法...")
    X_train_pixel, y_train_pixel, X_test_pixel, y_test_pixel = preprocess_data(
        X_train, y_train, X_test, y_test, method='pixel_similarity'
    )
    
    pixel_classifier = PixelSimilarityClassifier(similarity_metric='euclidean', k=3)
    pixel_classifier.fit(X_train_pixel, y_train_pixel)
    
    # 只用少量测试样本以节省时间
    test_subset = 50
    pixel_results = pixel_classifier.evaluate(X_test_pixel[:test_subset], y_test_pixel[:test_subset])
    results['像素相似性'] = pixel_results['accuracy']
    
    print(f"✅ 像素相似性准确率: {pixel_results['accuracy']:.3f}")
    
    # 2. 简单神经网络
    print("\n🧠 测试简单神经网络...")
    X_train_nn, y_train_nn, X_test_nn, y_test_nn = preprocess_data(
        X_train, y_train, X_test, y_test, method='neural_network'
    )
    
    nn = NeuralNetwork(
        input_size=784,
        hidden_sizes=[64, 32],
        output_size=10,
        activation='relu'
    )
    
    nn.fit(X_train_nn, y_train_nn, epochs=10, batch_size=32, 
           learning_rate=0.01, verbose=False)
    
    nn_results = nn.evaluate(X_test_nn, y_test_nn)
    results['简单神经网络'] = nn_results['accuracy']
    
    print(f"✅ 神经网络准确率: {nn_results['accuracy']:.3f}")
    
    # 结果对比
    print("\n📊 结果对比:")
    print("-" * 30)
    for method, accuracy in results.items():
        print(f"{method:12s}: {accuracy:.3f}")
    
    # 可视化对比
    print("\n📈 生成对比图表...")
    plot_model_comparison(
        results, 
        metric='准确率',
        title="MNIST分类方法对比（快速演示）",
        save_path="quick_demo_comparison.png"
    )
    
    print("\n🎉 快速演示完成！")
    return results


def comprehensive_demo():
    """全面演示"""
    print("\n🎯 全面演示开始...")
    print("使用更多数据和完整功能进行演示")
    
    # 询问用户确认
    confirm = input("全面演示需要较长时间（10-30分钟），是否继续？(y/n): ").strip().lower()
    if confirm != 'y':
        print("已取消全面演示")
        return
    
    # 运行各个模块的完整演示
    print("\n🔍 运行像素相似性方法完整演示...")
    os.system("python examples/train_pixel_similarity.py")
    
    print("\n🧠 运行神经网络完整演示...")
    os.system("python examples/train_neural_network.py")
    
    print("\n🎉 全面演示完成！")


def data_exploration():
    """数据探索"""
    print("\n🔍 MNIST数据集探索...")
    
    # 加载数据
    train_images, train_labels, test_images, test_labels = load_mnist()
    
    print(f"📊 数据集统计:")
    print(f"   训练集: {train_images.shape[0]:,} 个样本")
    print(f"   测试集: {test_images.shape[0]:,} 个样本")
    print(f"   图像尺寸: {train_images.shape[1]}×{train_images.shape[2]}")
    print(f"   像素值范围: [{train_images.min()}, {train_images.max()}]")
    print(f"   标签范围: [{train_labels.min()}, {train_labels.max()}]")
    
    # 各类别样本数量
    print(f"\n📈 各类别样本数量:")
    for i in range(10):
        count = np.sum(train_labels == i)
        print(f"   数字 {i}: {count:,} 个样本")
    
    # 生成一些样本图像
    from utils.visualization import visualize_mnist_images
    
    print("\n🖼️  生成样本图像...")
    sample_indices = np.random.choice(len(train_images), 25, replace=False)
    sample_images = train_images[sample_indices]
    sample_labels = train_labels[sample_indices]
    
    visualize_mnist_images(
        sample_images, sample_labels,
        title="MNIST数据集样本",
        save_path="mnist_samples.png"
    )
    
    print("✅ 数据探索完成！样本图像已保存为 mnist_samples.png")


def project_info():
    """项目信息"""
    print("\n📋 项目信息:")
    print("-" * 50)
    
    # 统计项目文件
    python_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    print(f"🐍 Python文件数量: {len(python_files)}")
    print(f"📁 项目目录:")
    
    directories = ['data', 'models', 'utils', 'examples']
    for directory in directories:
        if os.path.exists(directory):
            files = [f for f in os.listdir(directory) if f.endswith('.py')]
            print(f"   {directory}/: {len(files)} 个文件")
    
    print(f"\n📚 实现的功能:")
    print(f"   ✅ MNIST数据加载和预处理")
    print(f"   ✅ 像素相似性分类方法")
    print(f"   ✅ 多层感知机神经网络")
    print(f"   ✅ 反向传播算法")
    print(f"   ✅ 多种优化器（SGD、Adam）")
    print(f"   ✅ 训练过程可视化")
    print(f"   ✅ 性能评估和对比")
    
    print(f"\n🎯 学习目标:")
    print(f"   📖 理解深度学习基础概念")
    print(f"   🔧 掌握从零实现神经网络")
    print(f"   📊 学会评估和优化模型")
    print(f"   🎨 体验渐进式学习过程")


def main():
    """主函数"""
    print_header()
    
    # 检查依赖
    if not check_dependencies():
        print("\n❌ 请先安装依赖包")
        return
    
    while True:
        print("\n🎮 请选择操作:")
        print("1. 🚀 快速演示 (推荐，5-10分钟)")
        print("2. 🎯 全面演示 (完整功能，10-30分钟)")
        print("3. 🔍 数据探索")
        print("4. 📋 项目信息")
        print("5. 📚 查看README")
        print("6. 🚪 退出")
        
        choice = input("\n请输入选择 (1-6): ").strip()
        
        if choice == '1':
            quick_demo()
        elif choice == '2':
            comprehensive_demo()
        elif choice == '3':
            data_exploration()
        elif choice == '4':
            project_info()
        elif choice == '5':
            if os.path.exists('README.md'):
                with open('README.md', 'r', encoding='utf-8') as f:
                    content = f.read()
                print("\n📖 README.md:")
                print("-" * 50)
                print(content)
            else:
                print("❌ README.md 文件不存在")
        elif choice == '6':
            break
        else:
            print("❌ 无效选择，请重新输入")
    
    print("\n👋 感谢使用深度学习入门项目！")
    print("📧 如有问题，请查看README.md或项目文档")
    print("🌟 祝您深度学习之旅愉快！")


if __name__ == "__main__":
    main() 