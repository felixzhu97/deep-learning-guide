#!/usr/bin/env python3
"""
像素相似性方法训练示例
演示如何使用像素相似性方法进行MNIST手写数字识别
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from time import time

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.mnist_loader import load_mnist
from data.preprocessor import preprocess_data
from models.pixel_similarity import PixelSimilarityClassifier, PixelSimilarityBenchmark
from utils.functions import accuracy_score


def visualize_similar_images(classifier, X_test, y_test, test_idx=0, n_similar=5):
    """
    可视化相似图像
    
    Args:
        classifier: 训练好的分类器
        X_test: 测试数据
        y_test: 测试标签
        test_idx: 测试图像索引
        n_similar: 显示的相似图像数量
    """
    test_image = X_test[test_idx]
    test_label = y_test[test_idx]
    
    # 获取最相似的图像
    similar_images = classifier.get_most_similar_images(test_image, n=n_similar)
    
    # 预测结果
    prediction = classifier.predict_single(test_image)
    
    # 创建图像展示
    fig, axes = plt.subplots(2, n_similar + 1, figsize=(15, 6))
    
    # 显示测试图像
    test_image_2d = test_image.reshape(28, 28)
    axes[0, 0].imshow(test_image_2d, cmap='gray')
    axes[0, 0].set_title(f'测试图像\n真实标签: {test_label}\n预测标签: {prediction}')
    axes[0, 0].axis('off')
    
    # 显示相似图像
    for i, (similarity, train_idx, train_label) in enumerate(similar_images):
        if i >= n_similar:
            break
            
        similar_image = classifier.X_train[train_idx].reshape(28, 28)
        axes[0, i + 1].imshow(similar_image, cmap='gray')
        axes[0, i + 1].set_title(f'相似图像 {i+1}\n标签: {train_label}\n相似度: {similarity:.3f}')
        axes[0, i + 1].axis('off')
    
    # 隐藏第二行未使用的子图
    for i in range(n_similar + 1):
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('pixel_similarity_examples.png', dpi=150, bbox_inches='tight')
    plt.show()


def run_quick_test():
    """运行快速测试"""
    print("=" * 60)
    print("像素相似性方法快速测试")
    print("=" * 60)
    
    # 加载数据
    print("正在加载MNIST数据...")
    train_images, train_labels, test_images, test_labels = load_mnist()
    
    # 预处理数据
    print("正在预处理数据...")
    X_train, y_train, X_test, y_test = preprocess_data(
        train_images, train_labels, test_images, test_labels,
        method='pixel_similarity'
    )
    
    # 为了快速测试，只使用部分数据
    print("使用部分数据进行快速测试...")
    X_train_small = X_train[:1000]  # 1000个训练样本
    y_train_small = y_train[:1000]
    X_test_small = X_test[:100]     # 100个测试样本
    y_test_small = y_test[:100]
    
    print(f"训练样本数量: {len(X_train_small)}")
    print(f"测试样本数量: {len(X_test_small)}")
    
    # 创建分类器
    classifier = PixelSimilarityClassifier(
        similarity_metric='euclidean',
        k=3
    )
    
    # 训练分类器
    print("\n正在训练分类器...")
    start_time = time()
    classifier.fit(X_train_small, y_train_small)
    train_time = time() - start_time
    print(f"训练时间: {train_time:.2f} 秒")
    
    # 评估分类器
    print("\n正在评估分类器...")
    start_time = time()
    results = classifier.evaluate(X_test_small, y_test_small)
    test_time = time() - start_time
    print(f"测试时间: {test_time:.2f} 秒")
    
    # 打印结果
    print("\n" + "=" * 40)
    print("快速测试结果")
    print("=" * 40)
    print(f"总体准确率: {results['accuracy']:.4f}")
    print(f"平均每样本预测时间: {test_time/len(X_test_small):.4f} 秒")
    
    print("\n各类别准确率:")
    for class_label, accuracy in sorted(results['class_accuracies'].items()):
        print(f"  数字 {class_label}: {accuracy:.4f}")
    
    # 可视化相似图像示例
    print("\n正在生成可视化示例...")
    try:
        visualize_similar_images(classifier, X_test_small, y_test_small, test_idx=0)
        print("可视化示例已保存为 'pixel_similarity_examples.png'")
    except Exception as e:
        print(f"可视化示例生成失败: {e}")
    
    return classifier, results


def run_comprehensive_benchmark():
    """运行全面的基准测试"""
    print("=" * 60)
    print("像素相似性方法全面基准测试")
    print("=" * 60)
    print("警告：此测试需要较长时间，请耐心等待...")
    
    # 加载数据
    print("正在加载MNIST数据...")
    train_images, train_labels, test_images, test_labels = load_mnist()
    
    # 预处理数据
    print("正在预处理数据...")
    X_train, y_train, X_test, y_test = preprocess_data(
        train_images, train_labels, test_images, test_labels,
        method='pixel_similarity'
    )
    
    # 运行基准测试
    benchmark = PixelSimilarityBenchmark()
    benchmark.run_benchmark(X_train, y_train, X_test, y_test, test_size=1000)
    
    # 打印结果
    benchmark.print_results()
    
    # 获取最佳配置
    best_config, best_results = benchmark.get_best_config()
    print(f"\n最佳配置: {best_config}")
    print(f"最佳准确率: {best_results['accuracy']:.4f}")
    
    return benchmark


def main():
    """主函数"""
    print("深度学习入门 - 像素相似性方法演示")
    print("基于《深度学习入门》一书的实现")
    print()
    
    # 询问用户选择
    print("请选择运行模式:")
    print("1. 快速测试 (推荐，约1-2分钟)")
    print("2. 全面基准测试 (较慢，约10-30分钟)")
    print("3. 仅加载数据测试")
    
    choice = input("请输入选择 (1/2/3): ").strip()
    
    if choice == '1':
        # 快速测试
        classifier, results = run_quick_test()
        
        # 提供进一步的交互选项
        print("\n" + "=" * 40)
        print("快速测试完成！")
        print("=" * 40)
        
        while True:
            print("\n可选操作:")
            print("1. 查看更多相似图像示例")
            print("2. 测试自定义样本")
            print("3. 退出")
            
            sub_choice = input("请输入选择 (1/2/3): ").strip()
            
            if sub_choice == '1':
                # 查看更多示例
                for i in range(min(5, len(X_test_small))):
                    print(f"\n测试样本 {i}:")
                    visualize_similar_images(classifier, X_test_small, y_test_small, test_idx=i)
                    
                    continue_choice = input("继续查看下一个? (y/n): ").strip().lower()
                    if continue_choice != 'y':
                        break
                        
            elif sub_choice == '2':
                # 测试自定义样本
                try:
                    idx = int(input("请输入要测试的样本索引 (0-99): "))
                    if 0 <= idx < len(X_test_small):
                        visualize_similar_images(classifier, X_test_small, y_test_small, test_idx=idx)
                    else:
                        print("索引超出范围！")
                except ValueError:
                    print("请输入有效的数字！")
                    
            elif sub_choice == '3':
                break
            else:
                print("无效选择，请重新输入！")
    
    elif choice == '2':
        # 全面基准测试
        benchmark = run_comprehensive_benchmark()
        
    elif choice == '3':
        # 仅测试数据加载
        print("正在测试数据加载...")
        try:
            train_images, train_labels, test_images, test_labels = load_mnist()
            print("数据加载成功！")
            print(f"训练集: {train_images.shape}")
            print(f"测试集: {test_images.shape}")
            
            # 预处理测试
            X_train, y_train, X_test, y_test = preprocess_data(
                train_images, train_labels, test_images, test_labels,
                method='pixel_similarity'
            )
            print("数据预处理成功！")
            print(f"处理后训练集: {X_train.shape}")
            print(f"处理后测试集: {X_test.shape}")
            
        except Exception as e:
            print(f"数据加载失败: {e}")
    
    else:
        print("无效选择，退出程序")
        return
    
    print("\n程序执行完成！")


if __name__ == "__main__":
    main() 