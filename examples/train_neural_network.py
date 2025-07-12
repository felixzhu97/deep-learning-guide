#!/usr/bin/env python3
"""
神经网络训练示例
演示如何使用多层感知机进行MNIST手写数字识别
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
from models.neural_network import NeuralNetwork
from utils.visualization import plot_training_history, visualize_mnist_images, plot_confusion_matrix


def run_neural_network_demo():
    """运行神经网络演示"""
    print("=" * 60)
    print("神经网络训练演示")
    print("=" * 60)
    
    # 加载数据
    print("正在加载MNIST数据...")
    train_images, train_labels, test_images, test_labels = load_mnist()
    
    # 预处理数据
    print("正在预处理数据...")
    X_train, y_train, X_test, y_test = preprocess_data(
        train_images, train_labels, test_images, test_labels,
        method='neural_network'
    )
    
    print(f"训练集形状: {X_train.shape}")
    print(f"训练标签形状: {y_train.shape}")
    print(f"测试集形状: {X_test.shape}")
    print(f"测试标签形状: {y_test.shape}")
    
    # 为演示目的，使用部分数据
    print("\n使用部分数据进行演示...")
    X_train_demo = X_train[:10000]
    y_train_demo = y_train[:10000]
    X_test_demo = X_test[:2000]
    y_test_demo = y_test[:2000]
    
    print(f"演示训练集大小: {X_train_demo.shape[0]}")
    print(f"演示测试集大小: {X_test_demo.shape[0]}")
    
    # 创建验证集
    val_size = 2000
    X_val = X_train_demo[-val_size:]
    y_val = y_train_demo[-val_size:]
    X_train_demo = X_train_demo[:-val_size]
    y_train_demo = y_train_demo[:-val_size]
    
    print(f"最终训练集大小: {X_train_demo.shape[0]}")
    print(f"验证集大小: {X_val.shape[0]}")
    
    # 创建神经网络
    print("\n创建神经网络...")
    nn = NeuralNetwork(
        input_size=784,
        hidden_sizes=[128, 64],
        output_size=10,
        activation='relu'
    )
    
    # 打印网络架构
    arch_summary = nn.get_architecture_summary()
    print("\n网络架构:")
    print("-" * 50)
    for layer_info in arch_summary['summary']:
        print(f"Layer {layer_info['layer']}: {layer_info['input_size']} → {layer_info['output_size']}")
        print(f"  激活函数: {layer_info['activation']}")
        print(f"  参数数量: {layer_info['parameters']:,}")
    print(f"总参数数量: {arch_summary['total_parameters']:,}")
    
    # 训练网络
    print("\n开始训练...")
    start_time = time()
    
    nn.fit(
        X_train_demo, y_train_demo,
        X_val, y_val,
        epochs=20,
        batch_size=128,
        learning_rate=0.001,
        verbose=True,
        early_stopping=True,
        patience=5
    )
    
    train_time = time() - start_time
    print(f"\n训练完成！总训练时间: {train_time:.2f} 秒")
    
    # 评估网络
    print("\n评估网络性能...")
    
    # 训练集评估
    train_results = nn.evaluate(X_train_demo, y_train_demo)
    print(f"训练集 - 损失: {train_results['loss']:.4f}, 准确率: {train_results['accuracy']:.4f}")
    
    # 验证集评估
    val_results = nn.evaluate(X_val, y_val)
    print(f"验证集 - 损失: {val_results['loss']:.4f}, 准确率: {val_results['accuracy']:.4f}")
    
    # 测试集评估
    test_results = nn.evaluate(X_test_demo, y_test_demo)
    print(f"测试集 - 损失: {test_results['loss']:.4f}, 准确率: {test_results['accuracy']:.4f}")
    
    # 各类别准确率
    print("\n各类别准确率:")
    for class_label, accuracy in sorted(test_results['class_accuracies'].items()):
        print(f"  数字 {class_label}: {accuracy:.4f}")
    
    # 可视化训练历史
    print("\n生成训练历史图表...")
    plot_training_history(
        nn.train_losses, nn.train_accuracies,
        nn.val_losses, nn.val_accuracies,
        title="神经网络训练历史",
        save_path="neural_network_training_history.png"
    )
    
    # 可视化预测结果
    print("\n生成预测结果可视化...")
    test_predictions = nn.predict_classes(X_test_demo)
    true_labels = np.argmax(y_test_demo, axis=1)
    
    # 找到一些预测错误的例子
    wrong_indices = np.where(test_predictions != true_labels)[0]
    if len(wrong_indices) > 0:
        print(f"发现 {len(wrong_indices)} 个预测错误的样本")
        
        # 显示一些错误样本
        sample_indices = wrong_indices[:25] if len(wrong_indices) >= 25 else wrong_indices
        sample_images = X_test_demo[sample_indices].reshape(-1, 28, 28)
        sample_true = true_labels[sample_indices]
        sample_pred = test_predictions[sample_indices]
        
        visualize_mnist_images(
            sample_images, sample_true, sample_pred,
            title="预测错误的样本",
            save_path="neural_network_errors.png"
        )
    
    # 显示一些正确预测的样本
    correct_indices = np.where(test_predictions == true_labels)[0]
    if len(correct_indices) > 0:
        sample_indices = correct_indices[:25]
        sample_images = X_test_demo[sample_indices].reshape(-1, 28, 28)
        sample_true = true_labels[sample_indices]
        sample_pred = test_predictions[sample_indices]
        
        visualize_mnist_images(
            sample_images, sample_true, sample_pred,
            title="预测正确的样本",
            save_path="neural_network_correct.png"
        )
    
    # 绘制混淆矩阵
    print("\n生成混淆矩阵...")
    plot_confusion_matrix(
        true_labels, test_predictions,
        class_names=[str(i) for i in range(10)],
        title="神经网络混淆矩阵",
        save_path="neural_network_confusion_matrix.png"
    )
    
    return nn, test_results


def compare_different_architectures():
    """比较不同网络架构的性能"""
    print("=" * 60)
    print("神经网络架构对比")
    print("=" * 60)
    
    # 加载和预处理数据
    print("正在加载和预处理数据...")
    train_images, train_labels, test_images, test_labels = load_mnist()
    X_train, y_train, X_test, y_test = preprocess_data(
        train_images, train_labels, test_images, test_labels,
        method='neural_network'
    )
    
    # 使用小规模数据集进行对比
    X_train_small = X_train[:5000]
    y_train_small = y_train[:5000]
    X_test_small = X_test[:1000]
    y_test_small = y_test[:1000]
    
    # 定义不同的网络架构
    architectures = [
        {'name': '单隐藏层(64)', 'hidden_sizes': [64]},
        {'name': '单隐藏层(128)', 'hidden_sizes': [128]},
        {'name': '双隐藏层(128,64)', 'hidden_sizes': [128, 64]},
        {'name': '双隐藏层(256,128)', 'hidden_sizes': [256, 128]},
        {'name': '三隐藏层(128,64,32)', 'hidden_sizes': [128, 64, 32]},
    ]
    
    results = {}
    
    for arch in architectures:
        print(f"\n测试架构: {arch['name']}")
        
        # 创建网络
        nn = NeuralNetwork(
            input_size=784,
            hidden_sizes=arch['hidden_sizes'],
            output_size=10,
            activation='relu'
        )
        
        # 训练网络
        nn.fit(
            X_train_small, y_train_small,
            epochs=10,
            batch_size=64,
            learning_rate=0.001,
            verbose=False
        )
        
        # 评估性能
        test_results = nn.evaluate(X_test_small, y_test_small)
        results[arch['name']] = test_results['accuracy']
        
        print(f"准确率: {test_results['accuracy']:.4f}")
    
    # 可视化对比结果
    print("\n生成架构对比图表...")
    from utils.visualization import plot_model_comparison
    plot_model_comparison(
        results, 
        metric='accuracy',
        title="不同神经网络架构性能对比",
        save_path="neural_network_architecture_comparison.png"
    )
    
    # 找到最佳架构
    best_arch = max(results.items(), key=lambda x: x[1])
    print(f"\n最佳架构: {best_arch[0]}")
    print(f"最佳准确率: {best_arch[1]:.4f}")
    
    return results


def hyperparameter_tuning():
    """超参数调优演示"""
    print("=" * 60)
    print("超参数调优演示")
    print("=" * 60)
    
    # 加载数据
    print("正在加载数据...")
    train_images, train_labels, test_images, test_labels = load_mnist()
    X_train, y_train, X_test, y_test = preprocess_data(
        train_images, train_labels, test_images, test_labels,
        method='neural_network'
    )
    
    # 使用小规模数据集
    X_train_small = X_train[:3000]
    y_train_small = y_train[:3000]
    X_test_small = X_test[:500]
    y_test_small = y_test[:500]
    
    # 定义超参数组合
    hyperparams = [
        {'lr': 0.001, 'batch_size': 32, 'name': 'LR:0.001, BS:32'},
        {'lr': 0.001, 'batch_size': 64, 'name': 'LR:0.001, BS:64'},
        {'lr': 0.001, 'batch_size': 128, 'name': 'LR:0.001, BS:128'},
        {'lr': 0.01, 'batch_size': 64, 'name': 'LR:0.01, BS:64'},
        {'lr': 0.1, 'batch_size': 64, 'name': 'LR:0.1, BS:64'},
    ]
    
    results = {}
    
    for params in hyperparams:
        print(f"\n测试参数: {params['name']}")
        
        # 创建网络
        nn = NeuralNetwork(
            input_size=784,
            hidden_sizes=[128, 64],
            output_size=10,
            activation='relu'
        )
        
        # 训练网络
        nn.fit(
            X_train_small, y_train_small,
            epochs=15,
            batch_size=params['batch_size'],
            learning_rate=params['lr'],
            verbose=False
        )
        
        # 评估性能
        test_results = nn.evaluate(X_test_small, y_test_small)
        results[params['name']] = test_results['accuracy']
        
        print(f"准确率: {test_results['accuracy']:.4f}")
    
    # 可视化对比结果
    print("\n生成超参数对比图表...")
    from utils.visualization import plot_model_comparison
    plot_model_comparison(
        results, 
        metric='accuracy',
        title="超参数调优结果对比",
        save_path="neural_network_hyperparameter_tuning.png"
    )
    
    # 找到最佳参数
    best_params = max(results.items(), key=lambda x: x[1])
    print(f"\n最佳参数: {best_params[0]}")
    print(f"最佳准确率: {best_params[1]:.4f}")
    
    return results


def main():
    """主函数"""
    print("深度学习入门 - 神经网络训练演示")
    print("基于《深度学习入门》一书的实现")
    print()
    
    # 询问用户选择
    print("请选择运行模式:")
    print("1. 基本神经网络演示")
    print("2. 网络架构对比")
    print("3. 超参数调优")
    print("4. 全部运行")
    
    choice = input("请输入选择 (1/2/3/4): ").strip()
    
    if choice == '1':
        # 基本演示
        nn, results = run_neural_network_demo()
        print("\n基本演示完成！")
        
    elif choice == '2':
        # 架构对比
        results = compare_different_architectures()
        print("\n架构对比完成！")
        
    elif choice == '3':
        # 超参数调优
        results = hyperparameter_tuning()
        print("\n超参数调优完成！")
        
    elif choice == '4':
        # 全部运行
        print("运行全部演示...")
        
        print("\n第一部分: 基本神经网络演示")
        nn, demo_results = run_neural_network_demo()
        
        print("\n第二部分: 网络架构对比")
        arch_results = compare_different_architectures()
        
        print("\n第三部分: 超参数调优")
        param_results = hyperparameter_tuning()
        
        # 综合结果展示
        print("\n" + "="*60)
        print("综合结果总结")
        print("="*60)
        print(f"基本演示最终准确率: {demo_results['accuracy']:.4f}")
        
        best_arch = max(arch_results.items(), key=lambda x: x[1])
        print(f"最佳架构: {best_arch[0]} (准确率: {best_arch[1]:.4f})")
        
        best_params = max(param_results.items(), key=lambda x: x[1])
        print(f"最佳参数: {best_params[0]} (准确率: {best_params[1]:.4f})")
        
        print("\n全部演示完成！")
        
    else:
        print("无效选择，退出程序")
        return
    
    print("\n" + "="*60)
    print("神经网络训练演示完成！")
    print("="*60)
    print("生成的文件:")
    print("- neural_network_training_history.png: 训练历史")
    print("- neural_network_errors.png: 预测错误样本")
    print("- neural_network_correct.png: 预测正确样本")
    print("- neural_network_confusion_matrix.png: 混淆矩阵")
    print("- neural_network_architecture_comparison.png: 架构对比")
    print("- neural_network_hyperparameter_tuning.png: 超参数调优")
    print("\n感谢使用深度学习入门项目！")


if __name__ == "__main__":
    main() 