"""
可视化工具模块
用于绘制训练过程和结果
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns


plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_loss(train_losses, val_losses=None, title="损失函数变化", save_path=None):
    """
    绘制损失函数变化图
    
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        title: 图表标题
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='训练损失', linewidth=2)
    
    if val_losses is not None:
        plt.plot(epochs, val_losses, 'r-', label='验证损失', linewidth=2)
    
    plt.xlabel('训练轮数')
    plt.ylabel('损失值')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"损失图表已保存至: {save_path}")
    
    plt.show()


def plot_accuracy(train_accuracies, val_accuracies=None, title="准确率变化", save_path=None):
    """
    绘制准确率变化图
    
    Args:
        train_accuracies: 训练准确率列表
        val_accuracies: 验证准确率列表
        title: 图表标题
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_accuracies) + 1)
    plt.plot(epochs, train_accuracies, 'b-', label='训练准确率', linewidth=2)
    
    if val_accuracies is not None:
        plt.plot(epochs, val_accuracies, 'r-', label='验证准确率', linewidth=2)
    
    plt.xlabel('训练轮数')
    plt.ylabel('准确率')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"准确率图表已保存至: {save_path}")
    
    plt.show()


def plot_training_history(train_losses, train_accuracies, val_losses=None, val_accuracies=None, 
                         title="训练历史", save_path=None):
    """
    绘制训练历史（损失和准确率）
    
    Args:
        train_losses: 训练损失列表
        train_accuracies: 训练准确率列表
        val_losses: 验证损失列表
        val_accuracies: 验证准确率列表
        title: 图表标题
        save_path: 保存路径
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    # 绘制损失
    ax1.plot(epochs, train_losses, 'b-', label='训练损失', linewidth=2)
    if val_losses is not None:
        ax1.plot(epochs, val_losses, 'r-', label='验证损失', linewidth=2)
    ax1.set_xlabel('训练轮数')
    ax1.set_ylabel('损失值')
    ax1.set_title('损失函数变化')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 绘制准确率
    ax2.plot(epochs, train_accuracies, 'b-', label='训练准确率', linewidth=2)
    if val_accuracies is not None:
        ax2.plot(epochs, val_accuracies, 'r-', label='验证准确率', linewidth=2)
    ax2.set_xlabel('训练轮数')
    ax2.set_ylabel('准确率')
    ax2.set_title('准确率变化')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练历史图表已保存至: {save_path}")
    
    plt.show()


def visualize_weights(weights, layer_name="权重", save_path=None):
    """
    可视化权重矩阵
    
    Args:
        weights: 权重矩阵
        layer_name: 层名称
        save_path: 保存路径
    """
    plt.figure(figsize=(12, 8))
    
    # 如果权重是向量，重塑为矩阵
    if len(weights.shape) == 1:
        weights = weights.reshape(-1, 1)
    
    # 绘制热力图
    sns.heatmap(weights, cmap='RdBu_r', center=0, 
                xticklabels=False, yticklabels=False,
                cbar_kws={'label': '权重值'})
    
    plt.title(f'{layer_name} 权重可视化')
    plt.xlabel('输出神经元')
    plt.ylabel('输入神经元')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"权重可视化图表已保存至: {save_path}")
    
    plt.show()


def visualize_mnist_images(images, labels, predictions=None, num_images=25, title="MNIST图像", save_path=None):
    """
    可视化MNIST图像
    
    Args:
        images: 图像数据
        labels: 真实标签
        predictions: 预测标签
        num_images: 显示图像数量
        title: 图表标题
        save_path: 保存路径
    """
    fig, axes = plt.subplots(5, 5, figsize=(12, 12))
    axes = axes.ravel()
    
    for i in range(min(num_images, len(images))):
        # 确保图像是二维的
        if len(images[i].shape) == 1:
            image = images[i].reshape(28, 28)
        else:
            image = images[i]
        
        axes[i].imshow(image, cmap='gray')
        
        # 设置标题
        if predictions is not None:
            correct = '✓' if labels[i] == predictions[i] else '✗'
            axes[i].set_title(f'{correct} 真实: {labels[i]}, 预测: {predictions[i]}')
        else:
            axes[i].set_title(f'标签: {labels[i]}')
        
        axes[i].axis('off')
    
    # 隐藏多余的子图
    for i in range(num_images, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"MNIST图像可视化已保存至: {save_path}")
    
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names=None, title="混淆矩阵", save_path=None):
    """
    绘制混淆矩阵
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称
        title: 图表标题
        save_path: 保存路径
    """
    from sklearn.metrics import confusion_matrix
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    
    # 绘制热力图
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title(title)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存至: {save_path}")
    
    plt.show()


def plot_class_accuracies(class_accuracies, title="各类别准确率", save_path=None):
    """
    绘制各类别准确率条形图
    
    Args:
        class_accuracies: 各类别准确率字典
        title: 图表标题
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 6))
    
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())
    
    bars = plt.bar(classes, accuracies, color='skyblue', edgecolor='navy', alpha=0.7)
    
    # 在条形图上添加数值
    for bar, accuracy in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{accuracy:.3f}', ha='center', va='bottom')
    
    plt.xlabel('类别')
    plt.ylabel('准确率')
    plt.title(title)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"类别准确率图表已保存至: {save_path}")
    
    plt.show()


def plot_model_comparison(results_dict, metric='accuracy', title="模型对比", save_path=None):
    """
    绘制模型对比图
    
    Args:
        results_dict: 模型结果字典，如 {'model1': accuracy1, 'model2': accuracy2}
        metric: 对比指标
        title: 图表标题
        save_path: 保存路径
    """
    plt.figure(figsize=(12, 6))
    
    models = list(results_dict.keys())
    values = list(results_dict.values())
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    bars = plt.bar(models, values, color=colors, edgecolor='black', alpha=0.8)
    
    # 在条形图上添加数值
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.xlabel('模型')
    plt.ylabel(metric.capitalize())
    plt.title(title)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"模型对比图表已保存至: {save_path}")
    
    plt.show()


def plot_learning_curve(train_sizes, train_scores, val_scores, title="学习曲线", save_path=None):
    """
    绘制学习曲线
    
    Args:
        train_sizes: 训练集大小
        train_scores: 训练分数
        val_scores: 验证分数
        title: 图表标题
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 6))
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.plot(train_sizes, train_mean, 'b-', label='训练分数', linewidth=2)
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color='blue')
    
    plt.plot(train_sizes, val_mean, 'r-', label='验证分数', linewidth=2)
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2, color='red')
    
    plt.xlabel('训练集大小')
    plt.ylabel('分数')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"学习曲线已保存至: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # 测试可视化函数
    print("可视化工具测试")
    
    # 生成模拟数据
    np.random.seed(42)
    epochs = 50
    train_losses = np.exp(-np.linspace(0, 3, epochs)) + np.random.normal(0, 0.1, epochs)
    val_losses = np.exp(-np.linspace(0, 2.5, epochs)) + np.random.normal(0, 0.1, epochs)
    train_accuracies = 1 - np.exp(-np.linspace(0, 3, epochs)) + np.random.normal(0, 0.05, epochs)
    val_accuracies = 1 - np.exp(-np.linspace(0, 2.5, epochs)) + np.random.normal(0, 0.05, epochs)
    
    # 测试训练历史可视化
    print("\n测试训练历史可视化...")
    plot_training_history(train_losses, train_accuracies, val_losses, val_accuracies)
    
    # 测试权重可视化
    print("\n测试权重可视化...")
    weights = np.random.randn(20, 10)
    visualize_weights(weights, "测试层")
    
    # 测试类别准确率可视化
    print("\n测试类别准确率可视化...")
    class_accuracies = {i: np.random.uniform(0.8, 0.99) for i in range(10)}
    plot_class_accuracies(class_accuracies)
    
    # 测试模型对比
    print("\n测试模型对比...")
    results = {
        '像素相似性': 0.52,
        '简单神经网络': 0.89,
        '卷积神经网络': 0.98
    }
    plot_model_comparison(results, metric='accuracy', title="MNIST分类方法对比")
    
    print("\n可视化工具测试完成！") 