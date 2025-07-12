#!/usr/bin/env python3
"""
基本功能测试脚本
验证项目各模块的基本功能
"""

import sys
import os
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """测试导入功能"""
    print("测试模块导入...")
    
    try:
        from data.mnist_loader import load_mnist
        from data.preprocessor import preprocess_data
        from models.pixel_similarity import PixelSimilarityClassifier
        from models.neural_network import NeuralNetwork
        from utils.functions import sigmoid, relu, softmax
        from utils.optimizers import SGD, Adam
        from utils.visualization import plot_loss
        print("✅ 所有模块导入成功")
        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return False


def test_functions():
    """测试工具函数"""
    print("测试工具函数...")
    
    try:
        from utils.functions import sigmoid, relu, softmax, cross_entropy_loss
        
        # 测试数据
        x = np.array([-1, 0, 1, 2])
        
        # 测试激活函数
        sig_result = sigmoid(x)
        relu_result = relu(x)
        
        # 测试softmax
        logits = np.array([[1, 2, 3], [4, 5, 6]])
        softmax_result = softmax(logits)
        
        # 测试损失函数
        y_true = np.array([[0, 1, 0], [1, 0, 0]])
        y_pred = np.array([[0.1, 0.8, 0.1], [0.9, 0.05, 0.05]])
        loss = cross_entropy_loss(y_true, y_pred)
        
        print("✅ 工具函数测试通过")
        return True
    except Exception as e:
        print(f"❌ 工具函数测试失败: {e}")
        return False


def test_data_preprocessing():
    """测试数据预处理"""
    print("测试数据预处理...")
    
    try:
        from data.preprocessor import DataPreprocessor
        
        # 创建模拟数据
        images = np.random.randint(0, 255, (100, 28, 28))
        labels = np.random.randint(0, 10, 100)
        
        preprocessor = DataPreprocessor()
        
        # 测试归一化
        normalized = preprocessor.normalize_images(images)
        assert normalized.min() >= 0 and normalized.max() <= 1, "归一化失败"
        
        # 测试展平
        flattened = preprocessor.flatten_images(images)
        assert flattened.shape == (100, 784), "展平失败"
        
        # 测试one-hot编码
        onehot = preprocessor.one_hot_encode(labels)
        assert onehot.shape == (100, 10), "one-hot编码失败"
        
        print("✅ 数据预处理测试通过")
        return True
    except Exception as e:
        print(f"❌ 数据预处理测试失败: {e}")
        return False


def test_pixel_similarity():
    """测试像素相似性分类器"""
    print("测试像素相似性分类器...")
    
    try:
        from models.pixel_similarity import PixelSimilarityClassifier
        
        # 创建模拟数据
        X_train = np.random.rand(50, 784)
        y_train = np.random.randint(0, 10, 50)
        X_test = np.random.rand(10, 784)
        y_test = np.random.randint(0, 10, 10)
        
        # 创建分类器
        classifier = PixelSimilarityClassifier(similarity_metric='euclidean', k=3)
        
        # 训练
        classifier.fit(X_train, y_train)
        
        # 预测
        predictions = classifier.predict(X_test)
        assert len(predictions) == len(X_test), "预测结果长度不正确"
        
        # 评估
        results = classifier.evaluate(X_test, y_test)
        assert 'accuracy' in results, "评估结果缺少准确率"
        
        print("✅ 像素相似性分类器测试通过")
        return True
    except Exception as e:
        print(f"❌ 像素相似性分类器测试失败: {e}")
        return False


def test_neural_network():
    """测试神经网络"""
    print("测试神经网络...")
    
    try:
        from models.neural_network import NeuralNetwork
        
        # 创建模拟数据
        X_train = np.random.randn(100, 784)
        y_train = np.random.randint(0, 10, (100, 10))  # one-hot编码
        X_test = np.random.randn(20, 784)
        y_test = np.random.randint(0, 10, (20, 10))
        
        # 创建网络
        nn = NeuralNetwork(
            input_size=784,
            hidden_sizes=[64, 32],
            output_size=10,
            activation='relu'
        )
        
        # 测试前向传播
        output, _, _ = nn.forward(X_test)
        assert output.shape == (20, 10), "前向传播输出形状不正确"
        
        # 测试训练
        nn.fit(X_train, y_train, epochs=2, batch_size=32, verbose=False)
        
        # 测试预测
        predictions = nn.predict(X_test)
        assert predictions.shape == (20, 10), "预测输出形状不正确"
        
        # 测试评估
        results = nn.evaluate(X_test, y_test)
        assert 'accuracy' in results, "评估结果缺少准确率"
        
        print("✅ 神经网络测试通过")
        return True
    except Exception as e:
        print(f"❌ 神经网络测试失败: {e}")
        return False


def test_optimizers():
    """测试优化器"""
    print("测试优化器...")
    
    try:
        from utils.optimizers import SGD, Adam
        
        # 创建模拟参数和梯度
        param = np.random.randn(10, 5)
        grad = np.random.randn(10, 5)
        
        # 测试SGD
        sgd = SGD(learning_rate=0.01, momentum=0.9)
        updated_param = sgd.update('test_param', param, grad)
        assert updated_param.shape == param.shape, "SGD更新后参数形状不正确"
        
        # 测试Adam
        adam = Adam(learning_rate=0.001)
        updated_param = adam.update('test_param', param, grad)
        assert updated_param.shape == param.shape, "Adam更新后参数形状不正确"
        
        print("✅ 优化器测试通过")
        return True
    except Exception as e:
        print(f"❌ 优化器测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("=" * 50)
    print("🔍 深度学习入门项目基本功能测试")
    print("=" * 50)
    
    tests = [
        ("模块导入", test_imports),
        ("工具函数", test_functions),
        ("数据预处理", test_data_preprocessing),
        ("像素相似性", test_pixel_similarity),
        ("神经网络", test_neural_network),
        ("优化器", test_optimizers),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n🧪 {test_name}测试:")
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_name}测试异常: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print("📊 测试结果汇总:")
    print(f"✅ 通过: {passed}")
    print(f"❌ 失败: {failed}")
    print(f"📈 成功率: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 所有测试通过！项目基本功能正常。")
    else:
        print(f"\n⚠️  有 {failed} 个测试失败，请检查相关模块。")
    
    print("=" * 50)


if __name__ == "__main__":
    main() 