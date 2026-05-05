#!/usr/bin/env python3
"""
任务一：极简CNN识别MNIST
基于微信公众号文章《计算机视觉》第10篇实现
参考来源：TensorFlow官方教程 + 文章结构
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import time
import os

def load_data():
    """加载MNIST数据并预处理"""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # 归一化 + reshape添加通道维度
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    # one-hot编码
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    return (x_train, y_train), (x_test, y_test)

def build_model():
    """构建极简CNN模型"""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

def main():
    print("=" * 50)
    print("任务一：极简CNN训练")
    print("=" * 50)
    
    # 加载数据
    print("\n[1/4] 加载MNIST数据...")
    (x_train, y_train), (x_test, y_test) = load_data()
    print(f"训练集: {x_train.shape}, {y_train.shape}")
    print(f"测试集: {x_test.shape}, {y_test.shape}")
    
    # 构建模型
    print("\n[2/4] 构建极简CNN...")
    model = build_model()
    model.summary()
    
    # 编译
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 训练
    print("\n[3/4] 开始训练...")
    start_time = time.time()
    history = model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=10,
        validation_split=0.1,
        verbose=1
    )
    train_time = time.time() - start_time
    
    # 评估
    print("\n[4/4] 评估模型...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    
    # 输出结果
    print("\n" + "=" * 50)
    print("训练完成！")
    print("=" * 50)
    print(f"测试准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"训练耗时: {train_time:.1f}秒")
    print(f"总参数量: {model.count_params():,}")
    
    # 保存模型
    os.makedirs('./saved_models', exist_ok=True)
    model.save_weights('./saved_models/simple_cnn.weights.h5')
    print("模型权重已保存至 ./saved_models/simple_cnn.weights.h5")
    
    return test_acc

if __name__ == "__main__":
    main()