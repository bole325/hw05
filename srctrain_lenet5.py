#!/usr/bin/env python3
"""
任务二：LeNet-5训练与评估脚本
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import os
import sys

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lenet5 import LeNet5

def setup_device():
    """设置计算设备"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if device.type == 'cuda':
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    return device

def load_data(batch_size=128):
    """加载MNIST数据集"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST均值和标准差
    ])
    
    train_dataset = datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    test_dataset = datasets.MNIST(
        root='./data', 
        train=False, 
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, test_loader

def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(train_loader)
    return avg_loss, accuracy

def evaluate(model, test_loader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(test_loader)
    return avg_loss, accuracy

def main():
    print("=" * 50)
    print("任务二：LeNet-5训练与评估")
    print("=" * 50)
    
    # 超参数
    BATCH_SIZE = 128
    EPOCHS = 10
    LEARNING_RATE = 0.001
    
    # 设置设备
    device = setup_device()
    
    # 加载数据
    print("\n[1/4] 加载MNIST数据...")
    train_loader, test_loader = load_data(BATCH_SIZE)
    print(f"训练集批次: {len(train_loader)}")
    print(f"测试集批次: {len(test_loader)}")
    
    # 创建模型
    print("\n[2/4] 构建LeNet-5模型...")
    model = LeNet5().to(device)
    print(f"参数量: {model.count_parameters():,}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 训练
    print("\n[3/4] 开始训练...")
    start_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        print(f"Epoch {epoch:2d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
    
    train_time = time.time() - start_time
    
    # 最终评估
    print("\n[4/4] 最终评估...")
    final_loss, final_acc = evaluate(model, test_loader, criterion, device)
    
    # 输出结果
    print("\n" + "=" * 50)
    print("训练完成！")
    print("=" * 50)
    print(f"测试准确率: {final_acc:.2f}% ({final_acc/100:.4f})")
    print(f"训练耗时: {train_time:.1f}秒")
    print(f"总参数量: {model.count_parameters():,}")
    
    # 保存模型
    os.makedirs('./saved_models', exist_ok=True)
    torch.save(model.state_dict(), './saved_models/lenet5.pth')
    print("模型权重已保存至 ./saved_models/lenet5.pth")
    
    return final_acc

if __name__ == "__main__":
    main()