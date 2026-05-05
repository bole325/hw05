问题1：TensorFlow数据shape不匹配
现象：

python
ValueError: Input 0 of layer "conv2d" is incompatible with the layer: 
expected min_ndim=4, found ndim=3. Full shape received: (None, 28, 28)
原因分析：

Conv2D层期望输入格式为 (batch_size, height, width, channels)

MNIST原始数据 load_data() 返回shape (60000, 28, 28)，缺少通道维度

修改方案：

python
# 修改前
x_train, x_test = x_train / 255.0, x_test / 255.0

# 修改后
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
问题2：MNIST下载超时
现象：

text
Exception: URL fetch failure on https://storage.googleapis.com/.../mnist.npz: 
ConnectionResetError(104, 'Connection reset by peer')
原因分析：

国内网络访问Google Cloud Storage不稳定

默认超时时间较短（60秒）

解决方案（3选1）：

设置代理：

python
import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
手动下载：

从 https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat 下载

或使用备用源：pip install python-mnist

使用Torchvision替代（推荐）：

python
from torchvision import datasets
train_data = datasets.MNIST(root='./data', train=True, download=True)
最终采用方案3，同时兼容两种框架。

问题3：PyTorch和TensorFlow输出格式不统一
现象：训练结束后，两模型准确率对比时发现格式差异，手动对齐易出错。

原因分析：

TensorFlow: 标签为one-hot编码 [0,0,1,0,...]

PyTorch: 标签为整数索引 [2, 5, 0, ...]

解决方案：编写统一评估函数

python
# utils.py
def evaluate_model(model, test_loader, framework='pytorch'):
    """统一评估接口"""
    if framework == 'pytorch':
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        return correct / total
    else:  # tensorflow
        # ... TF评估逻辑
问题4：LeNet-5池化层误用MaxPooling
现象：LeNet-5测试准确率只有97.3%，低于预期。

原因分析：

误将经典LeNet-5的平均池化写成了最大池化

虽然现代CNN多用MaxPooling，但LeNet-5的设计是AveragePooling

平均池化保留更多背景信息，对数字识别有帮助

修改方案：

python
# 修改前
self.pool1 = nn.MaxPool2d(2)
self.pool2 = nn.MaxPool2d(2)

# 修改后
self.pool1 = nn.AvgPool2d(2)
self.pool2 = nn.AvgPool2d(2)
修改后准确率提升至98.85%，符合预期。

问题5：中文路径导致文件保存失败
现象：

text
OSError: [Errno 22] Invalid argument: './实验结果/lenet5.pth'
原因分析：

Windows系统下，某些Python库对中文路径支持不佳

特别是涉及文件I/O和模型序列化时

解决方案：

将所有中文目录名改为英文 saved_models/

代码中添加自动创建目录逻辑：

python
os.makedirs('./saved_models', exist_ok=True)
