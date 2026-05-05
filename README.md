#hw05/
├── README.md                       # 项目说明与运行指南
├── requirements.txt                # Python依赖
├── report.md                       # 实验报告
├── debug_notes.md                  # 调试记录
│
├── src/
│   ├── simple_cnn.py              # 任务一：极简CNN（基于Keras）
│   ├── lenet5.py                  # 任务二：LeNet-5模型定义
│   ├── train_lenet5.py            # 任务二：LeNet-5训练脚本
│   └── utils.py                   # 工具函数（数据加载等）
│
├── notebooks/
│   └── mnist_cnn_comparison.ipynb # Jupyter综合版本
│
└── saved_models/                   # 训练后的模型保存目录（自动创建）
    ├── simple_cnn.weights.h5
    └── lenet5.pth
