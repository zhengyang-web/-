# -
  A1: 全连接网络 (20 points)  • 任务：训练一个全连接神经网络分类器，完成图像分类(20 points)   • 数据：MNIST手写数字体识别 或 CIFAR-10  • 要求：前馈、反馈、评估等代码需自己动手用numpy实现（而非使用pytorch）附加题：   – 尝试使用不同的损失函数和正则化方法，观察并分析其对实验结果的影响 (+5 points)  – 尝试使用不同的优化算法，观察并分析其对训练过程和实验结果的影响 (如batch GD, online GD, mini-batch GD, SGD, 或其它的优化算法如Momentum, Adsgrad, Adam, Admax) (+5 points)
# 全连接神经网络实现 - MNIST手写数字识别

本项目实现了一个基于NumPy的全连接神经网络分类器，用于完成MNIST手写数字识别任务。项目包含了基本的前馈、反馈、评估等功能，以及多种优化算法、损失函数和正则化方法的实现，同时提供了直观的可视化功能。

## 项目结构

```
├── main.py                     # 主程序文件，包含神经网络的完整实现
├── extensions.py               # 扩展功能实现
├── demo_extensions.py          # 扩展功能演示脚本
├── demo_correct_classification.py # 正确分类样本演示脚本
├── custom_image_prediction.py  # 自定义图像预测脚本
├── requirements.txt            # 项目依赖库列表
├── run.bat                     # 批处理文件，提供菜单式操作界面
├── README.md                   # 项目说明文档
```

## 功能特点

### 基本功能
- 全连接神经网络的前向传播和反向传播实现
- 交叉熵损失函数和准确率计算
- 小批量梯度下降训练方法
- 训练过程可视化（损失和准确率曲线）
- 错误分类样本的可视化
- 正确分类样本的可视化

### 附加功能
- **不同优化算法**：SGD、Momentum、Adam
- **不同损失函数**：交叉熵、均方误差(MSE)
- **不同正则化方法**：L1正则化、L2正则化
- **图形化菜单界面**：通过run.bat提供直观的操作方式
- **自定义图像预测**：支持上传自己的手写数字图像进行识别

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

最简单的方式是通过运行批处理文件来使用本项目：

```bash
run.bat
```

这将显示一个菜单界面，您可以通过输入数字选择要运行的功能：

```
===== 全连接神经网络程序菜单 =====
1. 运行主程序 (MNIST手写数字识别)
2. 运行扩展功能演示
3. 运行正确分类演示
4. 演示所有程序

请选择要运行的程序: 
```

您也可以直接运行各个Python文件：

```bash
# 运行主程序
python main.py

# 运行扩展功能演示
python demo_extensions.py

# 运行正确分类演示
python demo_correct_classification.py
```

程序将自动下载MNIST数据集，进行预处理，然后训练神经网络模型，最后评估模型性能并生成可视化结果。

### 自定义图像预测
您可以使用`custom_image_prediction.py`脚本上传并识别自己的手写数字图像：

```bash
# 运行自定义图像预测脚本
python custom_image_prediction.py
```

运行后，程序将提示您输入图像路径。您可以：
1. 输入自己的手写数字图像路径（支持jpg、png等常见格式）
2. 输入'q'退出程序

程序会自动预处理图像、加载训练好的模型，并显示预测结果。

## 神经网络设计

### 基本模型结构
- 输入层：784个神经元（对应28×28的图像）
- 隐藏层：默认两个隐藏层，分别有128和64个神经元
- 输出层：10个神经元（对应0-9的数字分类）
- 激活函数：隐藏层使用ReLU，输出层使用Softmax

### 超参数设置
- 学习率：0.01
- 训练轮数：20
- 批量大小：64
- 权重初始化：使用He初始化方法

## 自定义图像预测的技术细节

### 图像预处理流程
当使用`custom_image_prediction.py`处理自定义图像时，程序会执行以下预处理步骤：
1. 读取图像并转换为灰度图
2. 调整图像大小为28×28像素（与MNIST数据集一致）
3. 反转图像颜色（使数字为白色，背景为黑色）
4. 对图像进行二值化处理
5. 将图像数据展平为784维向量
6. 归一化像素值到0-1范围

### 模型参数保存与加载

项目使用pickle模块保存和加载训练好的模型参数：

```python
# 保存模型参数（已在main.py中实现自动保存）
model.save_model_parameters('model_weights.pkl')

# 加载模型参数（custom_image_prediction.py会自动加载）
model.load_model_parameters('model_weights.pkl')
```

当您运行`main.py`训练模型时，程序会自动将训练好的参数保存到`model_weights.pkl`文件中。`custom_image_prediction.py`会自动尝试加载这个文件，如果文件不存在，程序会提供训练新模型的选项。

## 扩展功能使用

在`main.py`文件中，可以通过切换不同的神经网络类来测试各种功能：

```python
# 基本神经网络
model = NeuralNetwork(input_size, hidden_sizes, output_size, learning_rate)

# 使用不同的优化算法（可选：'sgd', 'momentum', 'adam'）
model = NeuralNetworkWithOptimizers(input_size, hidden_sizes, output_size, learning_rate, optimizer='adam')

# 使用不同的损失函数（可选：'cross_entropy', 'mse'）
model = NeuralNetworkWithDifferentLosses(input_size, hidden_sizes, output_size, learning_rate, loss_function='cross_entropy')

# 使用不同的正则化方法（可选：'none', 'l1', 'l2'）
model = NeuralNetworkWithRegularization(input_size, hidden_sizes, output_size, learning_rate, regularization='l2', lambda_reg=0.001)
```

## 实验结果

运行主程序后，将生成以下输出和文件：
1. 训练过程中的损失和准确率信息
2. 测试集上的最终性能指标
3. 训练和验证的损失和准确率曲线图（保存为`training_results.png`）
4. 错误分类的样本图像（保存为`misclassified_samples.png`）
5. 正确分类的样本图像（保存为`correctly_classified_samples.png`）

运行扩展功能演示后，还将生成以下文件：
1. 不同激活函数性能比较图（`activation_functions_comparison.png`）
2. 混淆矩阵可视化（`confusion_matrix.png`）
3. 网络权重可视化（`weights_visualization.png`）
4. 早停策略和学习率调度器结果（`early_stopping_lr_scheduler_results.png`）
5. 学习率变化曲线（`learning_rate.png`）

## 各功能实现说明

### 不同优化算法对比
- **SGD（随机梯度下降）**：基本的梯度下降方法，每次使用一个小批量数据计算梯度并更新参数
- **Momentum（动量法）**：在SGD的基础上加入动量项，加速收敛并减少震荡
- **Adam（自适应矩估计）**：结合了Momentum和RMSprop的优点，自适应调整每个参数的学习率

### 不同损失函数对比
- **交叉熵损失**：适用于分类问题的标准损失函数，对错误分类的惩罚更大
- **均方误差损失**：回归问题中常用的损失函数，在分类问题中可能收敛较慢

### 不同正则化方法对比
- **L1正则化**：通过惩罚权重的绝对值之和，鼓励稀疏权重
- **L2正则化**：通过惩罚权重的平方和，抑制权重过大，防止过拟合

## 注意事项
1. 运行程序前请确保已安装所有依赖库
2. 首次运行时需要下载MNIST数据集，请确保网络连接正常
3. 可以根据需要调整超参数以获得更好的性能
4. 可视化结果将保存在当前工作目录下
5. 使用自定义图像预测时的建议：
   - 为获得最佳识别效果，请确保数字居中且占据图像大部分区域
   - 尽量使用清晰的黑色数字和白色背景
   - 输入图像路径时，无需添加引号，程序会自动处理

## 扩展功能
项目新增了`extensions.py`文件，包含以下扩展功能：

### 1. 早停策略（EarlyStopping）
- 当验证损失不再改善时自动停止训练，防止过拟合
- 可设置耐心值（patience）和最小改进幅度（min_delta）

### 2. 学习率调度器（LearningRateScheduler）
- 支持多种学习率衰减策略：
  - 阶梯式衰减（step）
  - 指数衰减（exponential）
  - 线性衰减（linear）
  - 余弦退火衰减（cosine）

### 3. 更多激活函数
- 在`NeuralNetworkWithExtensions`类中支持以下激活函数：
  - ReLU
  - Sigmoid
  - Tanh
  - LeakyReLU
  - ELU

### 4. 模型保存和加载
- 支持将训练好的模型保存到文件和从文件加载模型
- 使用pickle实现序列化和反序列化

### 5. 增强的可视化功能
- 混淆矩阵可视化
- 学习率变化曲线
- 网络权重可视化

### 6. CIFAR-10数据集支持
- 提供了加载和预处理CIFAR-10数据集的功能

## 使用扩展功能

运行扩展功能示例：

```bash
python extensions.py
```

或者在您自己的代码中导入并使用：

```python
from extensions import NeuralNetworkWithExtensions, EarlyStopping, LearningRateScheduler, plot_confusion_matrix

# 创建带早停和学习率调度的模型
model = NeuralNetworkWithExtensions(input_size=784, hidden_sizes=[128, 64], output_size=10, learning_rate=0.01)
early_stopping = EarlyStopping(patience=5, min_delta=0.001)
lr_scheduler = LearningRateScheduler(initial_lr=0.01, decay_type='exponential', decay_rate=0.95)

# 使用LeakyReLU激活函数、早停和学习率调度训练模型
train_losses, val_losses, train_accuracies, val_accuracies, learning_rates = model.train(
    X_train, y_train, X_val, y_val, epochs=30, batch_size=64,
    activation='leaky_relu', early_stopping=early_stopping, lr_scheduler=lr_scheduler
)

# 保存模型
model.save_model('mnist_model.pkl')

# 加载模型
loaded_model = NeuralNetworkWithExtensions.load_model('mnist_model.pkl')

# 绘制混淆矩阵
y_pred = model.forward_propagation(X_test, activation='leaky_relu')
plot_confusion_matrix(y_test, y_pred)```

## 扩展建议
1. 尝试不同的网络结构（层数、神经元数量）
2. 实现批量归一化（Batch Normalization）以加速训练
3. 添加Dropout层以进一步防止过拟合
4. 探索更多的优化算法
5. 尝试实现卷积神经网络以提高图像分类性能
6. 扩展到其他数据集（如Fashion MNIST、CIFAR-10等）
7. 实现更高级的可视化功能，如特征图可视化
8. 添加模型集成方法以进一步提高性能

## 已知问题和限制
1. 当前的演示脚本使用了模拟预测结果，仅用于展示可视化效果
2. 完整训练可能需要较长时间，具体取决于硬件性能
3. 模型在复杂或模糊的手写数字样本上可能出现分类错误


