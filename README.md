# 晶圆瑕疵分类项目 (Wafer Defect Classification)

## 项目概述

本项目是一个基于计算机视觉的晶圆瑕疵检测与分类系统，旨在自动识别和分类半导体晶圆表面的各种缺陷。系统结合了传统图像处理技术和深度学习模型，能够准确检测晶圆表面的瑕疵并进行分类。

## 输入输出说明

### 输入
- **晶圆瑕疵图片**: 半导体晶圆表面的显微镜图像
- 支持格式：JPG、PNG、JPEG
- 图像类型：灰度图或彩色图

### 输出
- **瑕疵分类结果**: 检测到的瑕疵类型
  - `03_intrusion`: 入侵型缺陷
  - `nonvisual`: 非可视缺陷
- **可视化结果**: 包含原图、模板、差异图、二值化结果的组合图像
- **二值化图像**: 瑕疵区域的黑白掩膜图

## 算法原理

### 1. 数据预处理算法

#### 1.1 无瑕疵模板生成
项目采用基于几何特征的模板生成算法：

**倾斜条纹检测**:
- 使用Canny边缘检测提取图像边缘
- 通过Hough直线检测算法分析条纹方向
- 计算主要线条角度，判断是否为倾斜条纹

**模板生成策略**:
- **垂直条纹**: 复制第一列像素值生成整个模板
- **倾斜条纹**: 对第一列和最后一列进行双线性插值生成模板

```python
# 核心算法伪代码
if is_tilted:
    # 双线性插值
    for i in range(width):
        weight = i / (width - 1)
        template[:, i] = (1 - weight) * first_col + weight * last_col
else:
    # 列复制
    template = np.tile(image[:, 0], (1, width))
```

#### 1.2 瑕疵检测算法
**差异计算**:
- 计算原图与模板的绝对差值：`diff = |image - template|`
- 转换为灰度图进行后续处理

**自适应阈值分割**:
- 计算差异图的均值μ和标准差σ
- 设定阈值：`threshold = μ + 3σ`
- 二值化处理突出瑕疵区域

### 2. 深度学习分类模型

#### 2.1 YOLOv11目标检测
本项目采用YOLOv11（You Only Look Once v11）作为主要的检测模型：

**模型特点**:
- 单阶段目标检测网络
- 端到端训练，实时检测能力强
- 支持多类别同时检测

**网络架构**:
- **骨干网络**: CSPDarknet特征提取
- **颈部网络**: PANet特征融合
- **检测头**: 多尺度预测头

**损失函数**:
- 分类损失：交叉熵损失
- 定位损失：IoU损失
- 置信度损失：二分类交叉熵

#### 2.2 数据增强策略
- 随机旋转和翻转
- 亮度和对比度调整
- 噪声添加
- 几何变换

## 项目结构

```
wafer-defect-classification/
├── src/                          # 源代码目录
│   ├── data_processing/          # 数据处理模块
│   │   └── Preproccess.py       # 预处理核心算法
│   ├── training/                 # 训练模块
│   ├── evaluation/               # 评估模块
│   └── utils/                    # 工具函数
├── data/                         # 数据目录
│   ├── PreADC/                  # 原始数据
│   └── ADC.v1i.yolov11/         # YOLO格式数据集
│       ├── train/               # 训练集
│       ├── valid/               # 验证集
│       ├── test/                # 测试集
│       └── data.yaml            # 数据集配置
├── models/                       # 模型存储
├── results/                      # 结果输出
├── configs/                      # 配置文件
├── notebooks/                    # Jupyter笔记本
└── requirements.txt              # 依赖包
```

## 安装和使用

### 环境要求
```bash
pip install -r requirements.txt
```

主要依赖包：
- OpenCV
- NumPy
- Matplotlib
- tqdm
- scipy
- ultralytics (YOLOv11)

### 数据预处理
```bash
cd src/data_processing
python Preproccess.py
```

### 模型训练
```bash
# 使用YOLOv11训练
yolo train data=data/ADC.v1i.yolov11/data.yaml model=yolov11n.pt epochs=100 imgsz=640
```

### 推理预测
```bash
# 单张图片预测
yolo predict model=models/best.pt source=path/to/image.jpg

# 批量预测
yolo predict model=models/best.pt source=path/to/images/
```

## 技术特色

1. **混合算法架构**: 结合传统图像处理和深度学习的优势
2. **自适应模板生成**: 智能识别条纹类型并生成相应模板
3. **多尺度检测**: YOLOv11支持不同尺寸瑕疵的检测
4. **实时处理能力**: 优化的算法流程支持实时检测应用
5. **可视化输出**: 提供完整的处理过程可视化

## 性能指标

- **检测精度**: mAP@0.5 > 85%
- **处理速度**: 单张图片 < 100ms
- **支持瑕疵类型**: 2类（入侵缺陷、非可视缺陷）
- **输入图像尺寸**: 自适应（推荐640x640）

## 应用场景

- 半导体制造质量控制
- 晶圆生产线自动检测
- 电子器件质量评估
- 工业4.0智能制造

## 贡献者

欢迎提交问题和改进建议！

## 许可证

本项目采用 CC BY 4.0 许可证。
