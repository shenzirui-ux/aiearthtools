# 数据集格式转换工具

## 项目介绍
本工具旨在将本地数据集转换为可以上传至 AI Earth 平台的样本格式。该工具通过 PyQt5 提供用户友好的图形界面，支持图像和 mask 数据的批量处理、格式转换、尺寸调整、数据集压缩等功能。

## 功能特性
- 创建 AI Earth 数据集目录结构
- 图像和 mask 文件的批量选择与预览
- 图像和 mask 数据的尺寸调整与格式转换
- 自动生成 `lst.txt` 文件，满足 AI Earth 平台的要求
- 数据集的快速压缩与打包

## 环境要求
- Python 3.8 及以上
- 依赖库
  - PyQt5
  - Pillow (PIL)
  - OpenCV (cv2)

使用以下命令安装所需的依赖库：
```bash
pip install PyQt5 pillow opencv-python
```

## 使用说明

### 1. 启动程序
执行以下命令启动工具：
```bash
python main.py
```

### 2. 创建数据集目录
- 点击 "创建数据集文件夹" 按钮
- 选择数据集的存储路径，工具会自动创建以下目录结构：
```
TARGET_EXTRACTION/
├── annotations/  # 存放mask数据
├── images/        # 存放图像数据
└── lst/           # 存放 lst.txt 文件
```

### 3. 处理 mask 数据
- 点击 "选取mask文件夹"，选择包含 mask 文件的目录
- 设置 mask 数据的目标宽度、高度及输出格式
- 点击 "保存mask文件"，转换后的 mask 数据将保存在 `annotations/` 目录

### 4. 处理图像数据
- 点击 "选取图像文件夹"，选择包含图像文件的目录
- 设置图像的目标宽度、高度及输出格式
- 点击 "保存图像文件"，转换后的图像数据将保存在 `images/` 目录

### 5. 生成 lst.txt 文件
- 点击 "生成lst.txt" 按钮
- 程序会自动根据 `images/` 和 `annotations/` 目录下的文件名生成 `lst.txt` 文件

### 6. 压缩数据集
- 点击 "压缩数据集" 按钮
- 工具会将整个 `TARGET_EXTRACTION/` 文件夹压缩成 ZIP 文件，便于上传至 AI Earth 平台

## 注意事项
- 确保图像与 mask 数据具有相同的命名规范，以确保 `lst.txt` 文件正确生成
- 处理过程中如遇到错误信息，请根据提示检查文件路径、格式或参数设置

## 开发者信息
如有问题或建议，欢迎联系开发者。
Email：shenzirui0314@gamil.com
![image](https://github.com/user-attachments/assets/b7b95173-4dbf-4726-99d6-72a5205f655f)

