import os
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

def create_template(image):
    """
    创建无瑕疵模板
    
    Args:
        image: 输入图像
    
    Returns:
        template: 生成的无瑕疵模板
        is_tilted: 是否为倾斜条纹
        angle: 倾斜角度
    """
    # 转换为灰度图像（如果是彩色图像）
    if len(image.shape) == 3:
        # 转换为YUV空间并提取亮度通道
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        gray = yuv[:,:,0]
    else:
        gray = image.copy()
    
    # 检测条纹倾斜角度
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
    
    is_tilted = False
    angle = 0
    
    if lines is not None:
        # 计算主要线条的角度
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle_deg = np.degrees(theta) % 180
            # 垂直线条的角度接近0或180度
            if not (80 <= angle_deg <= 100):  # 非水平线
                angles.append(angle_deg)
        
        if angles:
            # 使用众数作为主要角度
            from scipy import stats
            angle = stats.mode(np.array(angles), keepdims=True)[0][0]
            # 如果角度不接近90度（垂直），则认为是倾斜的
            is_tilted = not (85 <= angle <= 95 or angle <= 5 or angle >= 175)
    
    template = np.zeros_like(image)
    
    if is_tilted:
        # 倾斜条纹：对第一列和最后一列进行双线性插值
        height, width = gray.shape[:2]
        
        if len(image.shape) == 3:
            # 彩色图像
            for c in range(3):  # 对每个通道进行处理
                first_col = image[:, 0, c]
                last_col = image[:, -1, c]
                
                for i in range(width):
                    # 计算插值权重
                    weight = i / (width - 1)
                    # 双线性插值
                    template[:, i, c] = (1 - weight) * first_col + weight * last_col
        else:
            # 灰度图像
            first_col = image[:, 0]
            last_col = image[:, -1]
            
            for i in range(width):
                weight = i / (width - 1)
                template[:, i] = (1 - weight) * first_col + weight * last_col
    else:
        # 垂直条纹：复制第一列
        if len(image.shape) == 3:
            # 彩色图像
            for c in range(3):
                template[:, :, c] = np.tile(image[:, 0, c].reshape(-1, 1), (1, image.shape[1]))
        else:
            # 灰度图像
            template = np.tile(image[:, 0].reshape(-1, 1), (1, image.shape[1]))
    
    return template, is_tilted, angle

def detect_defects(image, template):
    """
    检测图像中的瑕疵
    
    Args:
        image: 原始图像
        template: 无瑕疵模板
    
    Returns:
        diff_img: 差异图
        binary_img: 二值化结果
    """
    # 计算差异图（绝对差）
    diff_img = cv2.absdiff(image, template)
    
    # 转换为灰度图（如果是彩色图像）
    if len(diff_img.shape) == 3:
        diff_gray = cv2.cvtColor(diff_img, cv2.COLOR_BGR2GRAY)
    else:
        diff_gray = diff_img
    
    # 计算阈值（均值 + 3倍标准差）
    mean, std = cv2.meanStdDev(diff_gray)
    threshold = mean[0][0] + 3 * std[0][0]
    
    # 二值化
    _, binary_img = cv2.threshold(diff_gray, threshold, 255, cv2.THRESH_BINARY)
    
    # 也可以使用Otsu自动阈值
    # _, binary_img = cv2.threshold(diff_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return diff_img, binary_img

def process_image(image_path, output_dir):
    """
    处理单张图像并保存结果
    
    Args:
        image_path: 输入图像路径
        output_dir: 输出目录
    
    Returns:
        None
    """
    # 读取图像
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"无法读取图像: {image_path}")
        return
    
    # 创建无瑕疵模板
    template, is_tilted, angle = create_template(image)
    
    # 检测瑕疵
    diff_img, binary_img = detect_defects(image, template)
    
    # 创建可视化结果
    # 将二值图转为三通道以便可视化
    binary_vis = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    
    # 保存处理后的图像
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, filename)
    
    # 创建可视化结果（原图、模板、差异图、二值结果的组合）
    h, w = image.shape[:2]
    result = np.zeros((h, w*4, 3), dtype=np.uint8)
    
    # 确保所有图像都是三通道
    if len(image.shape) == 2:
        image_vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_vis = image.copy()
        
    if len(template.shape) == 2:
        template_vis = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
    else:
        template_vis = template.copy()
        
    if len(diff_img.shape) == 2:
        diff_vis = cv2.cvtColor(diff_img, cv2.COLOR_GRAY2BGR)
    else:
        diff_vis = diff_img.copy()
    
    # 组合图像
    result[:, 0:w] = image_vis
    result[:, w:2*w] = template_vis
    result[:, 2*w:3*w] = diff_vis
    result[:, 3*w:4*w] = binary_vis
    
    # 添加文本标签
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result, "Original", (10, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(result, "Template", (w+10, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(result, "Difference", (2*w+10, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(result, "Binary", (3*w+10, 30), font, 1, (0, 255, 0), 2)
    
    # 添加倾斜信息
    tilt_info = f"Tilted: {is_tilted}, Angle: {angle:.1f}" if is_tilted else "Vertical stripes"
    cv2.putText(result, tilt_info, (10, h-20), font, 0.7, (0, 255, 255), 2)
    
    # 保存结果
    cv2.imwrite(output_path, result)
    
    # 保存二值化后的图像到pre_images文件夹
    processed_dir = os.path.join(os.path.dirname(output_dir), "pre_images")
    os.makedirs(processed_dir, exist_ok=True)
    processed_path = os.path.join(processed_dir, filename)
    
    # 将二值图转换为三通道图像（保持黑色背景）
    binary_rgb = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    
    # 保存二值化后的图像
    cv2.imwrite(processed_path, binary_rgb)

def main():
    # 基础路径
    base_dir = Path("/Users/leo/ultralytics/ADC.v1i.yolov11")
    
    # 处理每个数据集
    for dataset in ["train", "valid", "test"]:
        print(f"处理 {dataset} 数据集...")
        
        # 输入图像目录
        image_dir = base_dir / dataset / "images"
        
        # 输出目录
        output_dir = base_dir / dataset / "visualization"
        os.makedirs(output_dir, exist_ok=True)
        
        # 处理后图像目录
        processed_dir = base_dir / dataset / "pre_images"
        os.makedirs(processed_dir, exist_ok=True)
        
        # 获取所有图像文件
        image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpeg"))
        
        # 处理每张图像
        for image_path in tqdm(image_files, desc=f"处理 {dataset} 图像"):
            process_image(image_path, output_dir)
    
    print("所有图像处理完成！")

if __name__ == "__main__":
    main()
