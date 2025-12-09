# visualize_boxes.py (支持多文件和颜色区分)

import os
from PIL import Image, ImageDraw
from typing import List, Tuple, Dict, Any

# --- 配置 ---
IMAGE_PATH = r"D:\Interview\InterviewTest\T2\testNresult\screenshot.png"
TABLE_COORDINATES_PATH = r"D:\Interview\InterviewTest\T2\testNresult\coordinates.txt" 
PIC_COORDINATES_PATH = r"D:\Interview\InterviewTest\T2\testNresult\pic_coordinates.txt" 
OUTPUT_PATH = r"D:\Interview\InterviewTest\T2\testNresult\screenshot_with_boxes.png" 

# 边界框样式配置 (根据标签来区分颜色和粗细)
BOX_STYLES = {
    # 针对表格/文本检测结果（通常来自 coordinates.txt 的标签）
    "table": {"color": "red", "width": 5},
    "text": {"color": "red", "width": 5},
    "table structure": {"color": "red", "width": 5},
    # 针对插图检测结果（来自 pic_coordinates.txt 的标签）
    "picture": {"color": "blue", "width": 5},
    "figure": {"color": "blue", "width": 5},
    # 默认颜色（如果标签未定义）
    "default": {"color": "orange", "width": 3}
}

# --- 辅助函数：解析坐标文件（返回标签、坐标和分数） ---

def parse_coordinates(file_path: str) -> List[Dict[str, Any]]:
    """
    从坐标文档中解析出包含标签和 (x1, y1, x2, y2) 的字典列表。
    预期格式: label x1 y1 x2 y2 score
    """
    detections = []
    if not os.path.exists(file_path):
        print(f"警告：未找到文件: {file_path}，跳过解析。")
        return detections
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        # 标签在第 1 位
                        label = parts[0].lower() 
                        # 坐标在第 2 到第 5 位
                        x1 = float(parts[1])
                        y1 = float(parts[2])
                        x2 = float(parts[3])
                        y2 = float(parts[4])
                        # 分数在第 6 位 (如果存在，否则默认为 1.0)
                        score = float(parts[5]) if len(parts) > 5 else 1.0
                        
                        # 确保坐标顺序正确 (x_min < x_max, y_min < y_max)
                        x_min = min(x1, x2)
                        y_min = min(y1, y2)
                        x_max = max(x1, x2)
                        y_max = max(y1, y2)
                        
                        detections.append({
                            "label": label, 
                            "box": (x_min, y_min, x_max, y_max), 
                            "score": score
                        })
                    except ValueError:
                        print(f"警告：跳过格式错误的坐标行（非数字）： {line.strip()}")
                elif line.strip():
                    print(f"警告：跳过不完整的坐标行: {line.strip()}")
    except Exception as e:
        print(f"读取坐标文件 ({file_path}) 时发生错误: {e}")
        
    return detections

# --- 核心函数：绘制边界框（现在接收完整的检测字典） ---

def visualize_and_save(image_path: str, detections: List[Dict[str, Any]], output_path: str, styles: Dict[str, Dict]):
    """
    在给定图片上绘制边界框并保存结果，根据标签使用不同的颜色。
    """
    if not detections:
        print("未检测到有效坐标，无法绘制。")
        return

    try:
        # 1. 打开图片
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        
        # 2. 绘制每个边界框
        for i, det in enumerate(detections):
            label = det['label']
            x1, y1, x2, y2 = det['box']
            
            # 查找样式，如果标签不在配置中，则使用默认样式
            style = styles.get(label, styles['default']) 
            box_color = style["color"]
            box_width = style["width"]
            
            # 使用多次绘制偏移矩形来模拟粗线。
            for offset in range(-box_width // 2, box_width // 2 + 1):
                # 偏移矩形以模拟粗线
                draw.rectangle(
                    (x1 + offset, y1 + offset, x2 + offset, y2 + offset), 
                    outline=box_color
                )
            
            print(f"绘制第 {i+1} 个框 ({label.upper()}, Score={det['score']:.4f}, Color={box_color}): ({int(x1)}, {int(y1)}) 到 ({int(x2)}, {int(y2)})")

        # 3. 保存图片
        img.save(output_path)
        print(f"\n绘制完成。带框图片已保存到: {output_path}")

    except FileNotFoundError:
        print(f"错误：未找到图片文件: {image_path}")
    except Exception as e:
        print(f"绘制或保存图片时发生错误: {e}")

# --- 主执行块 ---

if __name__ == "__main__":
    print("\n--- 启动坐标可视化验证 (表格/文本 RED, 插图 BLUE) ---")
    
    # 1. 解析表格/文本坐标
    table_detections = parse_coordinates(TABLE_COORDINATES_PATH)
    
    # 2. 解析插图坐标
    pic_detections = parse_coordinates(PIC_COORDINATES_PATH)
    
    # 3. 合并所有检测结果
    all_detections = table_detections + pic_detections
    
    if all_detections:
        # 4. 绘制并保存
        visualize_and_save(IMAGE_PATH, all_detections, OUTPUT_PATH, BOX_STYLES)
    else:
        print("无法进行可视化验证，因为没有从任何文档中解析出有效坐标。")