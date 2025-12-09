import os
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from typing import List, Tuple, Dict, Any
import uuid

# 假设这些模块已正确定义并能导入
from Task2_Table_hf import detect_with_hf
from Task2_Table_yolo import detect_with_yolo
from Task2_Figure import detect_with_pic


# --- 全局配置 ---
TILING_HEIGHT_THRESHOLD = 400
HF_CONTRAST_FACTOR = 1.5
NMS_IOU_THRESHOLD = 0.5


# --- 辅助函数：背景色采样 ---

def _get_robust_background_color(image: Image.Image, margin: int = 50) -> Tuple[int, int, int]:
    try:
        img_np = np.array(image.convert("RGB"))
        h, w, _ = img_np.shape
        
        sample_points = [
            (margin, margin), (h - margin, margin), (margin, w - margin), (h - margin, w - margin),
            (h // 2, margin), (h // 2, w - margin), (margin, w // 2), (h - margin, w // 2),
        ]
        
        colors = [img_np[y, x] for y, x in sample_points if 0 <= y < h and 0 <= x < w]

        if not colors:
            return (255, 255, 255)

        return tuple(np.mean(colors, axis=0).astype(int))
    except Exception:
        return (255, 255, 255)


# --- 辅助函数：垂直切分 ---

def _find_vertical_splits(image: Image.Image, bg_color: Tuple[int, int, int], threshold: int = 30) -> List[Tuple[int, int]]:
    img_np = np.array(image.convert("RGB"))
    height, _ , _ = img_np.shape
    
    is_bg = np.all(np.abs(img_np - bg_color) < threshold, axis=2)
    is_split = np.all(is_bg, axis=1)
    
    blocks = []
    in_content = False
    start_y = 0
    
    for y in range(height):
        if not is_split[y]:
            if not in_content:
                start_y = y
                in_content = True
        elif in_content:
            blocks.append((start_y, min(y + 10, height)))
            in_content = False
    
    if in_content:
        blocks.append((start_y, height))

    return blocks


# --- 辅助函数：水平裁剪范围计算 ---

def _get_content_width_range(image: Image.Image, bg_color: Tuple[int, int, int], threshold: int = 10, margin: int = 100) -> Tuple[int, int]:
    img_np = np.array(image.convert("RGB"))
    _, width, _ = img_np.shape

    is_bg = np.all(np.abs(img_np - bg_color) < threshold, axis=2)
    is_content_col = ~np.all(is_bg, axis=0)

    content_indices = np.where(is_content_col)[0]
    
    if len(content_indices) == 0:
        return 0, width

    x_start = content_indices[0]
    x_end = content_indices[-1] + 1
    
    x_min = max(0, x_start - margin)
    x_max = min(width, x_end + margin)
    
    return x_min, x_max


# --- 辅助函数：图片预处理 ---

def _preprocess_image_for_detection(image: Image.Image, temp_path: str, height: int, scheme: str) -> str:
    if height > TILING_HEIGHT_THRESHOLD:
        processed_image = image.convert("RGB") 
        
        if scheme in ['yolo', 'hf']: 
            print(f"[PREPROCESS] 对 {scheme.upper()} 切块 (H={height}) 应用对比度增强({HF_CONTRAST_FACTOR}) + 锐化...")

            enhancer = ImageEnhance.Contrast(processed_image)
            processed_image = enhancer.enhance(HF_CONTRAST_FACTOR)
            processed_image = processed_image.filter(ImageFilter.SHARPEN)
        
        processed_image.save(temp_path)
        processed_image.close()
        return temp_path
    else:
        image.save(temp_path)
        return temp_path


# --- 辅助函数：计算 IoU ---

def _calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    
    x_min_inter = np.maximum(box1[0], box2[0])
    y_min_inter = np.maximum(box1[1], box2[1])
    x_max_inter = np.minimum(box1[2], box2[2])
    y_max_inter = np.minimum(box1[3], box2[3])

    inter_width = np.maximum(0.0, x_max_inter - x_min_inter)
    inter_height = np.maximum(0.0, y_max_inter - y_min_inter)
    intersection_area = inter_width * inter_height
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - intersection_area
    
    if union_area == 0:
        return 0.0
        
    return intersection_area / union_area


# --- 辅助函数：全局 NMS (使用 IoU) ---

def _global_nms(detections: List[Dict], iou_threshold: float) -> List[Dict]:
    if not detections:
        return []

    detections.sort(key=lambda x: x['score'], reverse=True)
    
    boxes = np.array([d['box'] for d in detections], dtype=np.float32)
    keep = []
    
    while len(detections) > 0:
        current_detection = detections.pop(0)
        current_box = boxes[0]
        boxes = boxes[1:]
        
        keep.append(current_detection)

        if len(boxes) == 0:
            break
            
        remove_indices = []
        for j in range(len(detections)):
            if _calculate_iou(current_box, boxes[j]) > iou_threshold: 
                remove_indices.append(j)
        
        # 移除被抑制的框及其对应信息
        detections = [d for i, d in enumerate(detections) if i not in remove_indices]
        boxes = np.delete(boxes, remove_indices, axis=0)

    return keep


# --- 核心集成函数：切分、合并、检测与坐标累加 ---

def detect_with_tiling(image_path: str, temp_dir: str, scheme: str) -> str:
    """
    通过切块、合并、局部裁剪和全局 NMS 的方式，使用指定的检测方案对长图进行目标检测。
    
    Args:
        image_path: 原始图片路径。
        temp_dir: 临时文件存放目录。
        scheme: 检测方案 ('hf', 'yolo', 'pic')。
        
    Returns:
        str: 格式化后的检测结果字符串 (Label x1 y1 x2 y2 Score)。
    """
    
    # ----------------------------------------------------
    # 关键修改：添加 'pic' 方案分支
    # ----------------------------------------------------
    if scheme == 'hf':
        detector_func = detect_with_hf
        # 兼容性：hf 和 yolo 方案的输出中可能包含 "未检测到表格"
        error_msg = "未检测到表格" 
    elif scheme == 'yolo':
        detector_func = detect_with_yolo
        error_msg = "未检测到表格"
    elif scheme == 'pic': # <--- 新增的插图检测方案
        detector_func = detect_with_pic
        error_msg = "未检测到插图"
    else:
        raise ValueError(f"不支持的方案: {scheme}。")
    # ----------------------------------------------------
        
    all_detections: List[Dict] = []
    
    with Image.open(image_path) as original_image:
        
        bg_color = _get_robust_background_color(original_image)
        split_coords = _find_vertical_splits(original_image, bg_color)
        
        if not split_coords:
            split_coords = [(0, original_image.size[1])]

        i = 0
        total_splits = len(split_coords)
        
        while i < total_splits:
            y_start_curr, y_end_curr = split_coords[i]
            y_start_combined = y_start_curr
            y_end_combined = y_end_curr
            current_height = y_end_curr - y_start_curr
            
            # 1. 相邻块合并逻辑
            if current_height > TILING_HEIGHT_THRESHOLD:
                if i > 0:
                    y_start_combined = split_coords[i-1][0]
                
                if i < total_splits - 1:
                    y_end_combined = split_coords[i+1][1]
                    i += 1 # 跳过下一个，因为它已被合并
            
            # 2. 垂直裁剪
            with original_image.crop((0, y_start_combined, original_image.size[0], y_end_combined)) as full_width_tile:
                
                # 3. 计算并执行本地水平裁剪
                x_start_local_crop, x_end_local_crop = _get_content_width_range(full_width_tile, bg_color)
                local_width_offset = x_start_local_crop 
                
                with full_width_tile.crop((
                    x_start_local_crop, 
                    0, 
                    x_end_local_crop, 
                    y_end_combined - y_start_combined 
                )) as cropped_image:
                    
                    tile_height = cropped_image.size[1]
                    tile_width = cropped_image.size[0]
                    
                    temp_tile_path = os.path.join(temp_dir, f"tile_{i}_{uuid.uuid4()}.png")

                    # 4. 预处理
                    processed_tile_path = _preprocess_image_for_detection(
                        cropped_image, 
                        temp_tile_path, 
                        tile_height, 
                        scheme
                    )
                    
                    print(f"[TILING] 正在使用 {scheme.upper()} 处理块 (Y: {y_start_combined} to {y_end_combined}, W: {tile_width}, X_offset: {local_width_offset})...")

                    # 5. 运行检测
                    results_str = detector_func(processed_tile_path)
                    
                    # 6. 清理临时文件
                    try:
                        os.remove(processed_tile_path)
                    except Exception as e:
                        print(f"[ERROR] 清理临时文件失败 ({processed_tile_path}): {e}")

            # 7. 坐标累加与转换
            # 调整了条件，使用之前定义的 error_msg 变量
            for line in results_str.split('\n'):
                if line and error_msg not in line and "模型未初始化成功" not in line: 
                    try:
                        parts = line.split()
                        if len(parts) < 6: continue
                            
                        label = parts[0]
                        tx1, ty1, tx2, ty2, score = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                        
                        ox1 = tx1 + local_width_offset  
                        oy1 = ty1 + y_start_combined  
                        ox2 = tx2 + local_width_offset
                        oy2 = ty2 + y_start_combined
                        
                        all_detections.append({
                            "label": label, 
                            "box": [ox1, oy1, ox2, oy2], 
                            "score": score
                        })
                    except Exception as e:
                        print(f"[WARN] 格式错误，跳过: {line} 错误: {e}")
            
            i += 1 

        # 8. 后处理：全局 NMS
        print(f"[NMS] 在 {len(all_detections)} 个原始检测结果上应用全局 NMS (IoU={NMS_IOU_THRESHOLD})...")
        final_detections = _global_nms(all_detections, iou_threshold=NMS_IOU_THRESHOLD)
        print(f"[NMS] 最终保留 {len(final_detections)} 个去重后的结果。")

        # 9. 格式化最终输出
        output_lines = []
        for d in final_detections:
            box = d['box']
            output_lines.append(
                f"{d['label']} {box[0]:.2f} {box[1]:.2f} {box[2]:.2f} {box[3]:.2f} {d['score']:.2f}"
            )

        return "\n".join(output_lines)