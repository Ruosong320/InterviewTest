import os
import torch
import logging
import warnings
from typing import List, Dict
from PIL import Image
# 核心修正：导入正确的模型和处理器类
from transformers import AutoImageProcessor, DetrForSegmentation

# 设置代理
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:1080'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:1080'

# 抑制日志和警告
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# --- 全局配置 ---
MODEL_ID = "cmarkea/detr-layout-detection"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# 目标类别：已修正为小写 "picture"
TARGET_CLASSES = ["Picture"]
CONFIDENCE_THRESHOLD = 0.5

# --- 模型初始化 ---
PROCESSOR = None
MODEL = None

try:
    # 1. 修正: 使用 AutoImageProcessor
    PROCESSOR = AutoImageProcessor.from_pretrained(MODEL_ID)

    # 2. 修正: 使用 DetrForSegmentation 加载模型
    MODEL = DetrForSegmentation.from_pretrained(MODEL_ID).to(DEVICE)
    print(f"DETR 模型加载成功并部署到 {DEVICE}。")

    # 调试信息：确认模型支持的标签
    available_labels = list(MODEL.config.id2label.values())
    print(f"模型支持的标签有: {', '.join(available_labels)}")

except Exception as e:
    print(f"无法加载 DETR 模型或处理器。错误: {e}")


# --- 核心检测函数 ---

def detect_with_pic(image_path: str) -> str:
    """
    使用 cmarkea/detr-layout-detection 模型检测图片中的布局元素，
    并输出 'picture' 类别的边界框。
    """
    if MODEL is None or PROCESSOR is None:
        return "DETR 模型未初始化成功，无法执行检测。"

    try:
        # 1. 加载和预处理图片
        image = Image.open(image_path).convert("RGB")
        inputs = PROCESSOR(images=image, return_tensors="pt").to(DEVICE)

        # 2. 执行推理
        with torch.no_grad():
            outputs = MODEL(**inputs)

        # 3. 后处理：只提取边界框
        target_sizes = torch.tensor([image.size[::-1]]).to(DEVICE)

        # 使用 post_process_object_detection 获取 Bounding Box
        results = PROCESSOR.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=CONFIDENCE_THRESHOLD
        )[0]

    except Exception as e:
        return f"DETR 推理过程中发生错误: {e}"

    # 4. 格式化和过滤结果
    detections: List[str] = []

    for score, label_id, box in zip(results["scores"], results["labels"], results["boxes"]):

        label = MODEL.config.id2label[label_id.item()]

        # 只保留目标类别 ("picture")
        if label not in TARGET_CLASSES:
            continue

        # 提取和格式化坐标
        x1, y1, x2, y2 = box.cpu().tolist()

        line = f"{label} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {score.item():.4f}"
        detections.append(line)

    return "\n".join(detections) if detections else "未检测到插图"

# --- 测试调用 ---
# test = r"D:\Interview\InterviewTest\T2\testNresult\screenshot.png"
# print(detect_with_pic(test))