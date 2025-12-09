import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:1080'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:1080'

# scheme_hf.py
from transformers import DetrImageProcessor, TableTransformerForObjectDetection
from PIL import Image
import torch
import warnings

# 临时忽略 max_size 弃用警告
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.models.detr.image_processing_detr")


def detect_with_hf(image_path: str):
    """
    使用 Hugging Face Table Transformer 检测表格，并返回所有检测到的图表结构坐标。
    """
    # 加载模型 (现在应已从本地缓存加载)
    model_name = "microsoft/table-transformer-detection"
    processor = DetrImageProcessor.from_pretrained(model_name)
    model = TableTransformerForObjectDetection.from_pretrained(model_name)

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    # 推理
    with torch.no_grad():
        outputs = model(**inputs)

    # 后处理：转换坐标
    target_sizes = torch.tensor([image.size[::-1]])
    
    results = processor.post_process_object_detection(outputs, threshold=0.35, target_sizes=target_sizes)[0]

    detections = []
    
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]

        label_name = model.config.id2label[label.item()]
        
        # 格式: label_name x_min y_min x_max y_max score
        detections.append(f"{label_name} {box[0]} {box[1]} {box[2]} {box[3]} {round(score.item(), 2)}")
    
    return "\n".join(detections)