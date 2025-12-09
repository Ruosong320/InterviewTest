import os
from ultralytics import YOLO
from huggingface_hub import hf_hub_download 
from PIL import Image

# 代理设置 (保持不变，确保下载能通过代理)
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:1080'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:1080'

# 模型ID和文件名
MODEL_ID = 'foduucom/table-detection-and-extraction'
WEIGHTS_FILENAME = 'best.pt'

# 尝试从本地缓存获取模型文件，如果不存在则报错
try:
    # 设置 local_files_only=True 将只从本地缓存加载，不会下载
    local_weights_path = hf_hub_download(
        repo_id=MODEL_ID, 
        filename=WEIGHTS_FILENAME,
        local_files_only=True  # 只从本地缓存加载
    )
    print(f"从本地缓存加载模型权重: {local_weights_path}")
except Exception as e:
    print(f"无法从缓存中找到模型文件，请确保已下载。错误: {e}")
    exit(1)

# 加载YOLO模型
YOLO_MODEL = YOLO(local_weights_path)
print("YOLO模型加载成功。")

def detect_with_yolo(image_path: str):
    """
    使用 foduucom/table-detection-and-extraction 模型检测图片中的表格。
    """
    if YOLO_MODEL is None:
        return "YOLO 模型未初始化成功，无法执行检测。"
        
    # 执行推理
    results = YOLO_MODEL.predict(source=image_path, conf=0.25, verbose=False) 

    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = box.conf[0].item()
            class_id = int(box.cls[0])
            class_name = result.names[class_id]

            # 格式化输出
            line = f"{class_name} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {confidence:.2f}"
            detections.append(line)

    return "\n".join(detections) if detections else "未检测到表格"
