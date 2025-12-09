# scheme_yolo.py
from ultralytics import YOLO

def detect_with_yolo(image_path: str):
    """
    使用 YOLOv8 检测 (需替换为训练过 Figure/Table 的权重)
    """
    # ⚠️ 注意：这里加载的是官方通用权重。
    # 实际使用请加载你自己训练的权重，例如: model = YOLO('path/to/best_figure_table.pt')
    model = YOLO('yolov8n.pt') 
    
    # 预测，imgsz 根据长图大小可能需要调整，或者启用切片推理(sahi)
    results = model.predict(image_path, conf=0.25, save=False)
    
    detections = []
    for result in results:
        for box in result.boxes:
            coords = box.xyxy[0].tolist() # x1, y1, x2, y2
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label_name = model.names[cls_id]
            
            # 格式化输出
            line = f"{label_name} {coords[0]:.2f} {coords[1]:.2f} {coords[2]:.2f} {coords[3]:.2f} {conf:.2f}"
            detections.append(line)

    return "\n".join(detections)