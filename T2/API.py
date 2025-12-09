import os
import shutil
import uuid
import zipfile
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

# 导入必要的模块
# 假设此文件存在
from webLongShot import capture_full_page 
# 切分集成方案，现在需要支持 'hf', 'yolo', 和 'pic'
from image_splitter import detect_with_tiling 

app = FastAPI(
    title="长图表格/插图检测服务",
    description="运行 tiling_hf 或 tiling_yolo 时，将同时运行 tiling_pic。",
    version="1.0.0"
)

# 定义请求体
class CaptureRequest(BaseModel):
    url: str
    scheme: str = "yolo"  # 简化 scheme 名称，与 detect_with_tiling 兼容

# 临时文件清理任务 (保持不变)
def remove_file(path: str):
    try:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
    except Exception as e:
        print(f"Error removing file {path}: {e}")

@app.post("/api/v1/capture_and_locate")
async def capture_and_locate(req: CaptureRequest, background_tasks: BackgroundTasks):
    # 1. 创建唯一的工作目录和文件路径
    task_id = str(uuid.uuid4())
    work_dir = os.path.join(".", f"temp_{task_id}")
    os.makedirs(work_dir, exist_ok=True)
    
    img_path = os.path.join(work_dir, "screenshot.png")
    # 表格/文本结果
    table_txt_path = os.path.join(work_dir, "coordinates.txt") 
    # 插图结果
    pic_txt_path = os.path.join(work_dir, "pic_coordinates.txt") 
    zip_path = os.path.join(work_dir, "result.zip")

    try:
        # 2. 长截图
        print(f"正在截图: {req.url}")
        # 注意：这里假设 capture_full_page 内部是异步的
        await capture_full_page(req.url, img_path)

        # 3. 确定表格/文本检测方案
        table_scheme = ""
        if req.scheme in ["hf", "tiling_hf"]:
            table_scheme = 'hf'
        elif req.scheme in ["yolo", "tiling_yolo"]:
            table_scheme = 'yolo'
        else:
            # 错误处理：只接受 'hf' 或 'yolo' 及其 tiling 前缀
            raise HTTPException(
                status_code=400,
                detail=f"方案 '{req.scheme}' 不支持。请选择 'hf' 或 'yolo'。"
            )

        # 4. 执行 表格/文本 检测
        print(f"正在执行表格/文本检测，方案: {table_scheme.upper()}")
        table_content = detect_with_tiling(img_path, work_dir, scheme=table_scheme)
        
        # 5. 执行 插图 检测 (固定运行 'pic' 方案)
        print("正在执行插图检测，方案: PIC")
        pic_content = detect_with_tiling(img_path, work_dir, scheme='pic')

        # 6. 写入两个独立的坐标文件
        with open(table_txt_path, "w", encoding="utf-8") as f:
            f.write(table_content)
        print(f"表格/文本坐标已写入: {table_txt_path}")

        with open(pic_txt_path, "w", encoding="utf-8") as f:
            f.write(pic_content)
        print(f"插图坐标已写入: {pic_txt_path}")

        # 7. 打包成 ZIP
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(img_path, arcname="screenshot.png")
            zipf.write(table_txt_path, arcname="coordinates.txt") # 表格/文本结果
            zipf.write(pic_txt_path, arcname="pic_coordinates.txt") # 插图结果
        print(f"结果已打包到: {zip_path}")

        # 8. 返回文件流 (并安排后台任务在发送后删除临时文件)
        background_tasks.add_task(remove_file, work_dir)
        
        return FileResponse(
            zip_path, 
            media_type='application/zip', 
            filename=f"capture_{task_id}.zip"
        )

    except HTTPException as http_exc:
        remove_file(work_dir)
        raise http_exc
        
    except Exception as e:
        remove_file(work_dir) 
        print(f"内部错误: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"处理请求时发生内部错误: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)