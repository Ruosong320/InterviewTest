import os
import shutil
import uuid
import zipfile
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel

# 导入必要的模块：长截图
from webLongShot import capture_full_page
from Task2_Solve1 import detect_with_hf #方案一 
# from scheme_yolo import detect_with_yolo  # 方案二

app = FastAPI()

# 定义请求体
class CaptureRequest(BaseModel):
    url: str
    # 默认值改为 'hf'，因为它是唯一可用的方案
    scheme: str = "hf" 

# 临时文件清理任务
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
    # 1. 创建唯一的工作目录
    task_id = str(uuid.uuid4())
    work_dir = f"./temp_{task_id}"
    os.makedirs(work_dir, exist_ok=True)
    
    img_path = os.path.join(work_dir, "screenshot.png")
    txt_path = os.path.join(work_dir, "coordinates.txt")
    zip_path = os.path.join(work_dir, "result.zip")

    try:
        # 2. 长截图
        print(f"正在截图: {req.url}")
        # 使用 Google 搜索工具来验证 URL 的可访问性和内容（可选步骤，但有助于确保链接有效）
        
        await capture_full_page(req.url, img_path)

        # 3. 根据方案进行检测
        print(f"正在检测，使用方案: {req.scheme}")
        content = ""
        if req.scheme == "hf":
            content = detect_with_hf(img_path)
        elif req.scheme == "yolo":
            # 禁用未导入的方案
            content = f"方案 {req.scheme} 尚未部署，请使用 'hf' 方案。"

        # 4. 写入坐标文件
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(content)

        # 5. 打包成 ZIP
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(img_path, arcname="screenshot.png")
            zipf.write(txt_path, arcname="coordinates.txt")

        # 6. 返回文件流 (并安排后台任务在发送后删除临时文件)
        background_tasks.add_task(remove_file, work_dir)
        
        return FileResponse(
            zip_path, 
            media_type='application/zip', 
            filename=f"capture_{task_id}.zip"
        )

    except Exception as e:
        remove_file(work_dir) # 出错也要清理
        # 捕捉 Playwright 截图失败等错误
        return {"status": "error", "message": f"处理请求时出错: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)