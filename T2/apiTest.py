import requests
import os
import time

# --- 配置 ---
API_URL = "http://127.0.0.1:8000/api/v1/capture_and_locate" #api地址
#TEST_URL = "https://baijiahao.baidu.com/s?id=1804062717515460724&wfr=spider&for=pc" #测试网页
TEST_URL = "https://www.joca.cn/CN/column/column17.shtml"
OUTPUT_DIR = "results" 
# --------------

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
   
    output_path = os.path.join(script_dir, OUTPUT_DIR)
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 禁用代理
    os.environ['NO_PROXY'] = '127.0.0.1,localhost'
    
    # 准备请求数据
    payload = {
        "url": TEST_URL,
        "scheme": "yolo"   # 切换为yolo使用yolo模式 hf为使用table-transformer-detection
    }
    
    try:
        print(f"发送请求到: {API_URL}")
        print(f"目标URL: {TEST_URL}")
        
        # 发送请求
        response = requests.post(
            API_URL,
            json=payload,
            timeout=300
        )
        
        print(f"响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            # 生成文件名
            timestamp = int(time.time())
            filename = f"result_{timestamp}.zip"
            filepath = os.path.join(OUTPUT_DIR, filename)
            
            # 保存文件
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            print(f"文件已保存到: {filepath}")
            print(f"文件大小: {len(response.content) / 1024:.1f} KB")
        else:
            print(f"错误: {response.text}")
            
    except Exception as e:
        print(f"请求失败: {e}")

if __name__ == "__main__":
    main()