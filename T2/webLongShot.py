from playwright.async_api import async_playwright
import os

async def capture_full_page(url: str, save_path: str):
    """
    异步访问 URL 并截取长图保存到 save_path
    """
    # 切换到 async_playwright
    async with async_playwright() as p:
        # 使用 await 启动异步浏览器
        browser = await p.chromium.launch(headless=True)
        # 异步创建页面
        page = await browser.new_page(viewport={'width': 1280, 'height': 720})
        try:
            # 异步导航和等待
            await page.goto(url, wait_until="networkidle", timeout=60000)
            # 异步截图
            await page.screenshot(path=save_path, full_page=True)
        except Exception as e:
            print(f"截图失败: {e}")
            raise e
        finally:
            # 异步关闭浏览器
            await browser.close()
    return save_path