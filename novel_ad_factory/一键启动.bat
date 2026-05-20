@echo off
echo ==========================================
echo    正在检测环境并启动小说素材工厂...
echo ==========================================

:: 1. 自动安装缺失的库
pip install -r requirements.txt

:: 2. 启动服务器 (注意：这里直接使用 0.0.0.0 确保 cpolar 兼容)
echo 正在启动后端服务...
:: 自动打开浏览器查看（可选）
start http://127.0.0.1:8000/static/index.html
:: 启动 Uvicorn，host 必须是 0.0.0.0
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

pause