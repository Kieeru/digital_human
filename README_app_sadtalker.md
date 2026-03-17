# 文本+照片-->数字人视频（使用edge-tts默认音色，无音色克隆功能）
1.	下载requirements_app_sadtalker.txt依赖

下载依赖若出错： Ignored the following versions that require a different python version:0.22.0 Requires-python>=3.9.......

则加一个阿里源：
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple

2.	下载tts依赖
pip install edge-tts gradio asyncio

并手动下载gradio依赖（老版本）
pip install gradio==3.44.4

4.	python app_sadtalker.py直接运行
出错：ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'

则找到文件
digital_human-main\venv\lib\site-packages\basicsr\data\degradations.py
将 'torchvision.transforms.functional_tensor'替换成'torchvision.transforms._functional_tensor'

