import os
import sys
import asyncio
import edge_tts
from pathlib import Path
import gradio as gr
from src.gradio_demo import SadTalker
import subprocess


try:
    import webui  # in webui
    in_webui = True
except:
    in_webui = False


# 整合 DigitalHumanGenerator 核心逻辑
class DigitalHumanGenerator:
    def __init__(self, sadtalker_path="."):
        """初始化数字人生成器"""
        self.sadtalker_path = Path(sadtalker_path)
        self.output_path = Path("output")
        self.output_path.mkdir(exist_ok=True)

    async def text_to_speech(self, text, audio_path="temp_audio.wav"):
        """文本转语音（使用edge-tts）"""
        if not text:
            raise ValueError("输入文本不能为空")
        
        # 选择中文语音
        communicate = edge_tts.Communicate(text, "zh-CN-XiaoxiaoNeural")
        await communicate.save(audio_path)
        
        return audio_path

    def generate_video(self, photo_path, audio_path, preprocess_type='full', is_still_mode=True, 
                      size_of_image=256,  enhancer=False):
        """调用SadTalker生成视频"""
        if not os.path.exists(photo_path):
            raise Exception(f"照片文件不存在: {photo_path}")
        if not os.path.exists(audio_path):
            raise Exception(f"音频文件不存在: {audio_path}")

        # 构建SadTalker命令
        cmd = [
            sys.executable,
            str(self.sadtalker_path / "inference.py"),
            "--driven_audio", audio_path,
            "--source_image", photo_path,
            "--result_dir", str(self.output_path),
            "--preprocess", preprocess_type,
            "--size", str(size_of_image),

        ]
        
        # 添加静态模式参数
        if is_still_mode:
            cmd.append("--still")
        
        # 添加人脸增强参数
        if enhancer:
            cmd.append("--enhancer")
            cmd.append("gfpgan")

        # 执行生成
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.sadtalker_path)
        
        if result.returncode != 0:
            raise Exception(f"SadTalker执行出错: {result.stderr}")

        # 查找生成的视频
        video_files = sorted(list(Path(self.output_path).glob("*.mp4")), key=os.path.getmtime)
        if video_files:
            return str(video_files[-1])  # 取最新生成的
        else:
            raise Exception("未找到生成的视频文件")

    async def generate(self, photo_path, text, preprocess_type='full', is_still_mode=True,
                      size_of_image=256, enhancer=False, cleanup=True):
        """主流程：文本→语音→视频"""
        # 生成语音
        audio_path = await self.text_to_speech(text)
        
        try:
            # 生成视频
            video_path = self.generate_video(
                photo_path, audio_path,
                preprocess_type=preprocess_type,
                is_still_mode=is_still_mode,
                size_of_image=size_of_image,

                enhancer=enhancer
            )
            return video_path
        finally:
            # 清理临时文件
            if cleanup and os.path.exists(audio_path):
                os.remove(audio_path)


# 初始化生成器实例
generator = DigitalHumanGenerator()


def toggle_audio_file(choice):
    if choice == False:
        return gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True)
    
def ref_video_fn(path_of_ref_video):
    if path_of_ref_video is not None:
        return gr.update(value=True)
    else:
        return gr.update(value=False)

# 整合后的生成函数（同步包装异步逻辑）
def generate_digital_human_video(source_image, input_text, preprocess_type, is_still_mode, 
                                enhancer,  size_of_image):
    """前端调用的生成函数"""
    try:
        # 验证输入
        if not source_image:
            raise ValueError("请上传人脸图片")
        if not input_text or input_text.strip() == "":
            raise ValueError("请输入要合成的文本内容")
        
        # 异步执行生成逻辑
        video_path = asyncio.run(generator.generate(
            photo_path=source_image,
            text=input_text,
            preprocess_type=preprocess_type,
            is_still_mode=is_still_mode,

            size_of_image=size_of_image,

            enhancer=enhancer
        ))
        
        return video_path
    except Exception as e:
        # 抛出友好的错误信息
        raise gr.Error(f"生成失败：{str(e)}")


def sadtalker_demo(checkpoint_path='checkpoints', config_path='src/config', warpfn=None):
    """构建Gradio界面"""
    sad_talker = SadTalker(checkpoint_path, config_path, lazy_load=True)

    with gr.Blocks(analytics_enabled=False, title="数字人视频生成工具") as sadtalker_interface:
        gr.Markdown("""
        <div align='center'> 
            <h2>  数字人视频生成工具 </h2>
        </div>
        """)
        
        with gr.Row().style(equal_height=False):
            # 左侧：输入区域
            with gr.Column(variant='panel', scale=1):
                # 图片上传
                with gr.Tabs(elem_id="sadtalker_source_image"):
                    with gr.TabItem('人脸图片'):
                        source_image = gr.Image(
                            label="上传人脸图片（建议正面照）", 
                            source="upload", 
                            type="filepath", 
                            elem_id="img2img_image"
                        ).style(width=512)

                # 文本输入（替换原有音频上传）
                with gr.Tabs(elem_id="sadtalker_driven_audio"):
                    with gr.TabItem('文本'):
                        input_text = gr.Textbox(
                            label="请输入要生成的播报文本", 
                            lines=8, 
                            placeholder="例如：大家好，我是你的数字人分身。今天很高兴能和大家见面，希望我的介绍能帮助你更好地了解数字人技术。",
                            elem_id="digital_human_text"
                        ).style(width=512)

            # 右侧：设置和输出区域
            with gr.Column(variant='panel', scale=1): 
                # 设置选项
                with gr.Tabs(elem_id="sadtalker_checkbox"):
                    with gr.TabItem('生成设置'):
                        gr.Markdown("""
                        ### 设置说明
                        - 人脸模型分辨率：256（速度快）/ 512（质量高）
                        - 预处理方式：full（适合静态模式）
                        - 静态模式：减少头部运动，配合full预处理使用
                        """)
                        with gr.Column(variant='panel'):
                         
                            size_of_image = gr.Radio(
                                [256, 512], value=256, 
                                label='人脸模型分辨率', 
                                info="256(速度快) / 512(质量高)"
                            )
                            preprocess_type = gr.Radio(
                                ['full'], 
                                value='full', label='图片预处理方式', 
                                info="full适合静态模式"
                            )
                            is_still_mode = gr.Checkbox(
                                label="静态模式（减少头部运动）", 
                                value=True
                            )
                           
                            enhancer = gr.Checkbox(
                                label="启用GFPGAN人脸增强", 
                                value=False
                            )
                            submit = gr.Button(
                                '生成数字人视频', 
                                elem_id="digital_human_generate", 
                                variant='primary'
                            ).style(full_width=True)
                        
                # 视频输出
                with gr.Tabs(elem_id="sadtalker_genearted"):
                    gen_video = gr.Video(
                        label="生成的数字人视频", 
                        format="mp4"
                    ).style(width=512)

        # 绑定生成按钮事件
        submit.click(
            fn=generate_digital_human_video,
            inputs=[
                source_image,        # 上传的图片
                input_text,          # 输入的文本
                preprocess_type,     # 预处理方式
                is_still_mode,       # 静态模式
                enhancer,            # 人脸增强

                size_of_image,       # 模型分辨率

            ],
            outputs=[gen_video]
        )

    return sadtalker_interface


if __name__ == "__main__":
    # 设置SadTalker路径（根据你的实际路径调整）
    if len(sys.argv) > 1:
        generator.sadtalker_path = Path(sys.argv[1])
    else:
        # 默认路径，根据实际情况修改
        if os.path.exists("D:\SadTalker"):
            generator.sadtalker_path = Path("D:\SadTalker")

    # 启动Gradio界面
    demo = sadtalker_demo()
    demo.queue(max_size=10)  # 启用队列，支持并发
    demo.launch(
        server_name="0.0.0.0",  # 允许外部访问
        server_port=7860,       # 端口号
        share=False,            # 是否生成公网链接
        inbrowser=True          # 自动打开浏览器
    )