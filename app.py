import os
import sys
import asyncio
import subprocess
from pathlib import Path

if sys.platform == "win32":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    try:
        import site
        import ctypes

        site_paths = site.getsitepackages()
        if site_paths:
            torch_lib = Path(site_paths[0]) / "torch" / "lib"
            if torch_lib.exists():
                os.add_dll_directory(str(torch_lib))
                libiomp = torch_lib / "libiomp5md.dll"
                if libiomp.exists():
                    ctypes.CDLL(str(libiomp))
    except Exception:
        pass

from TTS.api import TTS
import gradio as gr
import numpy as np
import soundfile as sf
import torch
import torchaudio

from src.gradio_demo import SadTalker


try:
    import webui  
    in_webui = True
except Exception:
    in_webui = False


def _patch_torchaudio_load():
    original_load = torchaudio.load

    def compat_load(uri, *args, **kwargs):
        try:
            return original_load(uri, *args, **kwargs)
        except Exception as exc:
            message = str(exc).lower()
            if "torchcodec" not in message and "libtorchcodec" not in message:
                raise

            audio, sample_rate = sf.read(uri, always_2d=True, dtype="float32")
            audio = torch.from_numpy(np.ascontiguousarray(audio.T))
            return audio, sample_rate

    torchaudio.load = compat_load


_patch_torchaudio_load()


class DigitalHumanGenerator:
    def __init__(self, sadtalker_path="."):
        """Wrapper around SadTalker + TTS pipelines."""
        self.sadtalker_path = Path(sadtalker_path)
        self.output_path = Path("output")
        self.output_path.mkdir(exist_ok=True)
        self.use_cuda = torch.cuda.is_available()

        # 参考音色
        self.voice_presets = {
            "李老师": {"type": "xtts", "ref": Path("ref_audios/teacher_li.wav")},
            "张老师": {"type": "xtts", "ref": Path("ref_audios/teacher_zhang.wav")},
            "王老师": {"type": "xtts", "ref": Path("ref_audios/teacher_wang.wav")},
        }
        self.xtts = None
        self._torch_load_patched = False

    def _patch_torch_load(self):
        if self._torch_load_patched:
            return

        original_torch_load = torch.load

        def compat_torch_load(*args, **kwargs):
            if "weights_only" not in kwargs:
                kwargs["weights_only"] = False
            return original_torch_load(*args, **kwargs)

        torch.load = compat_torch_load
        self._torch_load_patched = True

    def _load_xtts(self):
        if self.xtts is None:
            self._patch_torch_load()
            self.xtts = TTS(
                "tts_models/multilingual/multi-dataset/xtts_v2",
                progress_bar=False,
                gpu=self.use_cuda,
            )

    async def text_to_speech(
        self,
        text: str,
        voice_preset: str = "李老师",
        ref_audio: Path | None = None,
        audio_path: str = "temp_audio.wav",
    ) -> str:
        if not text:
            raise ValueError("请输入需要合成的文本。")
        preset = self.voice_presets.get(voice_preset, self.voice_presets["李老师"])
        speaker_wav = ref_audio or preset["ref"]
        if not speaker_wav or not Path(speaker_wav).exists():
            raise ValueError("参考音频不存在，请上传有效的 WAV 文件。")
        self._load_xtts()
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self.xtts.tts_to_file(
                text=text,
                speaker_wav=str(speaker_wav),
                language="zh",
                file_path=audio_path,
            ),
        )
        return audio_path

    def generate_video(
        self,
        photo_path: str,
        audio_path: str,
        preprocess_type: str = "full",
        is_still_mode: bool = True,
        size_of_image: int = 256,
        enhancer: bool = False,
    ) -> str:
        if not os.path.exists(photo_path):
            raise FileNotFoundError(f"找不到图片：{photo_path}")
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"找不到音频：{audio_path}")

        cmd = [
            sys.executable,
            str(self.sadtalker_path / "inference.py"),
            "--driven_audio",
            audio_path,
            "--source_image",
            photo_path,
            "--result_dir",
            str(self.output_path),
            "--preprocess",
            preprocess_type,
            "--size",
            str(size_of_image),
        ]
        if is_still_mode:
            cmd.append("--still")
        if enhancer:
            cmd.extend(["--enhancer", "gfpgan"])
        if not self.use_cuda:
            cmd.append("--cpu")

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.sadtalker_path)
        if result.returncode != 0:
            raise RuntimeError(f"SadTalker 运行失败: {result.stderr}")

        video_files = sorted(
            Path(self.output_path).glob("*.mp4"), key=os.path.getmtime
        )
        if not video_files:
            raise RuntimeError("未找到生成的视频文件。")
        return str(video_files[-1])

    async def generate(
        self,
        photo_path: str,
        text: str,
        voice_preset: str = "李老师",
        ref_audio: Path | None = None,
        preprocess_type: str = "full",
        is_still_mode: bool = True,
        size_of_image: int = 256,
        enhancer: bool = False,
        cleanup: bool = True,
    ) -> str:
        audio_path = await self.text_to_speech(
            text, voice_preset=voice_preset, ref_audio=ref_audio
        )
        try:
            return self.generate_video(
                photo_path,
                audio_path,
                preprocess_type=preprocess_type,
                is_still_mode=is_still_mode,
                size_of_image=size_of_image,
                enhancer=enhancer,
            )
        finally:
            if cleanup and os.path.exists(audio_path):
                os.remove(audio_path)


generator = DigitalHumanGenerator()


def toggle_audio_file(choice):
    if choice is False:
        return gr.update(visible=True), gr.update(visible=False)
    return gr.update(visible=False), gr.update(visible=True)


def ref_video_fn(path_of_ref_video):
    if path_of_ref_video is not None:
        return gr.update(value=True)
    return gr.update(value=False)


def generate_digital_human_video(
    source_image,
    input_text,
    preprocess_type,
    is_still_mode,
    enhancer,
    size_of_image,
    voice_preset,
    ref_audio_path,
):
    try:
        if not source_image:
            raise ValueError("请上传人像图片。")
        if not input_text or input_text.strip() == "":
            raise ValueError("请输入需要合成的文本。")

        video_path = asyncio.run(
            generator.generate(
                photo_path=source_image,
                text=input_text,
                voice_preset=voice_preset,
                ref_audio=ref_audio_path,
                preprocess_type=preprocess_type,
                is_still_mode=is_still_mode,
                size_of_image=size_of_image,
                enhancer=enhancer,
            )
        )
        return video_path
    except Exception as e:
        raise gr.Error(f"生成失败: {e}")


def sadtalker_demo(checkpoint_path="checkpoints", config_path="src/config", warpfn=None):
    sad_talker = SadTalker(checkpoint_path, config_path, lazy_load=True)
    with gr.Blocks(analytics_enabled=False, title="数字人生成") as sadtalker_interface:
        gr.Markdown(
            """
        <div align='center'> 
            <h2>数字人生成</h2>
        </div>
        """
        )
        with gr.Row().style(equal_height=False):
            # 左侧：输入
            with gr.Column(variant="panel", scale=1):
                with gr.Tabs(elem_id="sadtalker_source_image"):
                    with gr.TabItem("人像图片"):
                        source_image = gr.Image(
                            label="上传人像图片",
                            source="upload",
                            type="filepath",
                            elem_id="img2img_image",
                        ).style(width=512)

                with gr.Tabs(elem_id="sadtalker_driven_audio"):
                    with gr.TabItem("文本 / 参考音频"):
                        input_text = gr.Textbox(
                            label="输入要说的话",
                            lines=8,
                            placeholder="请输入合成文本",
                            elem_id="digital_human_text",
                        ).style(width=512)
                        voice_preset = gr.Radio(
                            list(generator.voice_presets.keys()),
                            value="李老师",
                            label="声音预设",
                            info="使用 XTTS 参考音频生成声音",
                        )
                        ref_audio = gr.Audio(
                            label="参考音频 wav (3-15 秒)",
                            type="filepath",
                            interactive=True,
                        )

            # 右侧：参数与输出
            with gr.Column(variant="panel", scale=1):
                with gr.Tabs(elem_id="sadtalker_checkbox"):
                    with gr.TabItem("参数设置"):
                        gr.Markdown(
                            """
                        ### 生成参数
                        - 推荐 256/512 分辨率
                        - 建议使用 full 预处理
                        """
                        )
                        with gr.Column(variant="panel"):
                            size_of_image = gr.Radio(
                                [256, 512],
                                value=256,
                                label="输出分辨率",
                                info="256（快） / 512（清晰）",
                            )
                            preprocess_type = gr.Radio(
                                ["full"],
                                value="full",
                                label="预处理方式",
                                info="full 适合大多数场景",
                            )
                            is_still_mode = gr.Checkbox(
                                label="静态模式（减少抖动）", value=True
                            )
                            enhancer = gr.Checkbox(
                                label="使用 GFPGAN 增强人脸", value=False
                            )
                            submit = gr.Button(
                                "生成视频",
                                elem_id="digital_human_generate",
                                variant="primary",
                            ).style(full_width=True)

                with gr.Tabs(elem_id="sadtalker_genearted"):
                    gen_video = gr.Video(
                        label="生成结果", format="mp4"
                    ).style(width=512)

        submit.click(
            fn=generate_digital_human_video,
            inputs=[
                source_image,
                input_text,
                preprocess_type,
                is_still_mode,
                enhancer,
                size_of_image,
                voice_preset,
                ref_audio,
            ],
            outputs=[gen_video],
        )
    return sadtalker_interface


if __name__ == "__main__":
    if len(sys.argv) > 1:
        generator.sadtalker_path = Path(sys.argv[1])
    else:
        if os.path.exists("D:\digital_human-main"):
            generator.sadtalker_path = Path("D:\digital_human-main")

    demo = sadtalker_demo()
    demo.queue(max_size=10)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
    )
