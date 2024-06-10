# Adapted from https://github.com/luosiallen/latent-consistency-model

import os
import uuid
from omegaconf import OmegaConf

import random

import imageio
import torch
import torchvision
import gradio as gr
import numpy as np
from gradio.components import Textbox, Video

from utils.lora import collapse_lora, monkeypatch_remove_lora
from utils.lora_handler import LoraHandler
from utils.common_utils import load_model_checkpoint
from utils.utils import instantiate_from_config
from scheduler.t2v_turbo_scheduler import T2VTurboScheduler
from pipeline.t2v_turbo_vc2_pipeline import T2VTurboVC2Pipeline

DESCRIPTION = """# T2V-Turbo 🚀
We provide T2V-Turbo (VC2) distilled from [VideoCrafter2](https://ailab-cvc.github.io/videocrafter2/) with the reward feedback from [HPSv2.1](https://github.com/tgxs002/HPSv2/tree/master) and [InternVid2 Stage 2 Model](https://huggingface.co/OpenGVLab/InternVideo2-Stage2_1B-224p-f4).

You can download the the models from [here](https://huggingface.co/jiachenli-ucsb/T2V-Turbo-VC2). Check out our [Project page](https://t2v-turbo.github.io) 😄
"""
if torch.cuda.is_available():
    DESCRIPTION += "\n<p>Running on CUDA 😀</p>"
elif hasattr(torch, "xpu") and torch.xpu.is_available():
    DESCRIPTION += "\n<p>Running on XPU 🤓</p>"
else:
    DESCRIPTION += "\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>"
MAX_SEED = np.iinfo(np.int32).max


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


def save_video(video_array, video_save_path, fps: int = 16):
    video = video_array.detach().cpu()
    video = torch.clamp(video.float(), -1.0, 1.0)
    video = video.permute(1, 0, 2, 3)  # t,c,h,w
    video = (video + 1.0) / 2.0
    video = (video * 255).to(torch.uint8).permute(0, 2, 3, 1)

    torchvision.io.write_video(
        video_save_path, video, fps=fps, video_codec="h264", options={"crf": "10"}
    )

example_txt = [
    "An astronaut riding a horse.",
    "Darth vader surfing in waves.",
    "Robot dancing in times square.",
    "Clown fish swimming through the coral reef.",
    "Pikachu snowboarding.",
    "With the style of van gogh, A young couple dances under the moonlight by the lake.",
    "A young woman with glasses is jogging in the park wearing a pink headband.",
    "Impressionist style, a yellow rubber duck floating on the wave on the sunset",
    "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
    "With the style of low-poly game art, A majestic, white horse gallops gracefully across a moonlit beach.",
]

examples = [[i, 7.5, 4, 16, 16] for i in example_txt]

@torch.inference_mode()
def generate(
    prompt: str,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 4,
    num_frames: int = 16,
    fps: int = 16,
    seed: int = 0,
    randomize_seed: bool = False,
):

    seed = int(randomize_seed_fn(seed, randomize_seed))
    result = pipeline(
        prompt=prompt,
        frames=num_frames,
        fps=fps,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_videos_per_prompt=1,
    )

    torch.cuda.empty_cache()
    tmp_save_path = "tmp.mp4"
    root_path = "./videos/"
    os.makedirs(root_path, exist_ok=True)
    video_save_path = os.path.join(root_path, tmp_save_path)

    save_video(result[0], video_save_path, fps=fps)
    display_model_info = f"Video size: {num_frames}x320x512, Sampling Step: {num_inference_steps}, Guidance Scale: {guidance_scale}"
    return video_save_path, prompt, display_model_info, seed


block_css = """
#buttons button {
    min-width: min(120px,100%);
}
"""


if __name__ == "__main__":
    device = torch.device("cuda:0")

    config = OmegaConf.load("configs/inference_t2v_512_v2.0.yaml")
    model_config = config.pop("model", OmegaConf.create())
    pretrained_t2v = instantiate_from_config(model_config)
    pretrained_t2v = load_model_checkpoint(pretrained_t2v, "checkpoints/vc2_model.ckpt")

    unet_config = model_config["params"]["unet_config"]
    unet_config["params"]["time_cond_proj_dim"] = 256
    unet = instantiate_from_config(unet_config)

    unet.load_state_dict(
        pretrained_t2v.model.diffusion_model.state_dict(), strict=False
    )

    use_unet_lora = True
    lora_manager = LoraHandler(
        version="cloneofsimo",
        use_unet_lora=use_unet_lora,
        save_for_webui=True,
        unet_replace_modules=["UNetModel"],
    )
    lora_manager.add_lora_to_model(
        use_unet_lora,
        unet,
        lora_manager.unet_replace_modules,
        lora_path="checkpoints/unet_lora.pt",
        dropout=0.1,
        r=64,
    )
    unet.eval()
    collapse_lora(unet, lora_manager.unet_replace_modules)
    monkeypatch_remove_lora(unet)

    pretrained_t2v.model.diffusion_model = unet
    scheduler = T2VTurboScheduler(
        linear_start=model_config["params"]["linear_start"],
        linear_end=model_config["params"]["linear_end"],
    )
    pipeline = T2VTurboVC2Pipeline(pretrained_t2v, scheduler, model_config)

    pipeline.to(device)

    demo = gr.Interface(
        fn=generate,
        inputs=[
            Textbox(label="", placeholder="Please enter your prompt. \n"),
            gr.Slider(
                label="Guidance scale",
                minimum=2,
                maximum=14,
                step=0.1,
                value=7.5,
            ),
            gr.Slider(
                label="Number of inference steps",
                minimum=1,
                maximum=8,
                step=1,
                value=4,
            ),
            gr.Slider(
                label="Number of Video Frames",
                minimum=16,
                maximum=48,
                step=8,
                value=16,
            ),
            gr.Slider(
                label="FPS",
                minimum=8,
                maximum=32,
                step=4,
                value=16,
            ),
            gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
                randomize=True,
            ),
            gr.Checkbox(label="Randomize seed", value=True),
        ],
        outputs=[
            gr.Video(label="Generated Video", width=512, height=320, interactive=False, autoplay=True),
            Textbox(label="input prompt"),
            Textbox(label="model info"),
            gr.Slider(label="seed"),
        ],
        description=DESCRIPTION,
        theme=gr.themes.Default(),
        css=block_css,
        examples=examples,
        cache_examples=False,
        concurrency_limit=10,
    )
    demo.queue().launch(share=True)
