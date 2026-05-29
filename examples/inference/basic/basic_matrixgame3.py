from fastvideo import VideoGenerator

MODEL_PATH = "FastVideo/Matrix-Game-3.0-Base-Distilled-Diffusers"
IMAGE_URL = "https://raw.githubusercontent.com/SkyworkAI/Matrix-Game/main/Matrix-Game-3/demo_images/001/image.png"
PROMPT = "A colorful, animated cityscape with a gas station and various buildings."
OUTPUT_PATH = "video_samples_matrixgame3"


def main():
    generator = VideoGenerator.from_pretrained(
        MODEL_PATH,
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
        # transformer/config.json in the HF repo ships _class_name="WanModel",
        # which isn't in FastVideo's registry. model_index.json has the right
        # name; override here to force the matrixgame3 class.
        override_transformer_cls_name="MatrixGame3WanModel",
    )

    generator.generate_video(
        prompt=PROMPT,
        image_path=IMAGE_URL,
        height=720,
        width=1280,
        num_frames=57,
        num_inference_steps=3,
        guidance_scale=1.0,
        seed=42,
        output_path=OUTPUT_PATH,
        save_video=True,
    )


if __name__ == "__main__":
    main()
