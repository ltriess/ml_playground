import logging

import torch
from diffusers import AutoencoderKL, EulerDiscreteScheduler, UNet2DConditionModel
from torchvision.transforms import ToPILImage
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("Instantiate models")

    tokenizer = CLIPTokenizer.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="tokenizer"
    )

    text_encoder = CLIPTextModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="text_encoder", use_safetensors=True
    ).to("cuda")

    scheduler = EulerDiscreteScheduler.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="scheduler"
    )

    unet = UNet2DConditionModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="unet", use_safetensors=True
    ).to("cuda")

    vae = AutoencoderKL.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="vae", use_safetensors=True
    ).to("cuda")

    logger.info("Initialize parameters")

    # Create a single prompt
    prompts = [
        "silly cat wearing a batman costume, funny, realistic, canon, award winning photography",
    ]

    batch_size = 1
    inference_steps = 30
    seed = 1055747  # just for reproducibility
    cfg_scale = 7
    height = 512
    width = 512

    logger.info("Conditioning")

    # Convert text prompt into tokens
    cond_input = tokenizer(
        prompts,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # Create an embedding with attention for each token
    with torch.no_grad():
        cond_embeddings = text_encoder(cond_input.input_ids.to("cuda"))[0]

    logger.info("Unconditioning")

    # Create empty string tokens for each image
    uncond_input = tokenizer(
        [""] * batch_size,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to("cuda"))[0]

    # Join conditioned and unconditioned embeddings into a single tensor
    text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])

    logger.info("Generate a tensor with noise")

    # Generate noise with a generator using our seed
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)

    # Generate a tensor with noise
    latents = torch.randn(
        (batch_size, unet.config.in_channels, height // 8, width // 8),
        generator=generator,
        device="cuda",
    )

    logger.info("Clean up the noise tensor")

    # Set how many steps to use to "clean" the tensor
    scheduler.set_timesteps(inference_steps)

    # Multiply the tensor values with the standard deviation of the initial noise distribution
    latents *= scheduler.init_noise_sigma

    for t in tqdm(scheduler.timesteps):
        # Duplicate tensor -> for conditioned values and for unconditioned values
        latent_model_input = torch.cat([latents] * 2)

        # Scale input based on timestep for compatibility between various schedulers
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

        with torch.no_grad():
            # Predict amout of noise in the tensor with U-Net
            noise_pred = unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            ).sample

        # Assign half of estimated noise to conditioning and other half to unconditioning
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)

        # Adjust prediction towards conditioned result (more or less importance given to prompt)
        noise_pred = noise_pred_uncond + cfg_scale * (
            noise_pred_cond - noise_pred_uncond
        )

        # Substract amout of noise previously calculated -> step to clean tensor from noise
        latents = scheduler.step(
            noise_pred, t, latents, generator=generator
        ).prev_sample

    logger.info("Convert the tensor into an image")

    # Normalize the tensor with the scale factor and decode it into image space
    latents /= vae.config.scaling_factor
    with torch.no_grad():
        images = vae.decode(latents).sample

    # Values of "images" range from -1 to 1, therefore normalize from 0 to 1
    images = (images / 2 + 0.5).clamp(0, 1)

    to_pil = ToPILImage()
    for i in range(batch_size):
        image = to_pil(images[i])
        image.save(f"image_{i:04d}.png")


if __name__ == "__main__":
    main()
