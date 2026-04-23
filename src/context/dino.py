import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

DEFAULT_DEVICE = torch.device("cuda")

dinov3_processor =  AutoImageProcessor.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
dinov3_model = AutoModel.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m").to(DEFAULT_DEVICE).eval()

def dinov3_process(image_np):
    image_np = (image_np.clip(0.0, 1.0) * 255.0).astype(np.uint8)
    image = Image.fromarray(image_np, "RGB")

    patch_size = dinov3_model.config.patch_size

    inputs = dinov3_processor(images=image, return_tensors="pt").to(DEFAULT_DEVICE)

    with torch.inference_mode():
        outputs = dinov3_model(**inputs)

    last_hidden_states = outputs.last_hidden_state
    batch_size, _, img_height, img_width = inputs["pixel_values"].shape
    num_patches_height, num_patches_width = img_height // patch_size, img_width // patch_size

    cls_token = last_hidden_states[:, 0, :]
    patch_embeddings_flat = last_hidden_states[:, 1 + dinov3_model.config.num_register_tokens :, :]
    patch_embeddings = patch_embeddings_flat.unflatten(1, (num_patches_height, num_patches_width))

    return cls_token, patch_embeddings
