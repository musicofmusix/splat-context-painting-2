import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# we use boilerplate used in the official DINOv3 segmentation demo notebook
DEFAULT_DEVICE = torch.device("cuda")
IMAGE_SIZE = 768
PATCH_SIZE = 16

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

dinov3_processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
dinov3_model = AutoModel.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m").to(DEFAULT_DEVICE).eval()

def resize_transform(image):
    w, h = image.size
    h_patches = IMAGE_SIZE // PATCH_SIZE
    w_patches = round((w * IMAGE_SIZE) / (h * PATCH_SIZE))
    w_patches = max(w_patches, 1)
    return TF.to_tensor(TF.resize(image, (h_patches * PATCH_SIZE, w_patches * PATCH_SIZE)))


def dinov3_process(image_np: np.ndarray):
    image_np = (image_np.clip(0.0, 1.0) * 255.0).astype(np.uint8)
    image = Image.fromarray(image_np, "RGB")

    pixel_values = resize_transform(image)
    pixel_values = TF.normalize(pixel_values, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    pixel_values = pixel_values.unsqueeze(0).to(DEFAULT_DEVICE)

    _, _, img_height, img_width = pixel_values.shape
    num_patches_height = img_height // PATCH_SIZE
    num_patches_width = img_width  // PATCH_SIZE

    with torch.inference_mode():
        outputs = dinov3_model(pixel_values=pixel_values)

    last_hidden_states = outputs.last_hidden_state

    cls_token = last_hidden_states[:, 0, :]
    patch_embeddings_flat = last_hidden_states[:, 1 + dinov3_model.config.num_register_tokens:, :]
    patch_embeddings = patch_embeddings_flat.unflatten(1, (num_patches_height, num_patches_width))

    return cls_token, patch_embeddings