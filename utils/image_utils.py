# utils/image_utils.py

import torch
import timm
from torchvision import transforms
from PIL import Image
from functools import lru_cache

def get_transform_vit(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

@lru_cache(maxsize=1)
def load_model_vit(model_path: str) -> torch.nn.Module:
    """
    Load & return a ViT-Small model for deepfake detection.
    Cached so disk I/O happens only once.
    """
    model = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

def preprocess_image_vit(image: Image.Image, transform: transforms.Compose) -> torch.Tensor:
    """
    Apply transform and add batch dimension.
    """
    return transform(image).unsqueeze(0)
