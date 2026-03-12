from PIL import Image
from io import BytesIO

def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """
    Load a single image (PNG/JPG) from bytes into a PIL Image.
    """
    img = Image.open(BytesIO(image_bytes))
    return img.convert("RGB")  # Ensure model inputs are consistent