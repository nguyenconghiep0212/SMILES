from PIL import Image, ImageOps

def normalize_image(img: Image.Image, target_size=256):
    """
    Normalize image:
    - Convert to RGB
    - Pad to square
    - Resize to target_size x target_size
    """

    img = img.convert("RGB")

    # Make square by padding
    w, h = img.size
    max_side = max(w, h)

    # Center pad
    delta_w = max_side - w
    delta_h = max_side - h
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))

    squared = ImageOps.expand(img, padding, fill="white")  # White background

    # Resize to target size
    resized = squared.resize((target_size, target_size), Image.LANCZOS) # type: ignore

    return resized