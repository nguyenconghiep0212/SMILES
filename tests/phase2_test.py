from src.data.pdf_utils import pdf_to_images
from src.data.image_loader import load_image_from_bytes
from src.data.transforms import get_preprocess_transforms

# Test PDF loading
pdf_path = "data/raw/test.pdf"
with open(pdf_path, "rb") as f:
    images = pdf_to_images(f.read())

print(f"PDF pages extracted: {len(images)}")
images[0].show()

# Test image loading
img_path = "data/raw/caffeine.png"
with open(img_path, "rb") as f:
    img = load_image_from_bytes(f.read())

img.show()

# Test preprocessing
transform = get_preprocess_transforms()

tensor = transform(img)
print("Transformed tensor shape:", tensor.shape) # type: ignore
