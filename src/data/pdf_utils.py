import fitz  # PyMuPDF
from PIL import Image
import io

# Opens a PDF from raw bytes (compatible with FastAPI upload)
# Converts each page into a PNG-like bitmap
# Returns a list of PIL Image objects
def pdf_to_images(pdf_bytes: bytes):
    """
    Convert a PDF (given as bytes) into a list of PIL Images.
    Each PDF page becomes one image.
    Returns: List[PIL.Image]
    """
    images = []
    pdf = fitz.open(stream=pdf_bytes, filetype="pdf")

    for page_index in range(len(pdf)):
        page = pdf.load_page(page_index)

        # Render page to PNG image
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x upscale for quality
        img_bytes = pix.tobytes("png")
        pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        images.append(pil_image)

    pdf.close()
    return images