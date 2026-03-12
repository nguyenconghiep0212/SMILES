import os
import pandas as pd # type: ignore
from torch.utils.data import Dataset
from PIL import Image

from src.data.image_loader import load_image_from_bytes
from src.data.image_normalization import normalize_image
from src.data.transforms import get_preprocess_transforms
from src.utils.smiles_tokenizer import SmilesTokenizer


class MoleculeImageDataset(Dataset):
    def __init__(self, csv_path, images_dir="data/processed/images",
                 tokenizer: SmilesTokenizer = None, # type: ignore
                 image_size=256,
                 augment=False):
        """
        PyTorch dataset for image → SMILES training.

        csv_path columns:
            filename,smiles
        """

        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        
        self.tokenizer = tokenizer
        self.image_size = image_size

        if augment:
            from src.data.transforms import get_training_aug_transforms
            self.transform = get_training_aug_transforms(image_size)
        else:
            self.transform = get_preprocess_transforms(image_size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(self.images_dir, row["filename"])

        # Load image
        img = Image.open(img_path).convert("RGB")

        # Normalize
        img = normalize_image(img, self.image_size)

        # Transform (tensor)
        x = self.transform(img)

        # SMILES → token ids
        token_ids = self.tokenizer.encode(row["smiles"])

        return x, token_ids