from PIL import Image
import torchvision.transforms as T


def get_preprocess_transforms(image_size=256):
    """
    Preprocess for inference and training:
    - Resize
    - Convert to tensor
    - Normalize
    """

    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])


def get_training_aug_transforms(image_size=256):
    """
    Data augmentation for training.
    Helps model handle rotated/noisy chemical diagrams.
    """

    return T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomRotation(3),  # slight rotations
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])