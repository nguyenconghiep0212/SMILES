import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from src.data.dataset import MoleculeImageDataset
from src.utils.smiles_tokenizer import SmilesTokenizer
from src.models.image_to_smiles import ImageToSmilesModel


def train_model(
    train_csv="data/processed/train.csv",
    val_csv="data/processed/val.csv",
    images_dir="data/processed/images",
    tokenizer_path="data/processed/tokenizer.json",
    batch_size=8,
    max_len=256,
    epochs=10,
    lr=1e-4,
    device="cuda"
):
    # Load tokenizer
    tokenizer = SmilesTokenizer()
    tokenizer.load(tokenizer_path)

    vocab_size = len(tokenizer.vocab)

    # Datasets
    train_ds = MoleculeImageDataset(
        csv_path=train_csv,
        images_dir=images_dir,
        tokenizer=tokenizer,
        augment=True
    )

    val_ds = MoleculeImageDataset(
        csv_path=val_csv,
        images_dir=images_dir,
        tokenizer=tokenizer,
        augment=False
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Model
    model = ImageToSmilesModel(vocab_size=vocab_size, max_len=max_len)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.vocab[tokenizer.pad])
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, token_ids in train_loader:
            images = images.to(device)
            token_ids = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(t) for t in token_ids], batch_first=True,
                padding_value=tokenizer.vocab[tokenizer.pad]
            ).to(device)

            # Input to decoder is token_ids[:, :-1]
            # Target to match is token_ids[:, 1:]
            inputs = token_ids[:, :-1]
            targets = token_ids[:, 1:]

            logits = model(images, inputs)
            loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {total_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), "experiments/checkpoints/model_baseline.pt")