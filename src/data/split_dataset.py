import os
import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split # type: ignore

def create_splits(labels_csv: str, out_dir="data/processed", seed=42):
    """
    Create train/val/test splits from labels.csv.

    labels.csv columns:
        filename,smiles
    """

    df = pd.read_csv(labels_csv)

    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=seed, shuffle=True)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed, shuffle=True)

    os.makedirs(out_dir, exist_ok=True)

    train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(out_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, "test.csv"), index=False)

    print("Splits created:")
    print(f"Train: {len(train_df)}")
    print(f"Val:   {len(val_df)}")
    print(f"Test:  {len(test_df)}")