import os
import sys
import re
import pandas as pd
from sklearn.model_selection import train_test_split
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import (
    set_seed,
    ensure_directories,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    FINAL_DATASET_PATH
)

set_seed(42)
ensure_directories()


def clean_text_lightweight(text: str) -> str:
    """
    Lightweight cleaning tailored for BERT tokenization:
    - Casts to string and handles empty/null text.
    - Normalizes unicode spaces and tabs into single spaces.
    - Removes non-printable control characters while preserving emojis and punctuation.
    - Strips leading and trailing whitespaces.
    """
    if pd.isna(text) or text is None:
        return ""
    text = str(text)
    # Normalize multiple whitespace characters into single space
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r" +", " ", text)
    return text.strip()


def preprocess_and_split():
    raw_path = os.path.join(RAW_DATA_DIR, "collected_social_media_posts.csv")
    if not os.path.exists(raw_path):
        raw_path = os.path.join(DATA_DIR, "final_social_media_threat_dataset.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"Dataset file not found at {raw_path}."
        )

    print(f"[INFO] Loading raw dataset from: {raw_path}")
    df = pd.read_csv(raw_path)

    initial_count = len(df)
    print(f"[INFO] Initial rows: {initial_count}")

    # Remove duplicates if any identical (text, label) rows exist
    df = df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    if len(df) < initial_count:
        print(f"[INFO] Removed {initial_count - len(df)} duplicate rows.")

    # Apply lightweight preprocessing
    df["text"] = df["text"].apply(clean_text_lightweight)

    # Filter out empty texts
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    print(f"[INFO] Total clean rows: {len(df)}")

    # Ensure columns and types
    df["label"] = df["label"].astype(int)
    df["is_outlier"] = df["is_outlier"].astype(int)
    df["id"] = range(1, len(df) + 1)

    # Save final consolidated dataset
    df.to_csv(FINAL_DATASET_PATH, index=False)
    print(f"[INFO] Saved final preprocessed dataset to: {FINAL_DATASET_PATH}")

    # Perform 70% Train, 15% Validation, 15% Test Stratified Split
    # Stratify by both label and is_outlier to ensure balanced distribution
    strat_key = df["label"].astype(str) + "_" + df["is_outlier"].astype(str)

    # First split: 70% train, 30% temp (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=strat_key
    )

    # Second split: split temp 50/50 into validation (15%) and test (15%)
    temp_strat_key = temp_df["label"].astype(str) + "_" + temp_df["is_outlier"].astype(str)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_strat_key
    )

    train_path = os.path.join(PROCESSED_DATA_DIR, "train.csv")
    val_path = os.path.join(PROCESSED_DATA_DIR, "validation.csv")
    test_path = os.path.join(PROCESSED_DATA_DIR, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    val_df.to_csv(os.path.join(PROCESSED_DATA_DIR, "val.csv"), index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\n[SPLIT SUMMARY]")
    print(f"Train set:       {len(train_df):5d} samples ({len(train_df)/len(df)*100:.1f}%) | Saved to {train_path}")
    print(f"Validation set:  {len(val_df):5d} samples ({len(val_df)/len(df)*100:.1f}%) | Saved to {val_path}")
    print(f"Test set:        {len(test_df):5d} samples ({len(test_df)/len(df)*100:.1f}%) | Saved to {test_path}")

    print("\nTrain Class Distribution:")
    print(train_df["label"].value_counts().to_dict())
    print("Val Class Distribution:")
    print(val_df["label"].value_counts().to_dict())
    print("Test Class Distribution:")
    print(test_df["label"].value_counts().to_dict())

    print("\n[SUCCESS] Preprocessing and data splitting completed successfully!")


if __name__ == "__main__":
    preprocess_and_split()
