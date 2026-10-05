import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import ensure_directories, FINAL_DATASET_PATH, RESULTS_DIR, LABEL_MAP

ensure_directories()


def analyze_dataset():
    if not os.path.exists(FINAL_DATASET_PATH):
        raise FileNotFoundError(
            f"Final dataset not found at {FINAL_DATASET_PATH}. Run 'python src/preprocess.py' first."
        )

    print(f"[INFO] Reading final dataset from: {FINAL_DATASET_PATH}\n")
    df = pd.read_csv(FINAL_DATASET_PATH)

    # 1. Basic Structural Information
    total_rows = len(df)
    total_cols = len(df.columns)
    col_names = list(df.columns)
    missing_vals = df.isnull().sum().to_dict()
    total_missing = df.isnull().sum().sum()
    duplicate_rows = df.duplicated().sum()
    duplicate_texts = df["text"].duplicated().sum()
    empty_texts = (df["text"].str.strip() == "").sum()

    # 2. Distribution Statistics
    class_counts = df["label"].value_counts().sort_index().to_dict()
    outlier_counts = df["is_outlier"].value_counts().sort_index().to_dict()

    # 3. Text Length Statistics
    df["char_length"] = df["text"].astype(str).str.len()
    df["word_count"] = df["text"].astype(str).apply(lambda x: len(x.split()))

    min_char_len = df["char_length"].min()
    max_char_len = df["char_length"].max()
    avg_char_len = df["char_length"].mean()
    median_char_len = df["char_length"].median()

    min_words = df["word_count"].min()
    max_words = df["word_count"].max()
    avg_words = df["word_count"].mean()

    # 4. Print Structured CLI Summary
    print("=" * 60)
    print("           DATA QUALITY & STATISTICAL ANALYSIS")
    print("=" * 60)
    print(f"Total Rows:                 {total_rows}")
    print(f"Total Columns:              {total_cols} {col_names}")
    print(f"Total Missing Values:       {total_missing} ({missing_vals})")
    print(f"Duplicate Full Rows:        {duplicate_rows}")
    print(f"Duplicate Text Entries:     {duplicate_texts}")
    print(f"Empty / Whitespace Texts:   {empty_texts}")
    print("-" * 60)
    print("Class Distribution:")
    for lbl, count in class_counts.items():
        name = LABEL_MAP.get(lbl, f"Class {lbl}")
        pct = (count / total_rows) * 100
        print(f"  [{lbl}] {name:<10}: {count:5d} ({pct:.2f}%)")
    print("-" * 60)
    print("Outlier Distribution:")
    for out, count in outlier_counts.items():
        out_name = "Regular Sample" if out == 0 else "Intentionally Noisy Outlier"
        pct = (count / total_rows) * 100
        print(f"  [{out}] {out_name:<28}: {count:5d} ({pct:.2f}%)")
    print("-" * 60)
    print("Text Length Metrics (Characters):")
    print(f"  Minimum Length:           {min_char_len} chars")
    print(f"  Maximum Length:           {max_char_len} chars")
    print(f"  Average Length:           {avg_char_len:.2f} chars")
    print(f"  Median Length:            {median_char_len:.2f} chars")
    print("Word Count Metrics:")
    print(f"  Minimum Words:            {min_words} words")
    print(f"  Maximum Words:            {max_words} words")
    print(f"  Average Words:            {avg_words:.2f} words")
    print("=" * 60 + "\n")

    # 5. Generate Visualizations

    # Figure 1: Class Distribution
    fig, ax = plt.subplots(figsize=(6, 5))
    classes = [LABEL_MAP[0], LABEL_MAP[1]]
    counts = [class_counts.get(0, 0), class_counts.get(1, 0)]
    colors = ["#2b5c8f", "#d9534f"]
    bars = ax.bar(classes, counts, color=colors, width=0.5, edgecolor="black", linewidth=1.2)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 50, f"{yval:,}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title("Class Distribution (Normal vs. Threat)", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_ylim(0, max(counts) * 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    class_dist_path = os.path.join(RESULTS_DIR, "class_distribution.png")
    plt.savefig(class_dist_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved class distribution plot to: {class_dist_path}")

    # Figure 2: Outlier Distribution
    fig, ax = plt.subplots(figsize=(6, 5))
    outlier_labels = ["Regular (0)", "Noisy Outlier (1)"]
    outlier_vals = [outlier_counts.get(0, 0), outlier_counts.get(1, 0)]
    outlier_colors = ["#5cb85c", "#f0ad4e"]
    bars = ax.bar(outlier_labels, outlier_vals, color=outlier_colors, width=0.5, edgecolor="black", linewidth=1.2)
    for bar in bars:
        yval = bar.get_height()
        pct = (yval / total_rows) * 100
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 100, f"{yval:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Dataset Outlier Distribution", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_ylim(0, max(outlier_vals) * 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    outlier_dist_path = os.path.join(RESULTS_DIR, "outlier_distribution.png")
    plt.savefig(outlier_dist_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved outlier distribution plot to: {outlier_dist_path}")

    # Figure 3: Text Length Distribution (Character & Word Counts)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Subplot 1: Character Length Histogram
    axes[0].hist(df["char_length"], bins=35, color="#17a2b8", edgecolor="black", alpha=0.8)
    axes[0].axvline(avg_char_len, color="red", linestyle="dashed", linewidth=1.5, label=f"Mean ({avg_char_len:.1f})")
    axes[0].set_title("Character Length Distribution", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Character Count", fontsize=10)
    axes[0].set_ylabel("Frequency", fontsize=10)
    axes[0].legend()
    axes[0].grid(axis="y", linestyle="--", alpha=0.6)

    # Subplot 2: Word Count Histogram by Class
    normal_words = df[df["label"] == 0]["word_count"]
    threat_words = df[df["label"] == 1]["word_count"]
    axes[1].hist(normal_words, bins=30, alpha=0.6, label="Normal", color="#2b5c8f", edgecolor="black")
    axes[1].hist(threat_words, bins=30, alpha=0.6, label="Threat", color="#d9534f", edgecolor="black")
    axes[1].set_title("Word Count Distribution by Class", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Word Count", fontsize=10)
    axes[1].set_ylabel("Frequency", fontsize=10)
    axes[1].legend()
    axes[1].grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    text_len_dist_path = os.path.join(RESULTS_DIR, "text_length_distribution.png")
    plt.savefig(text_len_dist_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved text length distribution plot to: {text_len_dist_path}")
    print("\n[SUCCESS] Dataset analysis completed!")


if __name__ == "__main__":
    analyze_dataset()
