from pathlib import Path
import pandas as pd

def preprocess_reviews(input_path: Path, output_path: Path):
    # Load CSV
    df = pd.read_csv(input_path)

    # Basic cleaning steps (modify as needed)
    df.drop_duplicates(inplace=True)
    df.dropna(subset=["content"], inplace=True)
    df["content"] = df["content"].str.strip()

    # Add source column
    df["source"] = "Google Play"

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save cleaned CSV
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned data to {output_path} — {len(df)} rows")

if __name__ == "__main__":
    preprocess_reviews(
        input_path=Path("data/raw/ethiopian_bank_reviews.csv"),
        output_path=Path("data/clean/ethiopian_bank_reviews_cleaned.csv")
    )
