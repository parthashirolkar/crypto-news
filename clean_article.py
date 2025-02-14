from utils.text_cleaning import clean_html, normalize_text, remove_boilerplate
import argparse
import pandas as pd
import os


def preprocess_article(article_text):
    """Applies all preprocessing steps to a given article text."""
    text = clean_html(article_text)
    text = normalize_text(text)
    text = remove_boilerplate(text)
    return text


def main():
    parser = argparse.ArgumentParser(description="Preprocess news articles.")
    parser.add_argument(
        "file_path", help="Path to the CSV file containing news articles"
    )
    args = parser.parse_args()

    if not os.path.exists(args.file_path):
        print(f"File not found: {args.file_path}")
        return

    df = pd.read_csv(args.file_path)
    article_len = df["article"].apply(len)
    df = df[article_len > article_len.quantile(0.25)].copy(deep=True)

    df["article"] = df["article"].apply(preprocess_article)
    df.to_csv(args.file_path, index=False)

    print("Done.")


if __name__ == "__main__":
    main()
