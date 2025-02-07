import os
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from datetime import datetime, timedelta
from tqdm import tqdm
from typing import List, Tuple
import numpy as np
from dotenv import load_dotenv
from newsapi import NewsApiClient
import newspaper
import pandas as pd


load_dotenv()


def get_news_articles(
    query,
    days_back=7,  # Changed to relative date
    language="en",
    num_pages=2,
    sort_by="relevancy",
):
    newsapi = NewsApiClient(api_key=os.getenv("API_KEY"))

    # Calculate dates relative to current time
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days_back)

    # Format dates for API
    end_date_str = end_date.strftime("%Y-%m-%d")
    start_date_str = start_date.strftime("%Y-%m-%d")

    return newsapi.get_everything(
        q=query,
        from_param=start_date_str,
        to=end_date_str,
        language=language,
        sort_by=sort_by,
        page=num_pages,
    )


def get_article_content(url) -> Tuple[str, List[str]]:
    try:
        article = newspaper.Article(
            url,
            browser_user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
            request_timeout=10,
        )
        article.download()
        article.parse()
        article.nlp()
        return article.text, article.keywords
    except Exception:
        return None, []


def process_single_article(article):
    article_url = article["url"]
    article_text, article_keywords = get_article_content(article_url)
    if not article_text:
        return None
    return {
        "title": article["title"],
        "url": article_url,
        "article": article_text,
        "keywords": article_keywords,
        "author": article["author"] or "Unknown",
        "publish_date": datetime.strptime(
            article["publishedAt"], "%Y-%m-%dT%H:%M:%SZ"
        ).strftime("%Y-%m-%d")
        or "Unknown",
    }


def parse_article_dict(article_dict, top_n=None):
    parsed_articles = []
    n_articles = len(article_dict["articles"])

    # Determine the number of articles to process
    if top_n is None or top_n < 0 or top_n > n_articles:
        articles_to_process = article_dict["articles"]
    else:
        articles_to_process = article_dict["articles"][:top_n]

    # Parallel processing
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = []
        for article in articles_to_process:
            futures.append(executor.submit(process_single_article, article))

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Processing articles",
        ):
            try:
                result = future.result()
                if result:  # Filter out failed downloads
                    parsed_articles.append(result)

            except Exception as e:
                print(f"Error processing article: {e}")

    return parsed_articles


def main():
    article_dict = get_news_articles(
        query="bitcoin",
        days_back=7,  # Get articles from last week
        language="en",
        sort_by="relevancy",
        num_pages=5,
    )

    parsed_articles = parse_article_dict(article_dict)
    df = pd.DataFrame(parsed_articles)
    article_len = df["article"].apply(len)
    df = df[article_len > article_len.quantile(0.25)]
    df.drop_duplicates(subset=["article"], inplace=True)

    filename = f"CSVs/news_articles_{datetime.now().strftime('%Y-%m-%d')}.csv"
    df.to_csv(filename, index=False)
    print(f"Saved {len(parsed_articles)} articles to {filename}")


if __name__ == "__main__":
    main()
