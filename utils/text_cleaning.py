import re
from bs4 import BeautifulSoup


def clean_html(text):
    """Remove HTML tags from text."""
    return BeautifulSoup(text, "html.parser").get_text()


def normalize_text(text):
    """Perform text normalization (lowercase, whitespace cleanup, punctuation standardization)."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    text = (
        text.replace("’", "'").replace("“", '"').replace("”", '"')
    )
    return text


def remove_boilerplate(text):
    """Remove common boilerplate patterns like copyright, social media links, and ads."""
    patterns = [
        r"\b(Copyright|©)\s*\d{4}.*",
        r"http[s]?://\S+",
        r"(Follow us on|Subscribe|Sign up for|Read more at).*",
    ]
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return text.strip()
