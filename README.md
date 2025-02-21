# AI-Powered Crypto News Analysis and Trading Assistant

This project provides an intelligent chatbot that analyzes cryptocurrency news articles and Bitcoin price data to offer personalized trading insights and market sentiment analysis. The system combines natural language processing, vector search, and real-time market data to deliver informed financial advice to crypto trading novices.

The application scrapes and processes cryptocurrency news articles, generates semantic embeddings for efficient retrieval, and uses the ChatOllama language model to provide contextual financial analysis. By analyzing both historical price data and recent news sentiment, it helps users make more informed trading decisions with a comprehensive view of market dynamics.

## Repository Structure
```
.
├── app.py                      # Main Streamlit application entry point
├── clean_article.py            # Article text preprocessing and cleaning
├── news_embeddings.py          # Generates and stores article embeddings in Qdrant
├── news_scraping.py           # Fetches and processes news articles from NewsAPI
├── templates/                  # Template definitions for prompts and responses
│   ├── labelling_templates.py  # Templates for data labeling
│   └── prompt_templates.py     # Templates for chat interactions
└── utils/                     # Utility functions and helpers
    ├── get_bitoin_prices.py   # Bitcoin price data retrieval
    └── text_cleaning.py       # Text preprocessing utilities
```

## Usage Instructions
### Prerequisites
- Python 3.7+
- Ollama installed locally
- NewsAPI API key
- Qdrant cloud account and API key
- Environment variables configured in `.env`:
  - `API_KEY`: NewsAPI key
  - `QDRANT_ENDPOINT`: Qdrant cloud endpoint
  - `QDRANT_API_KEY`: Qdrant API key

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd <repository-name>
```

### Quick Start
1. Fetch and process news articles:
```bash
python news_scraping.py
```

2. Clean the articles:
```bash
python clean_article.py CSVs/news_articles_<date>.csv
```

3. Generate embeddings:
```bash
python news_embeddings.py CSVs/news_articles_<date>.csv
```

4. Run the Streamlit application:
```bash
streamlit run app.py
```

### More Detailed Examples
1. Customizing news scraping:
```python
from news_scraping import get_news_articles

articles = get_news_articles(
    query="bitcoin",
    days_back=14,  # Get two weeks of articles
    language="en",
    num_pages=5,
    sort_by="relevancy"
)
```

2. Using the chatbot:
```python
# In the Streamlit interface
# Example questions:
"What's the current market sentiment for Bitcoin based on recent news?"
"Should I invest in Bitcoin right now given the recent market conditions?"
"What are the key factors affecting Bitcoin's price this week?"
```

### Troubleshooting
Common Issues:
1. Embedding Generation Failures
   - Error: "Unexpected vector dimension"
   - Solution: Verify Ollama model is running and accessible
   - Debug: `ollama list` to check available models

2. News API Rate Limits
   - Error: "Rate limit exceeded"
   - Solution: Implement delay between requests
   - Debug: Check API usage in NewsAPI dashboard

3. Qdrant Connection Issues
   - Error: "Failed to connect to Qdrant"
   - Solution: Verify endpoint and API key in .env
   - Debug: Test connection using Qdrant's health check endpoint

## Data Flow
The system processes data through a pipeline of news collection, preprocessing, embedding generation, and interactive querying.

```ascii
[NewsAPI] -> [news_scraping.py] -> [clean_article.py] -> [news_embeddings.py] -> [Qdrant]
                                                                                    |
[User Query] -> [app.py] -> [Vector Search] -> [Context Assembly] -> [LLM] -> [Response]
                   ^                                                              |
                   |                                                             |
                   +---------------------[Chat History]--------------------------->
```

Key Component Interactions:
1. News scraper fetches articles using NewsAPI and processes them with Newspaper3k
2. Article cleaner removes HTML and standardizes text format
3. Embedding generator creates vector representations using OllamaEmbeddings
4. Qdrant stores and indexes embeddings for efficient similarity search
5. Streamlit app handles user interaction and maintains chat context
6. ChatOllama model generates responses based on retrieved context
7. Bitcoin price data is integrated with news context for comprehensive analysis
8. Vector search retrieves relevant article chunks based on user queries