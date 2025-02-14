# Crypto News Analysis and Trading Assistant

This project is a comprehensive system for scraping, analyzing, and providing insights on cryptocurrency news, with a focus on Bitcoin trading advice.

The application combines web scraping, natural language processing, and machine learning techniques to gather relevant news articles, process them, and generate trading insights. It uses a vector search engine to efficiently retrieve relevant articles based on user queries and leverages a language model to provide personalized financial advice.

## Repository Structure

```
.
├── app.py
├── clean_article.py
├── news_embeddings.py
├── news_scraping.py
├── templates
│   ├── __init__.py
│   ├── labelling_templates.py
│   └── prompt_templates.py
└── utils
    ├── __init__.py
    ├── get_bitoin_prices.py
    └── text_cleaning.py
```

### Key Files:

- `app.py`: The main application entry point, handling user interactions and generating responses.
- `news_scraping.py`: Responsible for fetching and processing news articles from NewsAPI.
- `clean_article.py`: Preprocesses the scraped news articles, removing HTML tags and normalizing text.
- `news_embeddings.py`: Generates embeddings for news articles and stores them in the Qdrant vector search engine.

### Important Integration Points:

- NewsAPI: Used for fetching recent cryptocurrency news articles.
- Qdrant: Vector search engine for storing and retrieving article embeddings.
- Ollama: Used for generating embeddings and powering the language model for responses.

## Usage Instructions

### Installation

Prerequisites:
- Python 3.7+
- pip (Python package manager)

Steps:
1. Clone the repository:
   ```
   git clone <repository_url>
   cd <repository_name>
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file in the project root and add the following:
   ```
   API_KEY=<your_newsapi_key>
   QDRANT_ENDPOINT=<your_qdrant_endpoint>
   QDRANT_API_KEY=<your_qdrant_api_key>
   ```

### Getting Started

1. Scrape news articles:
   ```
   python news_scraping.py
   ```

2. Clean and preprocess the articles:
   ```
   python clean_article.py CSVs/news_articles_<date>.csv
   ```

3. Generate embeddings and store in Qdrant:
   ```
   python news_embeddings.py CSVs/news_articles_<date>.csv
   ```

4. Run the main application:
   ```
   python app.py
   ```

### Configuration Options

- Adjust the `MODEL_NAME` in `app.py` to use different Ollama models.
- Modify the `query`, `days_back`, and other parameters in `news_scraping.py` to customize the news fetching process.

### Common Use Cases

1. Fetching recent Bitcoin news:
   ```python
   from news_scraping import get_news_articles
   
   articles = get_news_articles(query="bitcoin", days_back=7, language="en")
   ```

2. Generating embeddings for an article:
   ```python
   from app import generate_embedding
   
   embedding = generate_embedding("Bitcoin price surges to new all-time high")
   ```

3. Querying the chatbot:
   ```python
   user_input = "What's the current market sentiment for Bitcoin?"
   response = chatbot({"messages": [("user", user_input)]})
   print(response["messages"][-1]["content"])
   ```

### Testing & Quality

To ensure the quality of the system:
1. Regularly update the news database to maintain relevance.
2. Monitor the performance of the embedding model and Qdrant search results.
3. Periodically review and refine the prompt templates for improved responses.

### Troubleshooting

1. Issue: No results from news scraping
   - Check your NewsAPI key in the `.env` file
   - Ensure you have an active internet connection
   - Verify the query parameters in `news_scraping.py`

2. Issue: Embedding generation fails
   - Ensure Ollama is properly installed and running
   - Check if the specified model is available in your Ollama installation

3. Issue: Qdrant connection errors
   - Verify the Qdrant endpoint and API key in the `.env` file
   - Ensure your Qdrant instance is running and accessible

For debugging:
- Enable verbose logging in `app.py` by adding:
  ```python
  import logging
  logging.basicConfig(level=logging.DEBUG)
  ```
- Check the application logs for detailed error messages and stack traces.

## Data Flow

The data flow in this application follows these steps:

1. News articles are scraped from NewsAPI using `news_scraping.py`.
2. The articles are cleaned and preprocessed using `clean_article.py`.
3. Article embeddings are generated and stored in Qdrant using `news_embeddings.py`.
4. When a user query is received in `app.py`:
   a. Relevant articles are retrieved from Qdrant using the query embedding.
   b. Bitcoin price data is fetched for the relevant time period.
   c. The retrieved articles and price data are used to generate a context for the language model.
   d. The language model processes the context and user query to generate a response.

```
[User Query] -> [App] -> [Qdrant Search] -> [Article Retrieval]
                  |                               |
                  v                               v
           [Bitcoin Price Data] <- [Context Generation] -> [Language Model]
                                          |
                                          v
                                   [Response to User]
```

Note: The embedding generation and storage processes are performed offline, prior to user interactions.