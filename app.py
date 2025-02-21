import os
from collections import defaultdict
import re
from utils.get_bitoin_prices import get_data
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.schema import HumanMessage, AIMessage, SystemMessage
import streamlit as st
from qdrant_client import QdrantClient
from datetime import datetime


load_dotenv()
SYSTEM_PROMPT = """You are a knowledgeable financial advisor specializing in cryptocurrency markets, 
particularly Bitcoin. Use the provided news articles and price data to give informed advice and analysis. 
Always consider the article dates, their contents and recent price trends in your responses. Be factual and analytical 
while maintaining a professional tone.
"""

MODEL_NAME = "deepseek-r1:8b"
llm = ChatOllama(model=MODEL_NAME, temperature=0.0, keep_alive=-1)
collection_name = "crypto_news_chunked"
client = QdrantClient(
    url=os.environ.get("QDRANT_ENDPOINT"), api_key=os.environ.get("QDRANT_API_KEY")
)
embedding_function = OllamaEmbeddings(model=MODEL_NAME)


def search_articles_qdrant(query, top_n=10):
    """Search for relevant news article chunks using Qdrant vector search."""
    enhanced_query = f"Find articles relevant to: {query}"
    query_vector = embedding_function.embed_query(enhanced_query)

    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_n,
    )

    article_chunks = defaultdict(list)

    for hit in search_result:
        payload = hit.payload  # Extract payload

        # Use .get() to avoid KeyError and ensure the key exists
        article_id = payload.get("article_id")
        if not article_id:
            continue  # Skip this entry

        chunk_index = payload.get("chunk_index", 0)  # Default to 0 if missing

        article_chunks[article_id].append((payload, hit.score))

    return article_chunks


def format_context(article_chunks, bitcoin_data=None):
    """Format retrieved chunks into reconstructed articles and add Bitcoin price data."""

    context = "Here are the relevant articles and data:\n\n"

    # Format reconstructed articles
    for _, chunks in article_chunks.items():
        dates = []
        if not chunks:
            continue

        first_chunk = chunks[0][0]  # Metadata from the first chunk
        context += f"Date: {first_chunk['publish_date']}\n"
        context += f"Headline: {first_chunk['title']}\n"
        context += "Content:\n"

        # Merge chunks in order
        for chunk, _ in chunks:
            context += chunk["chunk"] + " "

        context += "\n\n"
        

        # Ensure the date is in the correct format
        publish_date = first_chunk["publish_date"]
        if isinstance(publish_date, str):
            # Convert to datetime object if necessary
            publish_date = datetime.strptime(
                publish_date, "%Y-%m-%d %H:%M:%S"
            ).strftime("%Y-%m-%dT%H:%M:%S")
            dates.append(publish_date)

    bitcoin_data = get_data(max(dates))

    # Add Bitcoin price data if available
    if bitcoin_data is not None:
        context += "\nRecent Bitcoin Price Data:\n"
        context += f"{bitcoin_data.to_string()}\n\n"

    return context


if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append(SystemMessage(SYSTEM_PROMPT))

for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)

    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)


prompt = st.chat_input("How are you?")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
        st.session_state.messages.append(HumanMessage(prompt))

    db_results = search_articles_qdrant(prompt, top_n=10)

    # Display article titles and similarity scores in an expander
    with st.expander("📰 Articles used for context"):
        if not db_results:
            st.markdown("No relevant articles found.")
        else:
            for article_id, chunks in db_results.items():
                for article, score in chunks:
                    st.markdown(
                        f"- **{article['title']}** ({article['publish_date']}) - Similarity Score: {score:.2f}"
                    )

    st.session_state.messages.append(SystemMessage(SYSTEM_PROMPT))

    formatted_context = format_context(db_results)
    st.session_state.messages.append(HumanMessage(formatted_context))

    result = llm.invoke(st.session_state.messages).content

    # Use regex to extract chain of thought reasoning
    think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    chain_of_thought_matches = think_pattern.findall(result)

    # Remove chain of thought from the result
    result = think_pattern.sub("", result).strip()

    with st.chat_message("assistant"):
        st.markdown(result)

        # Display each chain of thought in an expander
        for i, chain_of_thought in enumerate(chain_of_thought_matches, start=1):
            with st.expander(f"Chain of Thought {i}"):
                st.markdown(chain_of_thought.strip())

        st.session_state.messages.append(AIMessage(result))
