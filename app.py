import os
from utils.get_bitoin_prices import get_data
from typing import List
import warnings
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langgraph.graph import START, StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from langchain_community.embeddings import OllamaEmbeddings

from qdrant_client import QdrantClient
from rich.console import Console
from rich.text import Text

warnings.simplefilter(action="ignore")
load_dotenv()


class State(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: State):
    """Main chatbot function to fetch relevant articles and generate responses."""
    user_message = state["messages"][-1].content

    relevant_articles = search_articles_qdrant(user_message, top_n=3)

    dates = sorted([article["publish_date"] for article in relevant_articles])
    btc_data = get_data(dates[-1])

    context = "\n".join(
        [
            f"Title: {a['title']}\nPublish Date: {a['publish_date']}\nArticle: {a['article']}"
            for a in relevant_articles
        ]
    )

    modified_message = f"Relevant News:\n{context}\n Historical Bitcoin data: {btc_data.to_string()}\n\nUser Query: {user_message}"

    response = llm.invoke(modified_message)

    return {"messages": [{"role": "assistant", "content": response.content}]}


console = Console()
MODEL_NAME = "deepseek-r1:8b"
collection_name = "crypto_news"
llm = ChatOllama(model=MODEL_NAME, temperature=0.0, keep_alive=-1)

client = QdrantClient(
    url=os.environ.get("QDRANT_ENDPOINT"), api_key=os.environ.get("QDRANT_API_KEY")
)
embedding_function = OllamaEmbeddings(model=MODEL_NAME)

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

def search_articles_qdrant(query, top_n=5):
    """Search for relevant news articles using Qdrant vector search."""
    query_vector = embedding_function.embed_query(query)

    search_result = client.search(
        collection_name=collection_name, query_vector=query_vector, limit=top_n
    )

    return [hit.payload for hit in search_result]


while True:
    user_input = input("\033[1;34mUser:\033[0m ")
    if user_input.lower() in ["quit", "q"]:
        console.print("[bold red]Goodbye![/bold red]")
        break
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            last_message = value["messages"][-1]
            assistant_message = last_message.get("content", "No response generated.")
            console.print(
                Text("Assistant: ", style="bold green")
                + Text(assistant_message, style="green")
            )
