
import time
import logging

import requests
import streamlit as st
from ddgs import DDGS


# -------------------------------
# 1. Configure logging
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# -------------------------------
# 2. Configure Streamlit page
# -------------------------------
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔎",
    layout="centered",
)

st.title("AI Research Assistant")
st.write(
    "Enter a topic to get a three-point research brief "
    "based on current web search results."
)


# -------------------------------
# 3. Search the web
# -------------------------------
def search_web(topic):
    """Search the web and return relevant results."""
    results = DDGS().text(
        topic,
        max_results=5,
        backend="auto",
    )
    return list(results or [])


# -------------------------------
# 4. Summarize using Ollama
# -------------------------------
def summarize_topic(topic, results):
    """Generate a three-point brief using the local Ollama model."""

    if not results:
        raise ValueError("No search results were returned.")

    sources = "\n\n".join(
        f"Title: {item.get('title', '')}\n"
        f"URL: {item.get('href', '')}\n"
        f"Snippet: {item.get('body', '')}"
        for item in results
    )

    response = requests.post(
       "http://host.docker.internal:11434/api/chat",
        json={
            "model": "llama3.2:3b",
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a research assistant. Use only the "
                        "provided search results to produce exactly "
                        "three concise bullet points. Be clear and "
                        "factual. Do not invent facts. Mention when "
                        "the sources do not provide enough information."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Research topic: {topic}\n\n"
                        f"Search results:\n{sources}"
                    ),
                },
            ],
        },
        timeout=180,
    )

    response.raise_for_status()
    data = response.json()

    return data["message"]["content"]


# -------------------------------
# 5. User input
# -------------------------------
topic = st.text_input(
    "Enter a research topic",
    placeholder="e.g., Artificial Intelligence in Healthcare",
)

generate_button = st.button(
    "Generate Research Brief",
    type="primary",
)


# -------------------------------
# 6. Generate and display brief
# -------------------------------
if generate_button:
    if not topic.strip():
        st.warning("Please enter a research topic.")

    else:
        start_time = time.time()
        logger.info("Research request received: %s", topic)

        try:
            with st.spinner(
                "Searching the web and generating your brief..."
            ):
                results = search_web(topic)
                summary = summarize_topic(topic, results)

            st.subheader("Research Brief")
            st.markdown(summary)

            st.subheader("Sources")

            for item in results:
                title = item.get("title") or "Web source"
                url = item.get("href", "")

                if url:
                    st.markdown(f"- [{title}]({url})")
                else:
                    st.markdown(f"- {title}")

            duration = time.time() - start_time

            logger.info(
                "Research request completed in %.2f seconds",
                duration,
            )

            st.caption(
                f"Request duration: {duration:.2f} seconds"
            )

        except requests.exceptions.ConnectionError:
            duration = time.time() - start_time
            logger.exception("Could not connect to Ollama")

            st.error(
                "Could not connect to Ollama. Make sure Ollama "
                "is installed and running on your Mac."
            )
            st.caption(f"Failed request duration: {duration:.2f} seconds")

        except Exception as error:
            duration = time.time() - start_time

            logger.exception("Research request failed")

            st.error(f"Something went wrong: {error}")
            st.caption(f"Failed request duration: {duration:.2f} seconds")
