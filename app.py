
import os
import time
import logging

import requests
import streamlit as st
from ddgs import DDGS


# --------------------------------
# 1. Configure logging
# --------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


# --------------------------------
# 2. Configure Streamlit page
# --------------------------------

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔎",
    layout="centered",
)

st.title("AI Research Assistant")

st.write(
    "Enter a topic to search the web and generate "
    "a three-point research brief using AI."
)


# --------------------------------
# 3. Search the web
# --------------------------------

def search_web(topic):
    """Search the web using DuckDuckGo."""

    results = DDGS().text(
        topic,
        max_results=5,
        backend="auto",
    )

    return list(results or [])


# --------------------------------
# 4. Generate AI research brief
# --------------------------------

def summarize_topic(topic, results):
    """Generate a research brief using Hugging Face."""

    if not results:
        raise ValueError("No search results were returned.")

    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        raise ValueError(
            "HF_TOKEN environment variable is missing."
        )

    sources = "\n\n".join(
        f"Title: {item.get('title', '')}\n"
        f"URL: {item.get('href', '')}\n"
        f"Snippet: {item.get('body', '')}"
        for item in results
    )

    response = requests.post(
        "https://router.huggingface.co/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {hf_token}",
            "Content-Type": "application/json",
        },
        json={
            "model": "openai/gpt-oss-120b:fastest",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an AI research assistant. "
                        "Use only the supplied search results. "
                        "Generate exactly three clear and concise "
                        "bullet points. Do not invent facts."
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

    return data["choices"][0]["message"]["content"]


# --------------------------------
# 5. Display search results
# --------------------------------

def display_search_results(results):
    """Display original web search results."""

    st.subheader("Web Search Results")

    if not results:
        st.warning("No search results found.")
        return

    for index, item in enumerate(results, start=1):

        title = item.get("title", "No title")
        description = item.get(
            "body",
            "No description available.",
        )
        url = item.get("href", "")

        st.markdown(f"**{index}. {title}**")

        st.write(description)

        if url:
            st.markdown(f"[Read source]({url})")

        st.divider()


# --------------------------------
# 6. Main Streamlit application
# --------------------------------

topic = st.text_input(
    "Enter your research topic",
    placeholder="Example: Artificial Intelligence in Healthcare",
)

if st.button("Generate Research Brief", type="primary"):

    if not topic.strip():

        st.warning("Please enter a research topic.")

    else:

        start_time = time.time()

        try:

            # Search the web
            with st.spinner("Searching the web..."):

                results = search_web(topic)

            logger.info(
                "Search completed | Topic: %s | Results: %d",
                topic,
                len(results),
            )

            # Display original search results
            display_search_results(results)

            # Generate AI brief
            with st.spinner(
                "Generating your three-point research brief..."
            ):

                brief = summarize_topic(topic, results)

            # Display AI-generated output
            st.subheader("Three-Point Research Brief")

            st.markdown(brief)

            # Calculate request duration
            duration = time.time() - start_time

            st.success("Research brief generated successfully!")

            st.caption(
                f"Total request duration: {duration:.2f} seconds"
            )

            logger.info(
                "Request completed | Topic: %s | Duration: %.2f seconds",
                topic,
                duration,
            )

        except requests.exceptions.HTTPError as e:

            logger.exception("Hugging Face API error.")

            st.error(
                f"Hugging Face API error: {e.response.status_code}"
            )

            st.write(e.response.text[:1000])

        except Exception as e:

            logger.exception("Research request failed.")

            st.error(f"An error occurred: {str(e)}")
