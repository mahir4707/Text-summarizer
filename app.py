from transformers import pipeline

# Load summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Example text
text = """
LangChain is an open-source framework designed to simplify the creation of applications using large language models (LLMs). It provides a standard interface for integrating with other tools and end-to-end chains for common applications. It helps AI developers connect LLMs such as GPT-4 with external data and computation. This framework comes for both Python and JavaScript.

Key benefits include:

Modular Workflow: Simplifies chaining LLMs together for reusable and efficient workflows.
Prompt Management: Offers tools for effective prompt engineering and memory handling.
Ease of Integration: Streamlines the process of building LLM-powered applications.
"""

# Summarize
summary = summarizer(text, max_length=50, min_length=25, do_sample=False)

print("Summary:", summary[0]['summary_text'])
