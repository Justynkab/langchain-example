# LangChain Ollama Example

A minimal Python project that uses LangChain with an Ollama chat model to:

1. Summarize a long biography.
2. Extract two interesting facts.

## Overview

The script in `main.py` builds a prompt with `PromptTemplate`, sends it to an Ollama-backed model through `ChatOllama`, and prints the generated response.

## Tech Stack

- Python 3.12+
- [LangChain](https://python.langchain.com/)
- [langchain-ollama](https://python.langchain.com/docs/integrations/chat/ollama/)
- [uv](https://docs.astral.sh/uv/) for dependency and environment management

## Prerequisites

- Python `3.12` (see `.python-version`)
- `uv` installed
- Access to an Ollama-compatible model (default in this repo: `gpt-oss:20b-cloud`)

If you are running Ollama locally, make sure the Ollama service is running and the model is available before executing the script.

## Setup

Install dependencies:

```bash
uv sync
```

## Run

```bash
uv run python main.py
```

## Configuration

The default model is set in `main.py`:

```python
llm = ChatOllama(temperature=0, model="gpt-oss:20b-cloud")
```

You can change:

- `model` to any available Ollama model.
- `temperature` to control response creativity.
- The prompt template to adjust output format.

## Project Structure

```text
.
|-- main.py
|-- pyproject.toml
|-- uv.lock
`-- README.md
```

## Troubleshooting

- `Model not found`: Verify the model name in `main.py` and that it exists in your Ollama environment.
- Connection errors: Ensure Ollama is running and reachable from your terminal session.
