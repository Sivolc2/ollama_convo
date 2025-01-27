# Ollama Local Chat Interface

This project provides a simple Python interface to chat with locally hosted language models using Ollama.

## Prerequisites

1. Install Ollama on your system:
   - Visit [Ollama's website](https://ollama.ai/) and follow the installation instructions for your OS
   - For Linux: `curl https://ollama.ai/install.sh | sh`

2. Pull a model (e.g., llama2):
   ```bash
   ollama pull llama2
   ```

## Setup

1. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/Mac
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the chat interface:
   ```bash
   python src/chat_with_model.py
   ```

2. Enter your messages when prompted. Type 'quit' to exit.

## Changing Models

To use a different model:
1. Pull the desired model using Ollama:
   ```bash
   ollama pull mistral  # or any other supported model
   ```
2. Modify the `model` variable in `src/chat_with_model.py` to use your preferred model.

## Available Models

You can find a list of available models at [Ollama's model library](https://ollama.ai/library). 