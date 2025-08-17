# Enhanced Retrieval-Augmented Generation (RAG) System

Advanced document Q&A with semantic search, reranking, and hybrid retrieval.

## Features

- Hybrid dense/sparse retrieval
- Semantic chunking
- Query expansion
- Reranking
- Document compression
- GPU/CPU support

## Quick Start

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/SimpleRAG.git
   cd SimpleRAG/pythonProject
   ```

2. **Install dependencies**

   ```bash
   poetry install
   ```

3. **Configure environment variables**

   Create a `.env` file in the project root (see below for example).

   ```
   HF_TOKEN=your_huggingface_token
   MODEL_NAME=google/gemma-2b-it
   EMBEDDING_MODEL_NAME=BAAI/bge-small-en-v1.5
   RERANKER_MODEL_NAME=BAAI/bge-reranker-base
   DOC_DIR=docs
   CACHE_DIR=.cache
   ```

   - `HF_TOKEN` is required for some Hugging Face models. Get it from [Hugging Face](https://huggingface.co/settings/tokens).
   - Adjust other variables as needed.

4. **Add your documents**

   Place `.txt`, `.md`, or `.pdf` files in the `docs` directory.

5. **Run the app**

   ```bash
   streamlit run app.py
   ```

## .env Example

```
HF_TOKEN=your_huggingface_token
MODEL_NAME=google/gemma-2b-it
EMBEDDING_MODEL_NAME=BAAI/bge-small-en-v1.5
RERANKER_MODEL_NAME=BAAI/bge-reranker-base
DOC_DIR=docs
CACHE_DIR=.cache
```

## Usage

- Enter your question in the app UI.
- Adjust retrieval and generation settings in the sidebar.
- View sources and context for each answer.

## Notes

- For best results, use high-quality documents.
- GPU is recommended for faster inference.
- All configuration is managed via `.env`.

## License

MIT

## Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain)
- [Hugging Face](https://huggingface.co/)
- [Streamlit](https://streamlit.io/)

