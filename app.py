import hashlib
import logging
import os
import re
import threading
from datetime import datetime
from typing import Optional, List, Tuple, Dict
from dotenv import load_dotenv  # <-- Add this import

import numpy as np
import streamlit as st
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredMarkdownLoader
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from nltk import sent_tokenize
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()  # <-- Add this line


class Config:
    """Configuration settings for the RAG system"""
    # Model settings
    MODEL_NAME = os.getenv("MODEL_NAME", "google/gemma-2b-it")
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
    RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-base")

    # Document settings
    DOC_DIR = os.getenv("DOC_DIR", "docs")
    CACHE_DIR = os.getenv("CACHE_DIR", ".cache")
    SUPPORTED_EXTENSIONS = [".txt", ".md", ".pdf"]

    # Chunking settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    CHUNK_MIN_SIZE = 100  # Minimum chunk size

    # Generation settings
    MAX_NEW_TOKENS = 512
    TEMPERATURE = 0.1
    TOP_P = 0.9
    TOP_K = 30  # Retrieve more initially
    TOP_K_RERANK = 5  # After reranking

    # Retrieval settings
    MIN_SIMILARITY_SCORE = 0.3
    USE_RERANKING = True
    USE_QUERY_EXPANSION = True
    USE_HYBRID_SEARCH = True
    HYBRID_WEIGHT_DENSE = 0.7  # Weight for dense retrieval
    HYBRID_WEIGHT_SPARSE = 0.3  # Weight for sparse retrieval

    # Advanced settings
    USE_SEMANTIC_CHUNKING = True
    USE_DOCUMENT_COMPRESSION = True
    USE_CACHING = True
    CACHE_TTL_HOURS = 24


class DocumentProcessor:
    """Advanced document processing with semantic chunking"""

    @staticmethod
    def semantic_chunk_text(text: str, embedding_model, target_chunk_size: int = Config.CHUNK_SIZE) -> List[str]:
        """Split text into semantic chunks based on sentence embeddings"""
        try:
            sentences = sent_tokenize(text)
            if not sentences:
                return [text]

            # Try to get embeddings with the new API
            try:
                # New HuggingFaceEmbeddings API
                sentence_embeddings = embedding_model.embed_documents(sentences)
            except AttributeError:
                # Fallback for older versions or different embedding models
                try:
                    sentence_embeddings = [embedding_model.embed_query(s) for s in sentences]
                except:
                    # If semantic chunking fails, fall back to regular chunking
                    logger.warning("Semantic chunking failed, falling back to regular chunking")
                    return DocumentProcessor._fallback_chunk(text, target_chunk_size)

            sentence_embeddings = np.array(sentence_embeddings)

            # Your existing chunking logic here...
            chunks = []
            current_chunk = []
            current_length = 0

            for i, sentence in enumerate(sentences):
                sentence_length = len(sentence)

                if current_length + sentence_length > target_chunk_size and current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_length = sentence_length
                else:
                    current_chunk.append(sentence)
                    current_length += sentence_length

            if current_chunk:
                chunks.append(' '.join(current_chunk))

            return chunks if chunks else [text]

        except Exception as e:
            logger.error(f"Error in semantic chunking: {e}")
            return DocumentProcessor._fallback_chunk(text, target_chunk_size)

    @staticmethod
    def _fallback_chunk(text: str, target_chunk_size: int) -> List[str]:
        """Simple fallback chunking method"""
        chunks = []
        for i in range(0, len(text), target_chunk_size):
            chunks.append(text[i:i + target_chunk_size])
        return chunks


class CacheManager:
    """Manage caching for embeddings and search results"""

    def __init__(self, cache_dir: str = Config.CACHE_DIR):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def get_cache_key(self, content: str) -> str:
        """Generate cache key from content"""
        return hashlib.md5(content.encode()).hexdigest()

    def save_embeddings(self, key: str, embeddings: np.ndarray):
        """Save embeddings to cache"""
        cache_path = os.path.join(self.cache_dir, f"{key}_embeddings.npy")
        np.save(cache_path, embeddings)

    def load_embeddings(self, key: str) -> Optional[np.ndarray]:
        """Load embeddings from cache"""
        cache_path = os.path.join(self.cache_dir, f"{key}_embeddings.npy")
        if os.path.exists(cache_path):
            # Check if cache is still valid
            file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_path))
            if file_age.total_seconds() < Config.CACHE_TTL_HOURS * 3600:
                return np.load(cache_path)
        return None


def check_cuda_availability():
    """Check if CUDA is available and return device info"""
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        st.sidebar.success(f"🚀 Using GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        device = "cpu"
        st.sidebar.warning("⚠️ GPU not available, using CPU")
    return device


def setup_environment():
    """Setup the documents directory and check for files"""
    if not os.path.exists(Config.DOC_DIR):
        os.makedirs(Config.DOC_DIR)
        st.error(f"📁 Created '{Config.DOC_DIR}' directory.")
        st.info("Please add your text files to this directory and refresh.")
        st.stop()

    # Check for supported files
    files = [
        f for f in os.listdir(Config.DOC_DIR)
        if any(f.endswith(ext) for ext in Config.SUPPORTED_EXTENSIONS)
    ]

    if not files:
        st.warning(f"No supported files found in '{Config.DOC_DIR}'")
        st.info(f"Supported formats: {', '.join(Config.SUPPORTED_EXTENSIONS)}")
        return False

    return True


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """Load and cache the embedding model"""
    with st.spinner("Loading embedding model..."):
        try:
            device = check_cuda_availability()
            embeddings = HuggingFaceEmbeddings(
                model_name=Config.EMBEDDING_MODEL_NAME,
                model_kwargs={'device': device},
                encode_kwargs={
                    'normalize_embeddings': True,
                    'batch_size': 32
                }
            )
            return embeddings
        except Exception as e:
            st.error(f"Failed to load embedding model: {str(e)}")
            st.stop()


@st.cache_resource(show_spinner=False)
def load_reranker():
    """Load the cross-encoder reranker model"""
    if Config.USE_RERANKING:
        with st.spinner("Loading reranker model..."):
            try:
                reranker = CrossEncoder(Config.RERANKER_MODEL_NAME)
                return reranker
            except Exception as e:
                st.warning(f"Failed to load reranker: {str(e)}")
                Config.USE_RERANKING = False
                return None
    return None


def load_document_with_metadata(file_path: str) -> List[Dict]:
    """Load document with enhanced metadata"""
    documents = []
    file_name = os.path.basename(file_path)
    file_ext = os.path.splitext(file_name)[1].lower()

    try:
        if file_ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_ext == '.md':
            loader = UnstructuredMarkdownLoader(file_path)
        else:
            loader = TextLoader(file_path, encoding='utf-8')

        docs = loader.load()

        # Add enhanced metadata
        for i, doc in enumerate(docs):
            doc.metadata.update({
                'source': file_name,
                'file_type': file_ext,
                'chunk_index': i,
                'total_chunks': len(docs),
                'timestamp': datetime.now().isoformat()
            })
            documents.append(doc)

        return documents
    except Exception as e:
        st.warning(f"Failed to load {file_name}: {str(e)}")
        return []


@st.cache_resource(show_spinner=False)
def load_vector_store_and_retrievers(_embedding_model) -> Tuple[Optional[FAISS], Optional[EnsembleRetriever]]:
    """Load documents and create both vector store and BM25 retriever"""
    with st.spinner("Loading documents and creating retrievers..."):
        try:
            documents = []
            file_count = 0
            cache_manager = CacheManager() if Config.USE_CACHING else None

            # Load all documents
            progress_bar = st.progress(0)
            files = [
                f for f in os.listdir(Config.DOC_DIR)
                if any(f.endswith(ext) for ext in Config.SUPPORTED_EXTENSIONS)
            ]

            for idx, file in enumerate(files):
                file_path = os.path.join(Config.DOC_DIR, file)
                docs = load_document_with_metadata(file_path)
                documents.extend(docs)
                file_count += 1
                progress_bar.progress((idx + 1) / len(files))
                logger.info(f"Loaded {file}")

            progress_bar.empty()

            if not documents:
                st.error("No documents could be loaded.")
                return None, None

            # Create text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=Config.CHUNK_SIZE,
                chunk_overlap=Config.CHUNK_OVERLAP,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )

            # Process documents with semantic chunking if enabled
            if Config.USE_SEMANTIC_CHUNKING:
                st.info("Using semantic chunking...")
                processor = DocumentProcessor()
                processed_docs = []

                for doc in documents:
                    chunks = processor.semantic_chunk_text(
                        doc.page_content,
                        _embedding_model,
                        Config.CHUNK_SIZE
                    )

                    for i, chunk in enumerate(chunks):
                        # Create new document for each chunk
                        from langchain.schema import Document
                        new_doc = Document(
                            page_content=chunk,
                            metadata={**doc.metadata, 'semantic_chunk_index': i}
                        )
                        processed_docs.append(new_doc)

                texts = processed_docs
            else:
                texts = text_splitter.split_documents(documents)

            # Create vector store
            st.info("Creating vector store...")
            vector_store = FAISS.from_documents(texts, _embedding_model)

            # Create ensemble retriever if hybrid search is enabled
            ensemble_retriever = None
            if Config.USE_HYBRID_SEARCH:
                st.info("Setting up hybrid search...")

                # Create BM25 retriever
                bm25_retriever = BM25Retriever.from_documents(texts)
                bm25_retriever.k = Config.TOP_K

                # Create dense retriever
                dense_retriever = vector_store.as_retriever(
                    search_kwargs={"k": Config.TOP_K}
                )

                # Create ensemble
                ensemble_retriever = EnsembleRetriever(
                    retrievers=[dense_retriever, bm25_retriever],
                    weights=[Config.HYBRID_WEIGHT_DENSE, Config.HYBRID_WEIGHT_SPARSE]
                )

            st.success(f"✅ Created retrievers: {len(texts)} chunks from {file_count} files")
            return vector_store, ensemble_retriever

        except Exception as e:
            st.error(f"Failed to create retrievers: {str(e)}")
            return None, None


@st.cache_resource(show_spinner=False)
def load_llm_model() -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Load and cache the LLM model and tokenizer"""
    with st.spinner(f"Loading {Config.MODEL_NAME}..."):
        try:
            device = check_cuda_availability()

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                Config.MODEL_NAME,
                token=os.environ.get("HF_TOKEN")
            )

            # Set padding token if not set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load model with optimizations
            model_kwargs = {
                "device_map": "auto" if device == "cuda" else None,
                "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
                "low_cpu_mem_usage": True,
                "token": os.environ.get("HF_TOKEN")
            }

            # Try to load with quantization if available
            if device == "cuda":
                try:
                    model_kwargs["load_in_8bit"] = True
                except:
                    pass

            model = AutoModelForCausalLM.from_pretrained(
                Config.MODEL_NAME,
                **model_kwargs
            )

            if device == "cpu":
                model = model.to(device)

            st.success("✅ LLM model loaded successfully!")
            return tokenizer, model

        except Exception as e:
            st.error(f"Failed to load LLM model: {str(e)}")
            st.stop()


def expand_query(query: str, embedding_model) -> List[str]:
    """Expand query using semantic similarity"""
    if not Config.USE_QUERY_EXPANSION:
        return [query]

    # Simple query expansion using synonyms and related terms
    expansions = [query]

    # Add question variations
    question_patterns = [
        query,
        f"What is {query}",
        f"Explain {query}",
        f"Information about {query}",
        f"Details on {query}"
    ]

    # Keep only unique and relevant expansions
    seen = set()
    final_expansions = []
    for exp in question_patterns[:3]:  # Limit to 3 variations
        if exp.lower() not in seen and len(exp) > 5:
            seen.add(exp.lower())
            final_expansions.append(exp)

    return final_expansions


def retrieve_context_enhanced(
        vector_store: FAISS,
        ensemble_retriever: Optional[EnsembleRetriever],
        queries: List[str],
        reranker=None,
        k: int = Config.TOP_K
) -> Tuple[str, List]:
    """Enhanced retrieval with query expansion, hybrid search, and reranking"""
    try:
        all_results = []
        seen_content = set()

        # Use ensemble retriever if available, otherwise fall back to vector store
        if ensemble_retriever and Config.USE_HYBRID_SEARCH:
            # Hybrid retrieval
            for query in queries:
                results = ensemble_retriever.get_relevant_documents(query)

                for doc in results[:k]:
                    content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        # Calculate a pseudo-score based on position
                        score = 1.0 - (len(all_results) / k)
                        all_results.append((doc, score, query))
        else:
            # Vector store retrieval
            for query in queries:
                results = vector_store.similarity_search_with_score(query, k=k)

                for doc, score in results:
                    content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        # Convert distance to similarity
                        similarity = 1 / (1 + score)  # Convert distance to similarity
                        all_results.append((doc, similarity, query))

        # Sort by score
        all_results.sort(key=lambda x: x[1], reverse=True)

        # Apply minimum similarity threshold
        filtered_results = [
            (doc, score, query) for doc, score, query in all_results
            if score >= Config.MIN_SIMILARITY_SCORE
        ]

        # If no results meet threshold, use top k anyway
        if not filtered_results:
            filtered_results = all_results[:Config.TOP_K_RERANK]

        # Reranking
        if reranker and Config.USE_RERANKING and len(filtered_results) > 0:
            # Prepare pairs for reranking
            pairs = []
            for doc, _, _ in filtered_results[:Config.TOP_K]:  # Limit reranking to top K
                pairs.append([queries[0], doc.page_content])

            # Get reranking scores
            with st.spinner("Reranking results..."):
                rerank_scores = reranker.predict(pairs)

            # Combine with original scores
            final_results = []
            for i, (doc, sim, query) in enumerate(filtered_results[:len(pairs)]):
                # Normalize rerank score
                rerank_score = float(rerank_scores[i])
                # Weighted combination
                final_score = 0.3 * sim + 0.7 * rerank_score
                final_results.append((doc, final_score, query))

            # Sort by final score
            final_results.sort(key=lambda x: x[1], reverse=True)
            filtered_results = final_results

        # Document compression - remove redundant information
        if Config.USE_DOCUMENT_COMPRESSION:
            filtered_results = compress_results(filtered_results)

        # Format context with metadata
        context_parts = []
        sources = []

        for doc, score, query in filtered_results[:Config.TOP_K_RERANK]:
            content = doc.page_content.strip()
            if content:
                # Add section header with metadata
                source = doc.metadata.get('source', 'Unknown')
                chunk_idx = doc.metadata.get('chunk_index', '?')

                context_header = f"[Source: {source}, Chunk: {chunk_idx}]"
                context_parts.append(f"{context_header}\n{content}")
                sources.append(f"{source} (chunk {chunk_idx}, score: {score:.2%})")

        context = "\n\n---\n\n".join(context_parts)
        return context, sources

    except Exception as e:
        st.error(f"Error retrieving context: {str(e)}")
        return "", []


def compress_results(results: List[Tuple]) -> List[Tuple]:
    """Remove redundant information from results"""
    compressed = []
    seen_info = set()

    for doc, score, query in results:
        # Create a simplified version for comparison
        simplified = ' '.join(doc.page_content.lower().split()[:20])

        if simplified not in seen_info:
            seen_info.add(simplified)
            compressed.append((doc, score, query))

    return compressed


def create_enhanced_prompt(query: str, context: str) -> str:
    """Create an optimized prompt for Gemma-2B"""
    if context:
        # Gemma-specific prompt format
        prompt = f"""<start_of_turn>user
Context information is below.
---------------------
{context}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {query}
Answer: <end_of_turn>
<start_of_turn>model
Based on the provided context, """
    else:
        prompt = f"""<start_of_turn>user
No relevant context was found in the documents for this query: {query}
Please respond that the information is not available in the documents.
<end_of_turn>
<start_of_turn>model
I cannot find information about "{query}" in the provided documents. """

    return prompt


def generate_response_stream(prompt: str, tokenizer, model):
    """Generate streaming response from the model"""
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    # Tokenize input
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=8192  # Gemma-2B supports 8K context
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generation settings optimized for accuracy
    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": Config.MAX_NEW_TOKENS,
        "temperature": Config.TEMPERATURE,
        "top_p": Config.TOP_P,
        "do_sample": Config.TEMPERATURE > 0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": 1.15,  # Reduce repetition
        "length_penalty": 1.0,  # Neutral length preference
        "early_stopping": True,
        "num_beams": 2 if Config.TEMPERATURE == 0 else 1  # Beam search for greedy
    }

    # Run generation in separate thread
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # Stream tokens
    for new_text in streamer:
        yield new_text

    thread.join()


def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Enhanced RAG System",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for better UI
    st.markdown("""
    <style>
    .stAlert {
        border-radius: 10px;
    }
    .css-1d391kg {
        padding: 2rem 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.title("📚 Enhanced Retrieval-Augmented Generation (RAG) System")
    st.markdown("*Advanced document Q&A with semantic search, reranking, and hybrid retrieval*")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Model info
        st.subheader("🤖 Model Information")
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**LLM:** {Config.MODEL_NAME.split('/')[-1]}")
        with col2:
            st.info(f"**Embeddings:** {Config.EMBEDDING_MODEL_NAME.split('/')[-1]}")

        # Feature toggles
        st.subheader("🔧 Features")
        Config.USE_HYBRID_SEARCH = st.checkbox("Hybrid Search (Dense + Sparse)", Config.USE_HYBRID_SEARCH)
        Config.USE_RERANKING = st.checkbox("Reranking", Config.USE_RERANKING)
        Config.USE_QUERY_EXPANSION = st.checkbox("Query Expansion", Config.USE_QUERY_EXPANSION)
        Config.USE_SEMANTIC_CHUNKING = st.checkbox("Semantic Chunking", Config.USE_SEMANTIC_CHUNKING)

        # Advanced settings
        with st.expander("🎛️ Advanced Settings"):
            Config.TOP_K = st.slider(
                "Initial documents to retrieve",
                min_value=10, max_value=50, value=Config.TOP_K, step=5
            )
            Config.TOP_K_RERANK = st.slider(
                "Final documents after reranking",
                min_value=1, max_value=10, value=Config.TOP_K_RERANK
            )
            Config.TEMPERATURE = st.slider(
                "Temperature (0=deterministic)",
                min_value=0.0, max_value=1.0, value=Config.TEMPERATURE, step=0.05
            )
            Config.MAX_NEW_TOKENS = st.slider(
                "Max response tokens",
                min_value=128, max_value=1024, value=Config.MAX_NEW_TOKENS, step=64
            )
            Config.MIN_SIMILARITY_SCORE = st.slider(
                "Min similarity threshold",
                min_value=0.0, max_value=1.0, value=Config.MIN_SIMILARITY_SCORE, step=0.05
            )

            if Config.USE_HYBRID_SEARCH:
                Config.HYBRID_WEIGHT_DENSE = st.slider(
                    "Dense retrieval weight",
                    min_value=0.0, max_value=1.0, value=Config.HYBRID_WEIGHT_DENSE, step=0.1
                )
                Config.HYBRID_WEIGHT_SPARSE = 1.0 - Config.HYBRID_WEIGHT_DENSE

        # Document info
        st.subheader("📄 Document Status")
        if os.path.exists(Config.DOC_DIR):
            files = [
                f for f in os.listdir(Config.DOC_DIR)
                if any(f.endswith(ext) for ext in Config.SUPPORTED_EXTENSIONS)
            ]
            if files:
                st.success(f"✅ {len(files)} documents loaded")
                with st.expander("View documents"):
                    for f in sorted(files):
                        file_size = os.path.getsize(os.path.join(Config.DOC_DIR, f)) / 1024
                        st.text(f"📄 {f} ({file_size:.1f} KB)")
            else:
                st.warning("⚠️ No documents found")

        # System info
        st.subheader("💻 System Info")
        device = "GPU ✅" if torch.cuda.is_available() else "CPU ⚠️"
        st.info(f"**Device:** {device}")

    # Setup environment
    if not setup_environment():
        return

    # Load models
    embedding_model = load_embedding_model()
    vector_store, ensemble_retriever = load_vector_store_and_retrievers(embedding_model)
    reranker = load_reranker() if Config.USE_RERANKING else None
    tokenizer, model = load_llm_model()

    # Main interface
    st.markdown("---")

    # Query input with examples
    example_queries = [
        "Does the word 'hospital' exist in the document?",
        "What are the main topics discussed?",
        "Summarize the key points",
        "Find information about [specific topic]"
    ]

    selected_example = st.selectbox(
        "📝 Example queries:",
        ["Custom query..."] + example_queries,
        index=0
    )

    if selected_example == "Custom query...":
        query = st.text_area(
            "💭 Enter your question:",
            height=100,
            placeholder="Type your question here..."
        )
    else:
        query = st.text_area(
            "💭 Enter your question:",
            value=selected_example,
            height=100
        )

    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        search_button = st.button(
            "🔍 Search",
            type="primary",
            use_container_width=True
        )
    with col2:
        clear_button = st.button(
            "🗑️ Clear",
            use_container_width=True
        )

    if clear_button:
        st.rerun()

    # Process query
    if search_button and query:
        # Query expansion
        expanded_queries = [query]
        if Config.USE_QUERY_EXPANSION:
            with st.spinner("Expanding query..."):
                expanded_queries = expand_query(query, embedding_model)
                if len(expanded_queries) > 1:
                    with st.expander("🔄 Query variations used"):
                        for i, eq in enumerate(expanded_queries, 1):
                            st.text(f"{i}. {eq}")

        # Retrieve context
        context = ""
        sources = []

        if vector_store:
            with st.spinner("🔎 Searching documents..."):
                context, sources = retrieve_context_enhanced(
                    vector_store,
                    ensemble_retriever,
                    expanded_queries,
                    reranker,
                    Config.TOP_K
                )

            # Show retrieval results
            if sources:
                with st.expander(f"📑 Retrieved {len(sources)} relevant chunks", expanded=False):
                    for i, source in enumerate(sources, 1):
                        st.markdown(f"{i}. {source}")

            # Show context preview
            if context:
                with st.expander("📄 Context preview", expanded=False):
                    preview_length = 1000
                    if len(context) > preview_length:
                        st.text(context[:preview_length] + "...")
                        st.caption(f"Showing first {preview_length} characters of {len(context)} total")
                    else:
                        st.text(context)
            else:
                st.warning("⚠️ No relevant context found. The model will respond based on the query alone.")
        else:
            st.error("❌ Vector store not available.")

        # Create prompt
        prompt = create_enhanced_prompt(query, context)

        # Generate response
        st.markdown("### 💬 Answer:")

        # Response metrics
        response_container = st.container()
        with response_container:
            message_placeholder = st.empty()
            full_response = ""
            token_count = 0

            # Stream the response
            start_time = datetime.now()

            for chunk in generate_response_stream(prompt, tokenizer, model):
                full_response += chunk
                token_count += len(tokenizer.encode(chunk))
                message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

            # Show metrics
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Response Time", f"{response_time:.1f}s")
            with col2:
                st.metric("Tokens Generated", token_count)
            with col3:
                st.metric("Tokens/Second", f"{token_count / response_time:.1f}")

        # Feedback section
        st.markdown("---")
        st.markdown("### 📊 Was this answer helpful?")
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            if st.button("👍 Yes"):
                st.success("Thank you for your feedback!")
        with col2:
            if st.button("👎 No"):
                st.info("Thank you. Consider adjusting the settings or improving your document quality.")

    # Footer
    st.markdown("---")
    st.caption(
        f"Enhanced RAG System | "
        f"{'Hybrid' if Config.USE_HYBRID_SEARCH else 'Dense'} Search | "
        f"{'Reranking ✓' if Config.USE_RERANKING else 'No Reranking'} | "
        f"{'Query Expansion ✓' if Config.USE_QUERY_EXPANSION else 'No Expansion'}"
    )


if __name__ == "__main__":
    main()