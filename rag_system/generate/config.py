# config.py
"""
Configuration settings for answer generation, retrieval.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Make sure directories exist
for dir_path in [DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Model settings
# Falcon3-10B-Instruct, tiiuae/Falcon3-10B-Instruct
LLM_MODEL_ID = "tiiuae/Falcon3-10B-Instruct"  # ... or "tiiuae/Falcon3-10B-Instruct"
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
MAX_NEW_TOKENS = 512 # Was 1024 may cause issues. Can experiment with 25-1024+
TEMPERATURE = 0.6 # Controls randomness of generation
TOP_P = 0.9
DO_SAMPLE = True

# Retrieval settings
DEFAULT_TOP_K_FINAL = 10        # For final context
BM25_INDEX_PATH = "/local/scratch/.../fineweb_index/"
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"  # Lightweight model for dense retrieval

# Data file paths
FINEWEB_SUBSET_PATH = os.path.join(DATA_DIR, "fineweb_subset.parquet")
FAISS_INDEX_PATH = os.path.join(MODELS_DIR, "fineweb_faiss.index")
EMBEDDINGS_PATH = os.path.join(MODELS_DIR, "fineweb_embeddings.npy")
METADATA_PATH = os.path.join(MODELS_DIR, "fineweb_metadata.npz")

# Cache settings
CACHE_SIZE = 1000  # Number of items to cache
CACHE_EXPIRY = 3600  # Cache expiry time in seconds (1 hour)

# Logging settings
LOG_LEVEL = "INFO"
ENABLE_TELEMETRY = True