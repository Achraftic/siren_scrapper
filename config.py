import os
from pathlib import Path

# Base Directories
BASE_DIR = Path(os.environ.get("BASE_DIR", os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = BASE_DIR / "DATA"

# Subdirectories
SIRET_BATCHES_DIR = BASE_DIR / "data_api" / "siret_batches"
API_RESULTS_DIR = BASE_DIR / "data_api"

# File Paths
INPUT_PARQUET_PATH = DATA_DIR / "prospecting_leads_2026.parquet"

# File Prefixes
BATCH_FILE_PREFIX = "siret_batch"

# Scraping Configuration
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "4"))
MAX_BATCHES_PER_RUN = int(os.environ.get("MAX_BATCHES_PER_RUN", "999999"))
REQUEST_DELAY = 0.4  # Seconds between requests per thread
CHUNK_SIZE = 500     # Number of SIRETs to process before updating checkpoint

# API Configuration
API_URL = "https://recherche-entreprises.api.gouv.fr/search"
USER_AGENT = "Mozilla/5.0 (DataMiningProject; contact@example.com)"

# Ensure directories exist
SIRET_BATCHES_DIR.mkdir(parents=True, exist_ok=True)
API_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
