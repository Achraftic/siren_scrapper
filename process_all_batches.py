import os
import requests
import pandas as pd
import time
import json
import logging
from glob import glob
from concurrent.futures import ThreadPoolExecutor
import threading
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- CONFIGURATION (Static, no os.environ) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_API_DIR = os.path.join(BASE_DIR, "data_api")
SIRET_BATCHES_DIR = os.path.join(DATA_API_DIR, "siret_batches")

MAX_WORKERS = 5
MAX_BATCHES_PER_RUN = 5  # Increased, but limited by MAX_RUN_DURATION
CHUNK_SIZE = 200        # Smaller chunks for more frequent checkpoints
REQUEST_DELAY = 0.75    # Conservative delay to stay under 7 req/s (5 * 1/0.75 = 6.6 req/s)
API_URL = "https://recherche-entreprises.api.gouv.fr/search"
USER_AGENT = "Mozilla/5.0 (DataMiningProject; contact@example.com)"

# Execution Time Limit (5.5 hours to allow Git push)
MAX_RUN_DURATION = 5.5 * 3600 
START_TIME = time.time()

# Ensure directory exists
os.makedirs(DATA_API_DIR, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(threadName)s | %(message)s",
)
logger = logging.getLogger("batch_processor")

# Global state for rate limiting across threads
cooldown_until = 0
cooldown_lock = threading.Lock()

def setup_session():
    """
    Sets up a requests session with a retry strategy for common transient errors.
    """
    session = requests.Session()
    retry_strategy = Retry(
        total=10,
        backoff_factor=3,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
        connect=5,
        read=5,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": USER_AGENT})
    return session

def fetch_siret_data(session, siret):
    """
    Fetches data for a single SIRET with manual rate limit handling and global cooldown check.
    """
    global cooldown_until
    params = {"q": siret}

    while True:
        # Respect global cooldown
        current_time = time.time()
        if current_time < cooldown_until:
            time.sleep(cooldown_until - current_time + 0.1)
            continue

        try:
            response = session.get(API_URL, params=params, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            
            elif response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                wait_time = int(retry_after) if retry_after and retry_after.isdigit() else 20
                logger.warning(f"Rate limited for SIRET {siret}. Global cooldown for {wait_time}s")
                with cooldown_lock:
                    cooldown_until = time.time() + wait_time
                continue
            
            elif response.status_code == 404:
                return {"error": "not_found", "status": 404}
            
            else:
                logger.error(f"Error {response.status_code} for SIRET {siret}")
                return {"error": "api_error", "status": response.status_code}
                
        except Exception as e:
            logger.error(f"Request exception for SIRET {siret}: {e}")
            time.sleep(2)
            return {"error": "exception", "details": str(e)}

def process_batch(batch_file, output_parquet, session):
    """
    Processes a single batch file and saves it as a parquet using multi-threading.
    """
    batch_base = os.path.basename(batch_file)
    logger.info(f"Processing batch: {batch_base}")

    checkpoint_path = output_parquet + ".checkpoint"
    all_results = []
    start_index = 0

    # Resume logic: Load data from Parquet and index from tiny checkpoint
    if os.path.exists(output_parquet) and os.path.exists(checkpoint_path):
        try:
            # Load existing results from Parquet (compressed)
            df_existing = pd.read_parquet(output_parquet)
            all_results = df_existing.to_dict("records")
            
            # Load last index from tiny JSON
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                checkpoint_data = json.load(f)
                start_index = checkpoint_data.get("last_index", 0)
                
            logger.info(f"Resuming at index {start_index} with {len(all_results)} existing results...")
        except Exception as e:
            logger.warning(f"Could not load resume state: {e}. Starting fresh.")
            all_results = []
            start_index = 0

    with open(batch_file, "r", encoding="utf-8") as f:
        sirets = [line.strip() for line in f.readlines()]

    total_sirets = len(sirets)

    def worker(siret):
        data = fetch_siret_data(session, siret)
        time.sleep(REQUEST_DELAY)
        if data and "results" in data and len(data["results"]) > 0:
            res = data["results"][0]
            res["queried_siret"] = siret
            res["api_status"] = "success"
            return res
        return {"queried_siret": siret, "api_status": "no_data"}

    start_time = time.time()

    for i in range(start_index, total_sirets, CHUNK_SIZE):
        chunk = sirets[i : i + CHUNK_SIZE]

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results = list(executor.map(worker, chunk))
            all_results.extend(results)

        # Save PROGRESS: Tiny JSON for metadata + Parquet for data
        try:
            # Save metadata
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump({"last_index": i + len(chunk)}, f)
            
            # Save data to Parquet (much smaller than JSON)
            df = pd.DataFrame(all_results)
            # Ensure complex types are handled
            for col in df.columns:
                if df[col].apply(lambda x: isinstance(x, (dict, list))).any():
                    df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x)
            df.to_parquet(output_parquet, index=False)
            
        except Exception as e:
            logger.error(f"Failed checkpoint/save: {e}")

        elapsed = time.time() - start_time
        processed = i + len(chunk)
        speed = processed / (elapsed + 0.001)
        logger.info(f"[{processed}/{total_sirets}] Speed: {speed:.2f} req/s")

        # Check for global timeout
        if time.time() - START_TIME > MAX_RUN_DURATION:
            logger.warning("Approaching execution time limit. Results already saved.")
            return "TIMEOUT"

    # Final cleanup: remove tiny checkpoint if finished
    logger.info(f"Finished batch {batch_base}. Saved {len(all_results)} results.")
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

def main():
    batch_files = sorted(glob(os.path.join(SIRET_BATCHES_DIR, "siret_batch_*.txt")))

    if not batch_files:
        logger.error(f"No batches found in {SIRET_BATCHES_DIR}")
        return

    session = setup_session()
    batches_processed = 0

    for batch_file in batch_files:
        if batches_processed >= MAX_BATCHES_PER_RUN:
            break

        batch_name = os.path.basename(batch_file).replace(".txt", ".parquet")
        output_path = os.path.join(DATA_API_DIR, batch_name)
        checkpoint_path = output_path + ".checkpoint"

        # Check if batch is already fully completed
        if os.path.exists(output_path) and not os.path.exists(checkpoint_path):
            logger.info(f"Skipping {batch_name} (already finished)")
            continue

        try:
            status = process_batch(batch_file, output_path, session)
            if status == "TIMEOUT":
                logger.info("Stopping run due to timeout.")
                break
            batches_processed += 1
        except Exception as e:
            logger.error(f"Error in batch {batch_name}: {e}")

if __name__ == "__main__":
    main()
