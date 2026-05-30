import os
import requests
import pandas as pd
import time
import json
import logging
import hashlib
from glob import glob
from concurrent.futures import ThreadPoolExecutor
import threading
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- CONFIGURATION (Static, no os.environ) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_API_DIR = os.path.join(BASE_DIR, "data_api")
SIRET_BATCHES_DIR = os.path.join(DATA_API_DIR, "siret_batches")
HUGGINGFACE_TOKEN = "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"
MAX_WORKERS = 5
MAX_BATCHES_PER_RUN = 5  # Increased, but limited by MAX_RUN_DURATION
CHUNK_SIZE = 200  # Smaller chunks for more frequent checkpoints
REQUEST_DELAY = (
    0.75  # Conservative delay to stay under 7 req/s (5 * 1/0.75 = 6.6 req/s)
)
API_URL = "https://recherche-entreprises.api.gouv.fr/search"
USER_AGENT = "Mozilla/5.0 (DataMiningProject; contact@example.com)"
PROCESSED_INDEX_DIR = os.path.join(DATA_API_DIR, "processed_index")
DEDUP_KEY_COLUMN = "queried_identifier"

# Execution Time Limit (5.5 hours to allow Git push)
MAX_RUN_DURATION = 5.5 * 3600
START_TIME = time.time()

# Ensure directory exists
os.makedirs(DATA_API_DIR, exist_ok=True)
os.makedirs(PROCESSED_INDEX_DIR, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(threadName)s | %(message)s",
)
logger = logging.getLogger("batch_processor")


def download_state():
    """
    Download the current scraper state from Hugging Face dataset.
    """
    token = HUGGINGFACE_TOKEN
    if not token:
        logger.info("No Hugging Face token provided. Skipping state download.")
        return
    logger.info("Syncing state down from Hugging Face (axrafTic/siren_dataset)...")
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id="axrafTic/siren_dataset",
            repo_type="dataset",
            local_dir=DATA_API_DIR,
            token=token,
            ignore_patterns=["siret_batches/**", ".git/**", ".cache/**"],
        )
        logger.info("State sync down completed successfully.")
    except Exception as e:
        logger.error(f"Failed to sync state down from Hugging Face: {e}")


def upload_state():
    """
    Upload the current local scraper state to Hugging Face dataset.
    """
    token = HUGGINGFACE_TOKEN
    if not token:
        logger.info("No Hugging Face token provided. Skipping state upload.")
        return
    logger.info("Syncing state up to Hugging Face (axrafTic/siren_dataset)...")
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        api.upload_large_folder(
            folder_path=DATA_API_DIR,
            repo_id="axrafTic/siren_dataset",
            repo_type="dataset",
            token=token,
            ignore_patterns=["siret_batches/**", ".git/**", ".cache/**"],
            commit_message="Update scraper results and checkpoints",
        )
        logger.info("State sync up completed successfully.")
    except Exception as e:
        logger.error(f"Failed to sync state up to Hugging Face: {e}")


# Global state for rate limiting across threads
cooldown_until = 0
cooldown_lock = threading.Lock()


def normalize_identifier(raw_identifier):
    """
    Normalize SIRET/SIREN-like values to a stable key for deduplication.
    """
    value = str(raw_identifier).strip()
    digits_only = "".join(ch for ch in value if ch.isdigit())

    if len(digits_only) in {9, 14}:
        return digits_only

    return value.upper()


class ProcessedIdentifierStore:
    """
    Persistent processed-ID store backed by sharded text files.

    Files are stored under data_api/processed_index/<shard>.txt where each line
    is a completed identifier. This keeps state durable across CI runs while
    remaining merge-friendly in Git.
    """

    def __init__(self, index_dir):
        self.index_dir = index_dir
        self._lock = threading.Lock()
        self._loaded_shards = set()
        self._completed_by_shard = {}
        self._dirty_shards = set()
        self._in_progress = set()

    def _shard_for(self, identifier):
        digest = hashlib.sha1(identifier.encode("utf-8")).hexdigest()
        return digest[:2]

    def _shard_path(self, shard):
        return os.path.join(self.index_dir, f"{shard}.txt")

    def _load_shard_unlocked(self, shard):
        if shard in self._loaded_shards:
            return

        path = self._shard_path(shard)
        identifiers = set()
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                identifiers = {line.strip() for line in f if line.strip()}

        self._completed_by_shard[shard] = identifiers
        self._loaded_shards.add(shard)

    def claim(self, identifier):
        """
        Claim an identifier for processing if not yet completed/in-progress.
        """
        with self._lock:
            shard = self._shard_for(identifier)
            self._load_shard_unlocked(shard)

            if identifier in self._completed_by_shard[shard]:
                return False

            if identifier in self._in_progress:
                return False

            self._in_progress.add(identifier)
            return True

    def mark_completed(self, identifier):
        with self._lock:
            shard = self._shard_for(identifier)
            self._load_shard_unlocked(shard)
            self._in_progress.discard(identifier)

            if identifier not in self._completed_by_shard[shard]:
                self._completed_by_shard[shard].add(identifier)
                self._dirty_shards.add(shard)

    def release_claim(self, identifier):
        with self._lock:
            self._in_progress.discard(identifier)

    def flush(self):
        """
        Persist modified shards atomically.
        """
        with self._lock:
            dirty_shards = list(self._dirty_shards)

            for shard in dirty_shards:
                path = self._shard_path(shard)
                temp_path = path + ".tmp"
                identifiers = sorted(self._completed_by_shard.get(shard, set()))

                with open(temp_path, "w", encoding="utf-8") as f:
                    for value in identifiers:
                        f.write(value + "\n")

                os.replace(temp_path, path)

            self._dirty_shards.clear()

    def total_loaded_completed(self):
        with self._lock:
            return sum(len(values) for values in self._completed_by_shard.values())


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
                wait_time = (
                    int(retry_after) if retry_after and retry_after.isdigit() else 20
                )
                logger.warning(
                    f"Rate limited for SIRET {siret}. Global cooldown for {wait_time}s"
                )
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


def dedup_results(records):
    """
    Keep at most one row per queried identifier.
    """
    if not records:
        return records

    df = pd.DataFrame(records)

    if DEDUP_KEY_COLUMN not in df.columns and "queried_siret" in df.columns:
        df[DEDUP_KEY_COLUMN] = df["queried_siret"].astype(str)

    if DEDUP_KEY_COLUMN in df.columns:
        df = df.drop_duplicates(subset=[DEDUP_KEY_COLUMN], keep="last")

    return df.to_dict("records")


def process_batch(batch_file, output_parquet, session, processed_store):
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
            all_results = dedup_results(all_results)

            # Load last index from tiny JSON
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                checkpoint_data = json.load(f)
                start_index = checkpoint_data.get("last_index", 0)

            logger.info(
                f"Resuming at index {start_index} with {len(all_results)} existing results..."
            )
        except Exception as e:
            logger.warning(f"Could not load resume state: {e}. Starting fresh.")
            all_results = []
            start_index = 0

    with open(batch_file, "r", encoding="utf-8") as f:
        identifiers = [normalize_identifier(line.strip()) for line in f if line.strip()]

    # Keep original order while removing duplicates inside this batch file.
    identifiers = list(dict.fromkeys(identifiers))

    total_identifiers = len(identifiers)

    def worker(identifier):
        data = fetch_siret_data(session, identifier)
        time.sleep(REQUEST_DELAY)

        if data and "results" in data and len(data["results"]) > 0:
            res = data["results"][0]
            res["queried_siret"] = identifier
            res[DEDUP_KEY_COLUMN] = identifier
            res["api_status"] = "success"
            return res

        if data and data.get("error") == "not_found":
            return {
                "queried_siret": identifier,
                DEDUP_KEY_COLUMN: identifier,
                "api_status": "not_found",
            }

        if data and data.get("error") in {"api_error", "exception"}:
            return {
                "queried_siret": identifier,
                DEDUP_KEY_COLUMN: identifier,
                "api_status": "error",
                "error": data.get("error"),
                "status": data.get("status"),
                "details": data.get("details"),
            }

        return {
            "queried_siret": identifier,
            DEDUP_KEY_COLUMN: identifier,
            "api_status": "no_data",
        }

    start_time = time.time()

    for i in range(start_index, total_identifiers, CHUNK_SIZE):
        chunk = identifiers[i : i + CHUNK_SIZE]

        claimable = [
            identifier for identifier in chunk if processed_store.claim(identifier)
        ]

        results = []
        if claimable:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                results = list(executor.map(worker, claimable))

        for result in results:
            identifier = str(
                result.get(DEDUP_KEY_COLUMN) or result.get("queried_siret") or ""
            ).strip()
            if not identifier:
                continue

            if result.get("api_status") in {"success", "no_data", "not_found"}:
                processed_store.mark_completed(identifier)
            else:
                # Keep failed identifiers eligible for retry in subsequent chunks/runs.
                processed_store.release_claim(identifier)

        if results:
            all_results.extend(results)
            all_results = dedup_results(all_results)

            # Save PROGRESS: Tiny JSON for metadata + Parquet for data
            try:
                # Save metadata
                with open(checkpoint_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "last_index": i + len(chunk),
                            "processed_in_batch": len(all_results),
                        },
                        f,
                    )

                # Save data to Parquet (much smaller than JSON)
                df = pd.DataFrame(all_results)
                # Ensure complex types are handled
                for col in df.columns:
                    if df[col].apply(lambda x: isinstance(x, (dict, list))).any():
                        df[col] = df[col].apply(
                            lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x
                        )
                df.to_parquet(output_parquet, index=False)
                processed_store.flush()

            except Exception as e:
                logger.error(f"Failed checkpoint/save: {e}")
        else:
            # Skip Parquet rewriting to drastically improve speed when skipping processed items
            try:
                with open(checkpoint_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "last_index": i + len(chunk),
                            "processed_in_batch": len(all_results),
                        },
                        f,
                    )
            except Exception as e:
                logger.error(f"Failed checkpoint/save: {e}")

        elapsed = time.time() - start_time
        processed = i + len(chunk)
        speed = processed / (elapsed + 0.001)
        logger.info(
            f"[{processed}/{total_identifiers}] Speed: {speed:.2f} req/s | "
            f"submitted={len(claimable)}"
        )

        # Check for global timeout
        if time.time() - START_TIME > MAX_RUN_DURATION:
            logger.warning("Approaching execution time limit. Results already saved.")
            return "TIMEOUT"

    # Final cleanup: remove tiny checkpoint if finished
    logger.info(f"Finished batch {batch_base}. Saved {len(all_results)} results.")
    processed_store.flush()
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)


def main():
    # download_state()

    try:
        batch_files = sorted(glob(os.path.join(SIRET_BATCHES_DIR, "siret_batch_*.txt")))
        # Start tracking from batch 33 onwards
        batch_files = [
            f
            for f in batch_files
            if int(os.path.basename(f).replace("siret_batch_", "").replace(".txt", ""))
            >= 33
        ]

        if not batch_files:
            logger.error(f"No batches found in {SIRET_BATCHES_DIR}")
            return

        session = setup_session()
        processed_store = ProcessedIdentifierStore(PROCESSED_INDEX_DIR)
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
                status = process_batch(
                    batch_file, output_path, session, processed_store
                )
                if status == "TIMEOUT":
                    logger.info("Stopping run due to timeout.")
                    break
                batches_processed += 1
            except Exception as e:
                logger.error(f"Error in batch {batch_name}: {e}")

        processed_store.flush()
        logger.info(
            "Run completed. Loaded completed identifiers in memory: %s",
            processed_store.total_loaded_completed(),
        )
    finally:
        upload_state()


if __name__ == "__main__":
    main()
