import os
import requests
import pandas as pd
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
from glob import glob
from concurrent.futures import ThreadPoolExecutor
import threading

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
        read=5
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    # Adding a common User-Agent as good practice
    session.headers.update(
        {"User-Agent": "Mozilla/5.0 (DataMiningProject; contact@example.com)"}
    )
    return session


def fetch_siret_data(session, siret):
    """
    Fetches data for a single SIRET with manual rate limit handling and global cooldown check.
    """
    global cooldown_until
    url = f"https://recherche-entreprises.api.gouv.fr/search?q={siret}"

    while True:
        # Respect global cooldown if any thread hit a 429
        current_time = time.time()
        if current_time < cooldown_until:
            time.sleep(cooldown_until - current_time + 0.1)
            continue

        try:
            # Increased timeout to 30 seconds for better stability
            response = session.get(url, timeout=30)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                wait_time = (
                    int(retry_after) if retry_after and retry_after.isdigit() else 20
                )
                print(
                    f"Rate limited for SIRET {siret}. Global cooldown for {wait_time} seconds..."
                )
                with cooldown_lock:
                    cooldown_until = time.time() + wait_time
                continue
            elif response.status_code == 404:
                return None
            else:
                print(
                    f"Error {response.status_code} for SIRET {siret}: {response.text[:100]}"
                )
                return None
        except requests.exceptions.RequestException as e:
            print(f"Request exception for SIRET {siret}: {e}")
            time.sleep(5)  # Wait a bit on network error
            return None


def process_batch(batch_file, output_parquet, session, limit=None):
    """
    Processes a single batch file and saves it as a parquet using multi-threading and chunks.
    Supports resuming from a checkpoint if interrupted.
    """
    batch_base = os.path.basename(batch_file)
    print(f"Processing batch: {batch_base}")

    checkpoint_path = output_parquet + ".checkpoint"
    all_results = []
    start_index = 0

    # Load existing progress if checkpoint exists
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r") as f:
                checkpoint_data = json.load(f)
                all_results = checkpoint_data.get("results", [])
                start_index = checkpoint_data.get("last_index", 0)
                print(f"  Resuming from checkpoint at index {start_index}...")
        except Exception as e:
            print(f"  Warning: Could not load checkpoint: {e}. Starting from scratch.")

    with open(batch_file, "r") as f:
        sirets = [line.strip() for line in f.readlines()]

    if limit:
        sirets = sirets[:limit]

    max_workers = int(os.environ.get("MAX_WORKERS", "4"))
    chunk_size = 500

    print(f"  Using {max_workers} threads. Progress: {start_index}/{len(sirets)}")

    def worker(siret):
        data = fetch_siret_data(session, siret)
        # Increased sleep to be more conservative with API limits
        time.sleep(0.4) 
        if data and "results" in data and len(data["results"]) > 0:
            res = data["results"][0]
            res["queried_siret"] = siret
            res["api_status"] = "success"
            return res
        else:
            return {"queried_siret": siret, "api_status": "no_data"}

    start_time = time.time()
    total_sirets = len(sirets)

    for i in range(start_index, total_sirets, chunk_size):
        chunk = sirets[i : i + chunk_size]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(worker, chunk))
            all_results.extend(results)

        # Update checkpoint
        try:
            with open(checkpoint_path, "w") as f:
                json.dump({"last_index": i + chunk_size, "results": all_results}, f)
        except Exception as e:
            print(f"  Warning: Failed to save checkpoint: {e}")

        elapsed = time.time() - start_time
        processed = min(i + chunk_size, total_sirets)
        speed = processed / (elapsed + 0.001) if elapsed > 0 else 0
        print(
            f"  [{processed}/{total_sirets}] Speed: {speed:.2f} req/s | Last SIRET: {chunk[-1]}"
        )

    if all_results:
        df = pd.DataFrame(all_results)
        # Serialize nested fields
        for col in df.columns:
            if df[col].apply(lambda x: isinstance(x, (dict, list))).any():
                df[col] = df[col].apply(
                    lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x
                )

        df.to_parquet(output_parquet, index=False)
        print(f"Saved {len(df)} results to {output_parquet}")

        # Clean up checkpoint on success
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)


def main():
    base_dir = r"d:\entreprises_france"
    # Adjusted paths: siret_batches is now inside data_api
    data_api_dir = os.path.join(base_dir, "data_api")
    siret_batches_dir = os.path.join(data_api_dir, "siret_batches")
    os.makedirs(data_api_dir, exist_ok=True)

    batch_files = sorted(glob(os.path.join(siret_batches_dir, "siret_batch_*.txt")))

    if not batch_files:
        print("No batch files found.")
        return

    session = setup_session()

    print(f"Found {len(batch_files)} batches to process.")

    max_batches = int(os.environ.get("MAX_BATCHES_PER_RUN", "999999"))
    batches_processed = 0

    for batch_file in batch_files:
        if batches_processed >= max_batches:
            print(f"Reached limit of {max_batches} batches per run.")
            break

        batch_name = os.path.basename(batch_file).replace(".txt", ".parquet")
        output_path = os.path.join(data_api_dir, batch_name)

        if os.path.exists(output_path):
            print(f"Skipping {batch_name}, already exists.")
            continue

        # Full batch processing
        process_batch(batch_file, output_path, session)

        print(f"Batch {batch_name} completed successfully.")
        batches_processed += 1


if __name__ == "__main__":
    main()
