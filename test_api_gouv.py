import os
import requests
import pandas as pd
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json

def setup_session():
    """
    Sets up a requests session with a retry strategy for common transient errors.
    """
    session = requests.Session()
    retry_strategy = Retry(
        total=5,
        backoff_factor=2,  # Wait 2, 4, 8, 16, 32 seconds between retries
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

def fetch_siret_data(session, siret):
    """
    Fetches data for a single SIRET with manual rate limit handling.
    """
    url = f"https://recherche-entreprises.api.gouv.fr/search?q={siret}"
    while True:
        try:
            response = session.get(url, timeout=15)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                # Specific handling for Rate Limiting
                retry_after = response.headers.get("Retry-After")
                wait_time = int(retry_after) if retry_after and retry_after.isdigit() else 10
                print(f"Rate limited for SIRET {siret}. Sleeping for {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            elif response.status_code == 404:
                print(f"SIRET {siret} not found (404).")
                return None
            else:
                print(f"Error {response.status_code} for SIRET {siret}: {response.text[:100]}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Request exception for SIRET {siret}: {e}")
            return None

def main():
    # Configuration
    base_dir = r"d:\entreprises_france"
    data_api_dir = os.path.join(base_dir, "DATA", "data_api")
    os.makedirs(data_api_dir, exist_ok=True)
    
    siret_file = os.path.join(base_dir, "DATA", "siret_batches", "siret_batch_00001.txt")
    output_file = os.path.join(data_api_dir, "api_results.parquet")
    
    # Read sample SIRETs (top 20 for testing)
    if not os.path.exists(siret_file):
        print(f"SIRET file not found at {siret_file}")
        # Fallback to a known SIRET if file missing
        sirets = ["00032517500065", "00542002100056"]
    else:
        with open(siret_file, 'r') as f:
            sirets = [line.strip() for line in f.readlines()[:20]]
    
    print(f"Starting API test for {len(sirets)} SIRETs...")
    session = setup_session()
    all_results = []
    
    for i, siret in enumerate(sirets):
        print(f"[{i+1}/{len(sirets)}] Fetching data for SIRET: {siret}")
        data = fetch_siret_data(session, siret)
        
        if data and 'results' in data and len(data['results']) > 0:
            # We take the first result (usually the most relevant match)
            res = data['results'][0]
            res['queried_siret'] = siret
            res['api_status'] = 'success'
            all_results.append(res)
        else:
            print(f"No results or error for SIRET: {siret}")
            all_results.append({
                'queried_siret': siret, 
                'api_status': 'no_data',
                'nom_complet': None
            })
        
        # Rate limiting prevention: subtle delay between requests
        # The API Gouv documentation suggests being gentle.
        time.sleep(0.5) 
        
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Parquet does not support complex nested types (dicts/lists) easily in some engines
        # We'll serialize complex columns to JSON strings for maximum compatibility
        for col in df.columns:
            if df[col].apply(lambda x: isinstance(x, (dict, list))).any():
                df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x)
        
        try:
            df.to_parquet(output_file, index=False)
            print(f"\nSUCCESS: Saved {len(df)} results to {output_file}")
            print(df[['queried_siret', 'nom_complet', 'api_status']].head())
        except Exception as e:
            print(f"Error saving to Parquet: {e}")
            # Fallback to CSV if Parquet fails (e.g. missing pyarrow)
            csv_output = output_file.replace(".parquet", ".csv")
            df.to_csv(csv_output, index=False)
            print(f"Saved to CSV instead: {csv_output}")

if __name__ == "__main__":
    main()
