import os
import pandas as pd
import time
import json
import logging
from config import DATA_DIR, API_RESULTS_DIR, SIRET_BATCHES_DIR
from utils import setup_session, fetch_siret_data

# Setup logging for the test script
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("test_api")

def main():
    # Configuration
    output_file = API_RESULTS_DIR / "api_test_results.parquet"
    
    # Try to find a batch file for testing
    batch_files = sorted(list(SIRET_BATCHES_DIR.glob("siret_batch_*.txt")))
    
    if not batch_files:
        logger.warning(f"No batch files found in {SIRET_BATCHES_DIR}. Using fallback SIRETs.")
        sirets = ["00032517500065", "00542002100056"]
    else:
        siret_file = batch_files[0]
        logger.info(f"Using SIRETs from {siret_file}")
        with open(siret_file, "r") as f:
            sirets = [line.strip() for line in f.readlines()[:20]]

    logger.info(f"Starting API test for {len(sirets)} SIRETs...")
    session = setup_session()
    all_results = []

    for i, siret in enumerate(sirets):
        logger.info(f"[{i + 1}/{len(sirets)}] Fetching data for SIRET: {siret}")
        data = fetch_siret_data(session, siret)

        if isinstance(data, dict) and "results" in data and len(data["results"]) > 0:
            res = data["results"][0]
            res["queried_siret"] = siret
            res["api_status"] = "success"
            all_results.append(res)
        else:
            status = data.get("error", "no_data") if isinstance(data, dict) else "unknown"
            logger.warning(f"No results or error for SIRET {siret}: {status}")
            all_results.append({
                "queried_siret": siret, 
                "api_status": status, 
                "nom_complet": None
            })

        time.sleep(0.5)

    if all_results:
        df = pd.DataFrame(all_results)

        # Serialize complex columns
        for col in df.columns:
            if df[col].apply(lambda x: isinstance(x, (dict, list))).any():
                df[col] = df[col].apply(
                    lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x
                )

        try:
            df.to_parquet(output_file, index=False)
            logger.info(f"\nSUCCESS: Saved {len(df)} results to {output_file}")
            print(df[["queried_siret", "nom_complet", "api_status"]].head())
        except Exception as e:
            logger.error(f"Error saving to Parquet: {e}")
            csv_output = str(output_file).replace(".parquet", ".csv")
            df.to_csv(csv_output, index=False)
            logger.info(f"Saved to CSV instead: {csv_output}")

if __name__ == "__main__":
    main()
