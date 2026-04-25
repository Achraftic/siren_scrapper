import requests
import time
import logging
import threading
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from .config import USER_AGENT, API_URL

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(threadName)s | %(message)s",
)
logger = logging.getLogger("api_utils")

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
        # Respect global cooldown if any thread hit a 429
        current_time = time.time()
        if current_time < cooldown_until:
            sleep_time = cooldown_until - current_time + 0.1
            time.sleep(sleep_time)
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
                logger.warning(f"Rate limited (429) for SIRET {siret}. Global cooldown for {wait_time} seconds...")
                with cooldown_lock:
                    cooldown_until = time.time() + wait_time
                continue
            
            elif response.status_code == 404:
                logger.debug(f"SIRET {siret} not found (404).")
                return {"error": "not_found", "status": 404}
            
            else:
                logger.error(f"Error {response.status_code} for SIRET {siret}: {response.text[:100]}")
                return {"error": "api_error", "status": response.status_code}
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request exception for SIRET {siret}: {e}")
            time.sleep(2)
            return {"error": "network_error", "exception": str(e)}
