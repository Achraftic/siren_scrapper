from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests


BATCH_FILE = Path("./DATA/siret_batches/siret_batch_00001.txt")
OUTPUT_DIR = Path("./DATA/public_endpoint_tests")
MAX_SIRET_TO_TEST = 25
TIMEOUT_SECONDS = 12
SLEEP_BETWEEN_REQUESTS = 0.4
USER_AGENT = "public-endpoint-tester/1.0 (+no-auth-check)"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("public_source_tester")


@dataclass
class EndpointTest:
    name: str
    url_template: str
    kind: str


ENDPOINTS: list[EndpointTest] = [
    EndpointTest(
        name="recherche_entreprises_api_gouv",
        url_template="https://recherche-entreprises.api.gouv.fr/search?q={siret}",
        kind="json",
    ),
    EndpointTest(
        name="sirene_data_gouv_etablissement",
        url_template="https://entreprise.data.gouv.fr/api/sirene/v3/etablissements/{siret}",
        kind="json",
    ),
    EndpointTest(
        name="sirene_data_gouv_unite_legale",
        url_template="https://entreprise.data.gouv.fr/api/sirene/v3/unites_legales/{siren}",
        kind="json",
    ),
    EndpointTest(
        name="annuaire_entreprises_page",
        url_template="https://annuaire-entreprises.data.gouv.fr/entreprise/{siren}",
        kind="html",
    ),
    EndpointTest(
        name="insee_sirene_api_siret",
        url_template="https://api.insee.fr/entreprises/sirene/V3/siret/{siret}",
        kind="json",
    ),
]


def load_sirets(path: Path, max_count: int) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input batch file: {path}")
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    clean = []
    for value in values:
        value = "".join(ch for ch in value if ch.isdigit())
        if len(value) == 14:
            clean.append(value)
    unique_sorted = sorted(set(clean))
    return unique_sorted[:max_count]


def make_url(template: str, siret: str) -> str:
    return template.format(siret=siret, siren=siret[:9])


def test_single_url(
    session: requests.Session, url: str, kind: str
) -> tuple[bool, int, str]:
    try:
        response = session.get(url, timeout=TIMEOUT_SECONDS, allow_redirects=True)
    except requests.RequestException as exc:
        return False, 0, f"request_error: {exc}"

    status = response.status_code
    if status != 200:
        return False, status, f"http_{status}"

    if kind == "json":
        try:
            payload = response.json()
        except json.JSONDecodeError:
            return False, status, "invalid_json"

        if payload in (None, {}, []):
            return False, status, "empty_json"

        if isinstance(payload, dict) and payload.get("error"):
            return False, status, f"json_error:{payload.get('error')}"

        return True, status, "ok"

    body = response.text.strip()
    if len(body) < 80:
        return False, status, "html_too_short"

    return True, status, "ok"


def test_endpoint(
    session: requests.Session, endpoint: EndpointTest, sirets: list[str]
) -> dict[str, Any]:
    for idx, siret in enumerate(sirets, start=1):
        url = make_url(endpoint.url_template, siret)
        ok, status, reason = test_single_url(session, url, endpoint.kind)
        logger.info(
            "%s try %s/%s status=%s reason=%s",
            endpoint.name,
            idx,
            len(sirets),
            status,
            reason,
        )
        time.sleep(SLEEP_BETWEEN_REQUESTS)
        if ok:
            return {
                "name": endpoint.name,
                "url_template": endpoint.url_template,
                "kind": endpoint.kind,
                "works_without_api_key": True,
                "status_code": status,
                "sample_siret": siret,
                "reason": reason,
                "sample_url": url,
            }

    return {
        "name": endpoint.name,
        "url_template": endpoint.url_template,
        "kind": endpoint.kind,
        "works_without_api_key": False,
        "status_code": None,
        "sample_siret": None,
        "reason": "no_success_on_samples",
        "sample_url": None,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sirets = load_sirets(BATCH_FILE, MAX_SIRET_TO_TEST)
    logger.info("Loaded %s sample SIRET from %s", len(sirets), BATCH_FILE)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "*/*"})

    results = []
    for endpoint in ENDPOINTS:
        logger.info("Testing endpoint: %s", endpoint.name)
        results.append(test_endpoint(session, endpoint, sirets))

    all_df = pd.DataFrame(results)
    working_df = all_df[all_df["works_without_api_key"]].copy()

    all_csv = OUTPUT_DIR / "all_endpoint_results.csv"
    working_csv = OUTPUT_DIR / "working_endpoints.csv"
    working_txt = OUTPUT_DIR / "working_endpoints.txt"

    all_df.to_csv(all_csv, index=False)
    working_df.to_csv(working_csv, index=False)

    lines = []
    for _, row in working_df.iterrows():
        lines.append(f"{row['name']}|{row['url_template']}|{row['kind']}")
    working_txt.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    logger.info("Saved: %s", all_csv)
    logger.info("Saved: %s", working_csv)
    logger.info("Saved: %s", working_txt)
    logger.info("Working endpoints: %s/%s", len(working_df), len(all_df))


if __name__ == "__main__":
    main()
