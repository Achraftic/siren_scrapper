from config import INPUT_PARQUET_PATH, SIRET_BATCHES_DIR, BATCH_FILE_PREFIX
import logging
from pathlib import Path
import polars as pl

INPUT_PARQUET = INPUT_PARQUET_PATH
OUTPUT_DIR = SIRET_BATCHES_DIR
BATCH_SIZE = 100000
FILE_PREFIX = BATCH_FILE_PREFIX


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("siret_batch_export")


def load_siret_list(path: Path) -> list[str]:
    lf = (
        pl.scan_parquet(path)
        .select(
            pl.col("siret")
            .cast(pl.Utf8)
            .str.replace_all(r"[^0-9]", "")
            .str.zfill(14)
            .alias("siret")
        )
        .filter(pl.col("siret").str.len_chars() == 14)
        .unique()
        .sort("siret")
    )
    return lf.collect(engine="streaming").get_column("siret").to_list()


def write_batches(
    values: list[str], output_dir: Path, batch_size: int, prefix: str
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    for existing in output_dir.glob(f"{prefix}_*.txt"):
        existing.unlink()
    batch_count = 0
    for index in range(0, len(values), batch_size):
        chunk = values[index : index + batch_size]
        batch_count += 1
        file_path = output_dir / f"{prefix}_{batch_count:05d}.txt"
        file_path.write_text("\n".join(chunk) + "\n", encoding="utf-8")
    return batch_count


def main() -> None:
    if not INPUT_PARQUET.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_PARQUET}")

    logger.info("Loading SIRET from %s", INPUT_PARQUET)
    siret_values = load_siret_list(INPUT_PARQUET)
    logger.info("Loaded %s unique SIRET", len(siret_values))

    logger.info("Writing .txt batches to %s", OUTPUT_DIR)
    batch_count = write_batches(siret_values, OUTPUT_DIR, BATCH_SIZE, FILE_PREFIX)
    logger.info("Done: %s files written", batch_count)


if __name__ == "__main__":
    main()
