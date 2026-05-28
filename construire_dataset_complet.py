import csv
from pathlib import Path
import logging
from typing import Callable

import polars as pl


DATA_DIR = Path(__file__).resolve().parent / "DATA"
ENTREPRISES_IMMATRICULEES_PATH = DATA_DIR / "entreprises-immatriculees-en-2025.csv"
ENTREPRISES_RGE_PATH = DATA_DIR / "liste-des-entreprises-rge-2.csv"
SHOP_CRAFT_OFFICE_PATH = DATA_DIR / "shop_craft_office_csv" / "data.csv"
NAF_PATH = DATA_DIR / "nomenclature-dactivites-francaise-naf-rev-2-code-ape.csv"
BASE_CODES_POSTAUX_PATH = DATA_DIR / "base-officielle-codes-postaux.csv"
BASE_CODES_POSTAUX_ALT_PATH = DATA_DIR / "base-officielle-des-codes-postaux.csv"
LAPOSTE_PATH = DATA_DIR / "laposte-hexasmal (1).csv"
PROSPECTING_LEADS_PATH = DATA_DIR / "prospecting_leads_2026.parquet"
STOCK_ETABLISSEMENT_PATH = DATA_DIR / "StockEtablissement_utf8.parquet"
STOCK_UNITE_LEGALE_PATH = DATA_DIR / "StockUniteLegale_utf8.parquet"
OUTPUT_PATH = DATA_DIR / "complete_dataset.parquet"
OUTPUT_FORMAT = "parquet"
SUPPORTED_INPUT_SUFFIXES = {".csv", ".parquet"}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("dataset_builder")


def detect_separator(path: Path) -> str:
    sample = path.read_text(encoding="utf-8", errors="ignore")[:8192]
    candidates = [";", ",", "\t", "|"]
    try:
        return csv.Sniffer().sniff(sample, delimiters=candidates).delimiter
    except csv.Error:
        first_lines = sample.splitlines()[:5]
        return max(
            candidates,
            key=lambda separator: sum(line.count(separator) for line in first_lines),
        )


def scan_csv(path: Path) -> pl.LazyFrame:
    logger.info("Reading CSV: %s", path)
    return pl.scan_csv(
        path,
        separator=detect_separator(path),
        ignore_errors=True,
        truncate_ragged_lines=True,
    )


def scan_table(path: Path) -> pl.LazyFrame:
    if path.suffix.lower() == ".parquet":
        logger.info("Reading Parquet: %s", path)
        return pl.scan_parquet(path)
    return scan_csv(path)


def validate_paths(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing input files: {missing}")


def discover_data_files(data_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_INPUT_SUFFIXES
        and path.resolve() != OUTPUT_PATH.resolve()
    )


def normalize_postal_code(expr: pl.Expr) -> pl.Expr:
    return (
        expr.cast(pl.Utf8)
        .str.replace_all(r"[^0-9A-Za-z]", "")
        .str.to_uppercase()
        .str.pad_start(5, "0")
    )


def normalize_industry_code(expr: pl.Expr) -> pl.Expr:
    return expr.cast(pl.Utf8).str.replace_all(r"[^0-9A-Za-z]", "").str.to_uppercase()


def first_non_null_by_siret(frame: pl.LazyFrame) -> pl.LazyFrame:
    value_columns = [
        column for column in frame.collect_schema().names() if column != "siret"
    ]
    return frame.group_by("siret").agg(
        [pl.col(column).drop_nulls().first().alias(column) for column in value_columns]
    )


def build_postal_reference(
    base_codes_postaux_path: Path,
    base_codes_postaux_alt_path: Path,
    laposte_path: Path,
) -> pl.LazyFrame:
    postal_a = scan_csv(base_codes_postaux_path).select(
        [
            normalize_postal_code(pl.col("code_postal")).alias("postal_code"),
            pl.col("nom_de_la_commune").cast(pl.Utf8).alias("city_ref"),
            pl.lit(None, dtype=pl.Utf8).alias("department_ref"),
            pl.col("latitude").cast(pl.Utf8).alias("lat_ref"),
            pl.col("longitude").cast(pl.Utf8).alias("lon_ref"),
        ]
    )

    postal_b = scan_csv(base_codes_postaux_alt_path).select(
        [
            normalize_postal_code(pl.col("Code_postal")).alias("postal_code"),
            pl.col("Nom_de_la_commune").cast(pl.Utf8).alias("city_ref"),
            pl.col("Departement").cast(pl.Utf8).alias("department_ref"),
            pl.lit(None, dtype=pl.Utf8).alias("lat_ref"),
            pl.lit(None, dtype=pl.Utf8).alias("lon_ref"),
        ]
    )

    postal_c = scan_csv(laposte_path).select(
        [
            normalize_postal_code(pl.col("Code_postal")).alias("postal_code"),
            pl.col("Nom_de_la_commune").cast(pl.Utf8).alias("city_ref"),
            pl.lit(None, dtype=pl.Utf8).alias("department_ref"),
            pl.lit(None, dtype=pl.Utf8).alias("lat_ref"),
            pl.lit(None, dtype=pl.Utf8).alias("lon_ref"),
        ]
    )

    return (
        pl.concat([postal_a, postal_b, postal_c], how="vertical")
        .filter(pl.col("postal_code").is_not_null() & (pl.col("postal_code") != ""))
        .group_by("postal_code")
        .agg(
            [
                pl.col("city_ref").drop_nulls().first(),
                pl.col("department_ref").drop_nulls().first(),
                pl.col("lat_ref").drop_nulls().first(),
                pl.col("lon_ref").drop_nulls().first(),
            ]
        )
    )


def build_naf_reference(naf_path: Path) -> pl.LazyFrame:
    return scan_csv(naf_path).select(
        [
            normalize_industry_code(pl.col("code_naf")).alias("industry_code"),
            pl.col("intitule_naf").cast(pl.Utf8).alias("industry_description"),
            pl.col("intitule_naf_65").cast(pl.Utf8).alias("industry_group_65"),
            pl.col("intitule_naf_40").cast(pl.Utf8).alias("industry_sector_40"),
        ]
    )


def build_immatriculees_frame(path: Path) -> pl.LazyFrame:
    return first_non_null_by_siret(
        scan_csv(path)
        .select(
            [
                pl.col("Dénomination").cast(pl.Utf8).alias("company_name"),
                pl.col("Siren").cast(pl.Utf8).str.pad_start(9, "0").alias("siren"),
                pl.col("Nic").cast(pl.Utf8).str.pad_start(5, "0").alias("nic"),
                pl.col("Forme Juridique").cast(pl.Utf8).alias("legal_form"),
                normalize_industry_code(pl.col("Code APE")).alias("industry_code"),
                pl.col("Secteur d'activité").cast(pl.Utf8).alias("activity_sector"),
                pl.col("Adresse").cast(pl.Utf8).alias("address"),
                normalize_postal_code(pl.col("Code postal")).alias("postal_code"),
                pl.col("Ville").cast(pl.Utf8).alias("city"),
                pl.col("Num. dept.").cast(pl.Utf8).alias("department_code"),
                pl.col("Département").cast(pl.Utf8).alias("department_name"),
                pl.col("Région").cast(pl.Utf8).alias("region"),
                pl.col("Date immatriculation").cast(pl.Utf8).alias("registration_date"),
                pl.col("Date radiation").cast(pl.Utf8).alias("radiation_date"),
                pl.col("Statut").cast(pl.Utf8).alias("status"),
                pl.col("Geolocalisation").cast(pl.Utf8).alias("geolocation"),
            ]
        )
        .with_columns(pl.concat_str([pl.col("siren"), pl.col("nic")]).alias("siret"))
    )


def build_rge_frame(path: Path) -> pl.LazyFrame:
    return first_non_null_by_siret(
        scan_csv(path).select(
            [
                pl.col("siret").cast(pl.Utf8).str.pad_start(14, "0").alias("siret"),
                pl.col("nom_entreprise").cast(pl.Utf8).alias("rge_company_name"),
                pl.col("adresse").cast(pl.Utf8).alias("rge_address"),
                normalize_postal_code(pl.col("code_postal")).alias("rge_postal_code"),
                pl.col("commune").cast(pl.Utf8).alias("rge_city"),
                pl.col("latitude").cast(pl.Utf8).alias("rge_latitude"),
                pl.col("longitude").cast(pl.Utf8).alias("rge_longitude"),
                pl.col("telephone").cast(pl.Utf8).alias("rge_phone"),
                pl.col("email").cast(pl.Utf8).alias("rge_email"),
                pl.col("site_internet").cast(pl.Utf8).alias("rge_website"),
                pl.col("code_qualification")
                .cast(pl.Utf8)
                .alias("rge_code_qualification"),
                pl.col("nom_qualification")
                .cast(pl.Utf8)
                .alias("rge_name_qualification"),
                pl.col("url_qualification")
                .cast(pl.Utf8)
                .alias("rge_url_qualification"),
                pl.col("nom_certificat").cast(pl.Utf8).alias("rge_certificate_name"),
                pl.col("domaine").cast(pl.Utf8).alias("rge_domain"),
                pl.col("meta_domaine").cast(pl.Utf8).alias("rge_meta_domain"),
                pl.col("organisme").cast(pl.Utf8).alias("rge_organism"),
                pl.col("particulier").cast(pl.Utf8).alias("rge_particulier"),
                pl.col("lien_date_debut").cast(pl.Utf8).alias("rge_start_date"),
                pl.col("lien_date_fin").cast(pl.Utf8).alias("rge_end_date"),
            ]
        )
    )


def build_generic_siret_frame(path: Path) -> pl.LazyFrame | None:
    schema_names = scan_table(path).collect_schema().names()
    if "siret" not in schema_names:
        return None

    def pick(
        candidates: list[str],
        alias: str,
        normalizer: Callable[[pl.Expr], pl.Expr] | None = None,
    ) -> pl.Expr:
        for candidate in candidates:
            if candidate in schema_names:
                expr = pl.col(candidate)
                if normalizer is not None:
                    expr = normalizer(expr)
                else:
                    expr = expr.cast(pl.Utf8)
                return expr.alias(alias)
        return pl.lit(None, dtype=pl.Utf8).alias(alias)

    select_columns = [
        pl.col("siret").cast(pl.Utf8).str.pad_start(14, "0").alias("siret"),
        pick(
            [
                "company_name",
                "nom_entreprise",
                "name",
                "brand",
                "operator",
                "raison_sociale",
                "denomination",
                "nom",
            ],
            "source_company_name",
        ),
        pick(["address", "adresse"], "source_address"),
        pick(
            [
                "city",
                "com_nom",
                "commune",
                "libelleCommuneEtablissement",
                "Ville",
                "nom_de_la_commune",
            ],
            "source_city",
        ),
        pick(
            ["postal_code", "code_postal", "codePostalEtablissement", "Code_postal"],
            "source_postal_code",
            normalize_postal_code,
        ),
        pick(["website", "site_internet", "url"], "source_website"),
        pick(["phone", "telephone"], "source_phone"),
        pick(["email"], "source_email"),
        pick(
            [
                "activitePrincipaleEtablissement",
                "code_naf",
                "naf",
                "Code APE",
                "Code_APE",
            ],
            "source_industry_code",
            normalize_industry_code,
        ),
        pick(
            ["dateCreationUniteLegale", "date_etablissement", "date_entreprise"],
            "source_registration_date",
        ),
        pl.lit(path.name).alias("source_file"),
    ]

    return first_non_null_by_siret(scan_table(path).select(select_columns))


def build_shop_craft_frame(path: Path) -> pl.LazyFrame:
    return first_non_null_by_siret(
        scan_csv(path).select(
            [
                pl.col("siret").cast(pl.Utf8).str.pad_start(14, "0").alias("siret"),
                pl.coalesce(
                    [
                        pl.col("name"),
                        pl.col("brand"),
                        pl.col("operator"),
                    ]
                )
                .cast(pl.Utf8)
                .str.strip_chars()
                .alias("shop_company_name"),
                pl.col("address").cast(pl.Utf8).alias("shop_address"),
                pl.col("com_nom").cast(pl.Utf8).alias("shop_city"),
                pl.col("website").cast(pl.Utf8).alias("shop_website"),
                pl.col("phone").cast(pl.Utf8).alias("shop_phone"),
                pl.col("email").cast(pl.Utf8).alias("shop_email"),
                pl.col("type").cast(pl.Utf8).alias("shop_type"),
                pl.col("opening_hours").cast(pl.Utf8).alias("shop_opening_hours"),
                pl.col("last_update").cast(pl.Utf8).alias("shop_last_update"),
            ]
        )
    )


def build_prospecting_frame(path: Path) -> pl.LazyFrame:
    return first_non_null_by_siret(
        scan_table(path).select(
            [
                pl.col("siret").cast(pl.Utf8).str.pad_start(14, "0").alias("siret"),
                pl.col("company_name").cast(pl.Utf8).alias("pros_company_name"),
                normalize_industry_code(
                    pl.col("activitePrincipaleEtablissement")
                ).alias("pros_industry_code"),
                pl.col("trancheEffectifsEtablissement")
                .cast(pl.Utf8)
                .alias("pros_staff_size"),
                pl.col("categorieEntreprise").cast(pl.Utf8).alias("pros_company_type"),
                pl.col("address").cast(pl.Utf8).alias("pros_address"),
                pl.col("libelleCommuneEtablissement").cast(pl.Utf8).alias("pros_city"),
                normalize_postal_code(pl.col("codePostalEtablissement")).alias(
                    "pros_postal_code"
                ),
                pl.col("dateCreationUniteLegale")
                .cast(pl.Utf8)
                .alias("pros_creation_date"),
                pl.col("coordonneeLambertAbscisseEtablissement")
                .cast(pl.Utf8)
                .alias("pros_x_lambert_93"),
                pl.col("coordonneeLambertOrdonneeEtablissement")
                .cast(pl.Utf8)
                .alias("pros_y_lambert_93"),
            ]
        )
    )


def build_stock_reference(
    etablissement_path: Path, unite_legale_path: Path
) -> pl.LazyFrame:
    etablissement = scan_table(etablissement_path).filter(
        (pl.col("statutDiffusionEtablissement") == "O")
        & (pl.col("etatAdministratifEtablissement") == "A")
    )
    unite_legale = scan_table(unite_legale_path).filter(
        (pl.col("statutDiffusionUniteLegale") == "O")
        & (pl.col("etatAdministratifUniteLegale") == "A")
    )

    etablissement_clean = etablissement.select(
        [
            pl.col("siret").cast(pl.Utf8).str.pad_start(14, "0").alias("siret"),
            pl.col("siren").cast(pl.Utf8).str.pad_start(9, "0").alias("stock_siren"),
            normalize_industry_code(pl.col("activitePrincipaleEtablissement")).alias(
                "stock_industry_code"
            ),
            pl.col("trancheEffectifsEtablissement")
            .cast(pl.Utf8)
            .alias("stock_staff_size"),
            normalize_postal_code(pl.col("codePostalEtablissement")).alias(
                "stock_postal_code"
            ),
            pl.col("libelleCommuneEtablissement").cast(pl.Utf8).alias("stock_city"),
            pl.format(
                "{} {} {} {} {}",
                pl.col("numeroVoieEtablissement").fill_null(""),
                pl.col("typeVoieEtablissement").fill_null(""),
                pl.col("libelleVoieEtablissement").fill_null(""),
                pl.col("codePostalEtablissement").fill_null(""),
                pl.col("libelleCommuneEtablissement").fill_null(""),
            )
            .str.replace_all(r"\s+", " ")
            .str.strip_chars()
            .alias("stock_address"),
            pl.col("coordonneeLambertAbscisseEtablissement")
            .cast(pl.Utf8)
            .alias("stock_x_lambert_93"),
            pl.col("coordonneeLambertOrdonneeEtablissement")
            .cast(pl.Utf8)
            .alias("stock_y_lambert_93"),
        ]
    )

    unite_legale_clean = unite_legale.select(
        [
            pl.col("siren").cast(pl.Utf8).str.pad_start(9, "0").alias("siren"),
            pl.coalesce(
                [
                    pl.col("denominationUniteLegale"),
                    pl.col("nomUniteLegale"),
                    pl.format(
                        "{} {}",
                        pl.col("prenom1UniteLegale").fill_null(""),
                        pl.col("nomUniteLegale").fill_null(""),
                    ),
                ]
            )
            .cast(pl.Utf8)
            .str.strip_chars()
            .alias("stock_company_name"),
            pl.col("categorieEntreprise").cast(pl.Utf8).alias("stock_company_type"),
            pl.col("dateCreationUniteLegale")
            .cast(pl.Utf8)
            .alias("stock_creation_date"),
        ]
    )

    return first_non_null_by_siret(
        etablissement_clean.join(
            unite_legale_clean,
            left_on="stock_siren",
            right_on="siren",
            how="left",
        )
    )


def build_complete_dataset(
    entreprises_immatriculees_path: Path,
    entreprises_rge_path: Path,
    shop_craft_office_path: Path,
    naf_path: Path,
    base_codes_postaux_path: Path,
    base_codes_postaux_alt_path: Path,
    laposte_path: Path,
    prospecting_leads_path: Path,
    stock_etablissement_path: Path,
    stock_unite_legale_path: Path,
) -> pl.LazyFrame:
    immatriculees = build_immatriculees_frame(entreprises_immatriculees_path)
    rge = build_rge_frame(entreprises_rge_path)
    shop_craft = build_shop_craft_frame(shop_craft_office_path)
    naf = build_naf_reference(naf_path)
    prospecting = build_prospecting_frame(prospecting_leads_path)
    stock_reference = build_stock_reference(
        stock_etablissement_path,
        stock_unite_legale_path,
    )
    postal_reference = build_postal_reference(
        base_codes_postaux_path,
        base_codes_postaux_alt_path,
        laposte_path,
    )

    discovered_sources = []
    for path in discover_data_files(DATA_DIR):
        if path.resolve() in {
            OUTPUT_PATH.resolve(),
            entreprises_immatriculees_path.resolve(),
            entreprises_rge_path.resolve(),
            shop_craft_office_path.resolve(),
            naf_path.resolve(),
            base_codes_postaux_path.resolve(),
            base_codes_postaux_alt_path.resolve(),
            laposte_path.resolve(),
            prospecting_leads_path.resolve(),
            stock_etablissement_path.resolve(),
            stock_unite_legale_path.resolve(),
        }:
            continue
        generic_frame = build_generic_siret_frame(path)
        if generic_frame is not None:
            discovered_sources.append(generic_frame)
        else:
            logger.info("Skipping non-SIRET source: %s", path.name)

    discovered_sources_frame = (
        pl.concat(discovered_sources, how="vertical")
        if discovered_sources
        else pl.DataFrame(
            schema={
                "siret": pl.Utf8,
                "source_company_name": pl.Utf8,
                "source_address": pl.Utf8,
                "source_city": pl.Utf8,
                "source_postal_code": pl.Utf8,
                "source_website": pl.Utf8,
                "source_phone": pl.Utf8,
                "source_email": pl.Utf8,
                "source_industry_code": pl.Utf8,
                "source_registration_date": pl.Utf8,
                "source_file": pl.Utf8,
            }
        ).lazy()
    )

    siret_universe = (
        pl.concat(
            [
                immatriculees.select("siret"),
                rge.select("siret"),
                shop_craft.select("siret"),
                stock_reference.select("siret"),
                prospecting.select("siret"),
                *[source.select("siret") for source in discovered_sources],
            ],
            how="vertical",
        )
        .filter(pl.col("siret").is_not_null() & (pl.col("siret") != ""))
        .unique()
    )

    return (
        siret_universe.join(immatriculees, on="siret", how="left")
        .join(rge, on="siret", how="left")
        .join(shop_craft, on="siret", how="left")
        .join(stock_reference, on="siret", how="left")
        .join(prospecting, on="siret", how="left")
        .join(discovered_sources_frame, on="siret", how="left")
        .with_columns(
            [
                pl.coalesce(
                    [
                        pl.col("industry_code"),
                        pl.col("stock_industry_code"),
                        pl.col("pros_industry_code"),
                        pl.col("source_industry_code"),
                    ]
                ).alias("industry_code"),
                pl.coalesce(
                    [
                        pl.col("company_name"),
                        pl.col("rge_company_name"),
                        pl.col("shop_company_name"),
                        pl.col("stock_company_name"),
                        pl.col("pros_company_name"),
                        pl.col("source_company_name"),
                    ]
                ).alias("company_name"),
                pl.coalesce(
                    [
                        pl.col("address"),
                        pl.col("rge_address"),
                        pl.col("shop_address"),
                        pl.col("stock_address"),
                        pl.col("pros_address"),
                        pl.col("source_address"),
                    ]
                ).alias("address"),
                pl.coalesce(
                    [
                        pl.col("city"),
                        pl.col("rge_city"),
                        pl.col("shop_city"),
                        pl.col("stock_city"),
                        pl.col("pros_city"),
                        pl.col("source_city"),
                    ]
                ).alias("city"),
                pl.coalesce(
                    [
                        pl.col("postal_code"),
                        pl.col("stock_postal_code"),
                        pl.col("pros_postal_code"),
                        pl.col("source_postal_code"),
                    ]
                ).alias("postal_code"),
                pl.coalesce(
                    [
                        pl.col("siren"),
                        pl.col("stock_siren"),
                        pl.col("siret").str.slice(0, 9),
                    ]
                ).alias("siren"),
                pl.coalesce(
                    [
                        pl.col("registration_date"),
                        pl.col("stock_creation_date"),
                        pl.col("pros_creation_date"),
                        pl.col("source_registration_date"),
                    ]
                ).alias("registration_date"),
                pl.coalesce(
                    [
                        pl.col("pros_staff_size"),
                        pl.col("stock_staff_size"),
                    ]
                ).alias("staff_size_range"),
                pl.coalesce(
                    [
                        pl.col("pros_company_type"),
                        pl.col("stock_company_type"),
                    ]
                ).alias("company_type"),
                pl.coalesce(
                    [
                        pl.col("stock_x_lambert_93"),
                        pl.col("pros_x_lambert_93"),
                    ]
                ).alias("x_lambert_93"),
                pl.coalesce(
                    [
                        pl.col("stock_y_lambert_93"),
                        pl.col("pros_y_lambert_93"),
                    ]
                ).alias("y_lambert_93"),
                pl.coalesce(
                    [
                        pl.col("shop_website"),
                        pl.col("rge_website"),
                        pl.col("source_website"),
                    ]
                ).alias("website"),
                pl.coalesce(
                    [pl.col("shop_phone"), pl.col("rge_phone"), pl.col("source_phone")]
                ).alias("phone"),
                pl.coalesce(
                    [pl.col("shop_email"), pl.col("rge_email"), pl.col("source_email")]
                ).alias("email"),
            ]
        )
        .join(naf, on="industry_code", how="left")
        .join(postal_reference, on="postal_code", how="left")
        .with_columns(
            [
                pl.coalesce([pl.col("company_name"), pl.col("rge_company_name")]).alias(
                    "company_name"
                ),
                pl.coalesce([pl.col("address"), pl.col("rge_address")]).alias(
                    "address"
                ),
                pl.coalesce(
                    [pl.col("city"), pl.col("rge_city"), pl.col("city_ref")]
                ).alias("city"),
            ]
        )
    )


def main() -> None:
    logger.info("Dataset build started")
    discovered_files = discover_data_files(DATA_DIR)
    logger.info("Discovered %s CSV/parquet inputs under DATA", len(discovered_files))
    logger.info("Inputs found: %s", ", ".join(path.name for path in discovered_files))
    validate_paths(
        [
            ENTREPRISES_IMMATRICULEES_PATH,
            ENTREPRISES_RGE_PATH,
            SHOP_CRAFT_OFFICE_PATH,
            NAF_PATH,
            BASE_CODES_POSTAUX_PATH,
            BASE_CODES_POSTAUX_ALT_PATH,
            LAPOSTE_PATH,
            PROSPECTING_LEADS_PATH,
            STOCK_ETABLISSEMENT_PATH,
            STOCK_UNITE_LEGALE_PATH,
        ]
    )
    logger.info("All input files found")

    dataset = build_complete_dataset(
        entreprises_immatriculees_path=ENTREPRISES_IMMATRICULEES_PATH,
        entreprises_rge_path=ENTREPRISES_RGE_PATH,
        shop_craft_office_path=SHOP_CRAFT_OFFICE_PATH,
        naf_path=NAF_PATH,
        base_codes_postaux_path=BASE_CODES_POSTAUX_PATH,
        base_codes_postaux_alt_path=BASE_CODES_POSTAUX_ALT_PATH,
        laposte_path=LAPOSTE_PATH,
        prospecting_leads_path=PROSPECTING_LEADS_PATH,
        stock_etablissement_path=STOCK_ETABLISSEMENT_PATH,
        stock_unite_legale_path=STOCK_UNITE_LEGALE_PATH,
    )
    output_path = OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing output as %s to %s", OUTPUT_FORMAT, output_path)
    if OUTPUT_FORMAT == "parquet":
        dataset.sink_parquet(output_path)
    else:
        dataset.sink_csv(output_path)

    output_lf = (
        pl.scan_parquet(output_path)
        if OUTPUT_FORMAT == "parquet"
        else pl.scan_csv(output_path)
    )
    output_stats = output_lf.select(
        [
            pl.len().alias("rows"),
            pl.n_unique("siret").alias("unique_siret"),
        ]
    ).collect()
    logger.info("Dataset build finished: %s", output_stats.to_dicts()[0])


if __name__ == "__main__":
    main()
