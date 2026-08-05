from __future__ import annotations

import shutil
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer
import rasterio
from pystac_client import Client

import run_pipeline as rp
import run_pipeline_v2 as v2

COLLECTION_ID = "modis-13Q1-061"
NDVI_ASSET = "250m_16_days_NDVI"
QA_ASSET = "250m_16_days_pixel_reliability"
MIN_EXPECTED_COMPOSITES = 45
MIN_LATEST_DATE = date(2026, 5, 25)


def item_start_datetime(item: Any) -> datetime:
    raw = item.properties.get("start_datetime") or item.properties.get("datetime")
    if raw:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    if item.datetime is not None:
        return item.datetime
    raise RuntimeError(f"STAC item has no usable start datetime: {item.id}")


def gdal_env() -> dict[str, str]:
    return {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff",
        "GDAL_HTTP_MAX_RETRY": "6",
        "GDAL_HTTP_RETRY_DELAY": "10",
        "GDAL_HTTP_CONNECTTIMEOUT": "30",
        "GDAL_HTTP_TIMEOUT": "600",
    }


def build_remote_vrt(urls: list[str], vrt_path: Path, nodata: int) -> Path:
    if not urls:
        raise RuntimeError(f"No source URLs for {vrt_path.name}")
    vrt_path.parent.mkdir(parents=True, exist_ok=True)
    source_list = vrt_path.with_suffix(".txt")
    source_list.write_text("\n".join(f"/vsicurl/{url}" for url in urls) + "\n", encoding="utf-8")
    rp.run(
        [
            "gdalbuildvrt",
            "-overwrite",
            "-resolution",
            "highest",
            "-srcnodata",
            str(nodata),
            "-vrtnodata",
            str(nodata),
            "-input_file_list",
            str(source_list),
            str(vrt_path),
        ],
        env=gdal_env(),
    )
    return source_list


def warp_vrt(vrt_path: Path, out_path: Path, nodata: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rp.run(
        [
            "gdalwarp",
            "-overwrite",
            "-multi",
            "-wo",
            "NUM_THREADS=ALL_CPUS",
            "-t_srs",
            rp.TARGET_CRS,
            "-cutline",
            str(rp.INPUT / "study_area.shp"),
            "-crop_to_cutline",
            "-tr",
            "250",
            "250",
            "-tap",
            "-r",
            "near",
            "-srcnodata",
            str(nodata),
            "-dstnodata",
            str(nodata),
            "-co",
            "TILED=YES",
            "-co",
            "COMPRESS=DEFLATE",
            "-co",
            "BIGTIFF=IF_SAFER",
            str(vrt_path),
            str(out_path),
        ],
        env=gdal_env(),
    )


def quality_screen_and_scale(raw_ndvi: Path, qa_path: Path, final_path: Path) -> dict[str, float | int]:
    with rasterio.open(raw_ndvi) as ndvi_src, rasterio.open(qa_path) as qa_src:
        if (
            ndvi_src.width != qa_src.width
            or ndvi_src.height != qa_src.height
            or ndvi_src.transform != qa_src.transform
            or ndvi_src.crs != qa_src.crs
        ):
            raise RuntimeError("NDVI and pixel-reliability rasters are not aligned")

        raw = ndvi_src.read(1).astype("float32")
        qa = qa_src.read(1).astype("int16")
        invalid = ~np.isfinite(raw)
        if ndvi_src.nodata is not None:
            invalid |= raw == ndvi_src.nodata
        if qa_src.nodata is not None:
            invalid |= qa == qa_src.nodata
        invalid |= (raw < -2000) | (raw > 10000)
        invalid |= ~np.isin(qa, [0, 1])

        scaled = raw * 0.0001
        scaled[invalid] = -9999.0
        valid = scaled != -9999.0
        profile = ndvi_src.profile.copy()
        profile.update(dtype="float32", count=1, nodata=-9999.0, compress="deflate", tiled=True)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(final_path, "w", **profile) as dst:
            dst.write(scaled, 1)
            dst.update_tags(
                product="MOD13Q1.061",
                variable="NDVI",
                scale_applied="0.0001",
                quality_filter="pixel_reliability in {0,1}",
                valid_range="-0.2 to 1.0",
            )

        if not valid.any():
            raise RuntimeError("Quality screening removed every NDVI pixel")
        return {
            "valid_pixels": int(valid.sum()),
            "valid_fraction": float(valid.mean()),
            "mean_ndvi": float(np.mean(scaled[valid])),
            "min_ndvi": float(np.min(scaled[valid])),
            "max_ndvi": float(np.max(scaled[valid])),
        }


def download_modis_ndvi_v3(study: gpd.GeoDataFrame) -> list[Path]:
    source = "Microsoft Planetary Computer / NASA MOD13Q1 V6.1"
    output_root = rp.OUTPUT / "05_ndvi"
    work_root = rp.WORK / "mod13q1_v3"
    if output_root.exists():
        shutil.rmtree(output_root)
    if work_root.exists():
        shutil.rmtree(work_root)
    output_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    try:
        catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
        search = catalog.search(
            collections=[COLLECTION_ID],
            bbox=list(study.total_bounds),
            datetime="2024-04-01/2026-06-30",
        )
        grouped: dict[date, list[Any]] = defaultdict(list)
        start_bound = date.fromisoformat(rp.START_DATE)
        end_bound = date.fromisoformat(rp.END_DATE)
        for item in search.items():
            start_date = item_start_datetime(item).astimezone(timezone.utc).date()
            if start_bound <= start_date <= end_bound:
                grouped[start_date].append(item)

        dates = sorted(grouped)
        if len(dates) < MIN_EXPECTED_COMPOSITES:
            raise RuntimeError(
                f"Only {len(dates)} unique MOD13Q1 composite dates were found; expected at least {MIN_EXPECTED_COMPOSITES}"
            )
        if dates[0] < start_bound:
            raise RuntimeError(f"Earliest composite {dates[0]} predates requested start {start_bound}")
        if dates[-1] < MIN_LATEST_DATE:
            raise RuntimeError(f"Latest composite {dates[-1]} is too early; expected at least {MIN_LATEST_DATE}")

        inventory: list[dict[str, Any]] = []
        final_paths: list[Path] = []
        for index, composite_date in enumerate(dates, start=1):
            date_text = composite_date.strftime("%Y%m%d")
            day_work = work_root / date_text
            day_work.mkdir(parents=True, exist_ok=True)
            ndvi_vrt = day_work / "ndvi.vrt"
            qa_vrt = day_work / "qa.vrt"
            raw_ndvi = day_work / "ndvi_raw_250m.tif"
            qa_clip = output_root / "qa" / f"MOD13Q1_pixel_reliability_{date_text}.tif"
            final_ndvi = output_root / "ndvi_qc" / f"MOD13Q1_NDVI_QC_{date_text}.tif"

            created: list[Path] = [qa_clip, final_ndvi]
            try:
                ndvi_urls: list[str] = []
                qa_urls: list[str] = []
                item_ids: list[str] = []
                for item in grouped[composite_date]:
                    if NDVI_ASSET not in item.assets or QA_ASSET not in item.assets:
                        raise RuntimeError(f"Required assets missing from {item.id}: {list(item.assets)}")
                    ndvi_urls.append(planetary_computer.sign_url(item.assets[NDVI_ASSET].href))
                    qa_urls.append(planetary_computer.sign_url(item.assets[QA_ASSET].href))
                    item_ids.append(item.id)

                ndvi_list = build_remote_vrt(ndvi_urls, ndvi_vrt, -3000)
                qa_list = build_remote_vrt(qa_urls, qa_vrt, -1)
                warp_vrt(ndvi_vrt, raw_ndvi, -3000)
                warp_vrt(qa_vrt, qa_clip, -1)
                stats = quality_screen_and_scale(raw_ndvi, qa_clip, final_ndvi)
                if final_ndvi.stat().st_size < 1024 or qa_clip.stat().st_size < 1024:
                    raise RuntimeError("Output file is unexpectedly small")

                final_paths.append(final_ndvi)
                inventory.append(
                    {
                        "composite_start_date": composite_date.isoformat(),
                        "item_ids": "|".join(item_ids),
                        "tile_count": len(item_ids),
                        "ndvi_file": str(final_ndvi.relative_to(rp.ROOT)),
                        "qa_file": str(qa_clip.relative_to(rp.ROOT)),
                        **stats,
                    }
                )
                rp.log(f"MOD13Q1 {index}/{len(dates)} complete: {date_text}, tiles={len(item_ids)}")
                for temp in [ndvi_vrt, qa_vrt, ndvi_list, qa_list, raw_ndvi]:
                    temp.unlink(missing_ok=True)
                day_work.rmdir()
            except Exception:
                for path in created:
                    path.unlink(missing_ok=True)
                raise

        inventory_path = output_root / "ndvi_inventory.csv"
        pd.DataFrame(inventory).to_csv(inventory_path, index=False, encoding="utf-8-sig")
        rp.add_status(
            "MODIS MOD13Q1 NDVI",
            "downloaded",
            source,
            str(output_root.relative_to(rp.ROOT)),
            f"{len(final_paths)} unique quality-screened 16-day composites; {dates[0]} to {dates[-1]}",
        )
        return final_paths
    except Exception as exc:
        rp.add_status("MODIS MOD13Q1 NDVI", "failed", source, note=repr(exc))
        raise


def validate_final_package(
    feature_file: Path,
    rasters: dict[str, Path],
    ndvi_paths: list[Path],
) -> None:
    features = pd.read_csv(feature_file)
    if len(features) != 22:
        raise RuntimeError(f"Feature table contains {len(features)} rows rather than 22")

    required_rasters = {"dem", "slope", "worldcover", "built", "cropland", "population"}
    missing_rasters = sorted(required_rasters - set(rasters))
    if missing_rasters:
        raise RuntimeError(f"Required raster outputs are missing: {missing_rasters}")

    if len(ndvi_paths) < MIN_EXPECTED_COMPOSITES:
        raise RuntimeError(f"Only {len(ndvi_paths)} NDVI composites were produced")
    ndvi_dates = sorted(date.fromisoformat(path.stem.rsplit("_", 1)[-1]) for path in ndvi_paths)
    if ndvi_dates[0] < date.fromisoformat(rp.START_DATE):
        raise RuntimeError(f"NDVI series starts too early: {ndvi_dates[0]}")
    if ndvi_dates[-1] < MIN_LATEST_DATE:
        raise RuntimeError(f"NDVI series ends too early: {ndvi_dates[-1]}")

    ratio_columns = ["built_ratio_500m", "built_ratio_1km", "cropland_ratio_1km"]
    for column in ratio_columns:
        if column not in features:
            raise RuntimeError(f"Feature column missing: {column}")
        values = pd.to_numeric(features[column], errors="coerce")
        if values.isna().all() or ((values < 0) | (values > 1)).any():
            raise RuntimeError(f"Invalid values in {column}")
        if values.nunique(dropna=True) <= 1:
            raise RuntimeError(f"{column} is constant across all 22 stations")

    required_columns = [
        "population_sum_1km",
        "population_density_1km_per_km2",
        "ndvi_mean_1km_202405_202606",
        "ndvi_latest_1km",
        "ndvi_valid_observation_count",
    ]
    for column in required_columns:
        if column not in features or features[column].isna().all():
            raise RuntimeError(f"Required feature is missing or empty: {column}")
    if (features["ndvi_valid_observation_count"] < 30).any():
        raise RuntimeError("At least one station has fewer than 30 valid NDVI composites")

    core_datasets = {
        "Copernicus DEM GLO-30",
        "ESA WorldCover 2021",
        "WorldPop 2025 constrained population",
        "OpenStreetMap roads",
        "OpenStreetMap buildings",
        "MODIS MOD13Q1 NDVI",
        "Station spatial feature table",
    }
    failures = [status for status in rp.STATUSES if status.dataset in core_datasets and status.status != "downloaded" and status.status != "generated"]
    if failures:
        raise RuntimeError(f"Core dataset failures remain: {failures}")

    rp.add_status(
        "Automated quality checks",
        "passed",
        "Pipeline validation",
        str(feature_file.relative_to(rp.ROOT)),
        f"22 stations; {len(ndvi_paths)} NDVI composites; population present; nonconstant land-cover ratios",
    )


def main() -> None:
    for directory in [rp.OUTPUT, rp.RELEASE, rp.WORK / "mod13q1_v3"]:
        if directory.exists():
            shutil.rmtree(directory)
    rp.OUTPUT.mkdir(exist_ok=True)
    rp.RELEASE.mkdir(exist_ok=True)
    rp.WORK.mkdir(exist_ok=True)
    rp.STATUSES.clear()

    study, stations = rp.ensure_inputs()
    rp.log(f"Study bounds: {study.total_bounds.tolist()}")
    rasters = v2.fixed_download_core_rasters()
    vectors = rp.download_osm(study)
    ndvi_paths = download_modis_ndvi_v3(study)
    feature_file = rp.build_features(stations, rasters, vectors, ndvi_paths)
    validate_final_package(feature_file, rasters, ndvi_paths)

    rp.add_status(
        "CAMS 2024-2026 emissions",
        "not_downloaded",
        "Copernicus ADS",
        note="Requires ADS credentials; future years are forecast-oriented extensions, not local measured emissions",
    )
    rp.add_status(
        "EDGAR 2022 emissions",
        "deferred",
        "JRC EDGAR",
        note="Coarse 0.1-degree background excluded from the core package to avoid block artefacts",
    )
    rp.add_status(
        "Local industrial emission sources",
        "not_downloaded",
        "Pollution permit platform / local departments",
        note="Requires manual verification or departmental data",
    )
    rp.write_documentation()
    package = rp.package()
    rp.log(f"FINAL PACKAGE: {package} size={package.stat().st_size}")


if __name__ == "__main__":
    main()
