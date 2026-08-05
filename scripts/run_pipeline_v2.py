from __future__ import annotations

import math
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer
import rasterio
from pystac_client import Client
from rasterio.warp import Resampling
from shapely.geometry import mapping

import run_pipeline as rp


def make_binary_raster(source: Path, target: Path, class_value: int) -> None:
    """Create a 0/1 class raster while preserving zero as valid data."""
    with rasterio.open(source) as src:
        arr = src.read(1)
        nodata = src.nodata
        valid = np.isfinite(arr)
        if nodata is not None:
            valid &= arr != nodata
        result = np.full(arr.shape, 255, dtype=np.uint8)
        result[valid] = (arr[valid] == class_value).astype(np.uint8)
        profile = src.profile.copy()
        profile.update(dtype="uint8", count=1, nodata=255, compress="deflate", tiled=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(target, "w", **profile) as dst:
            dst.write(result, 1)


def download_file(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "curl", "-fL", "--retry", "6", "--retry-all-errors", "--retry-delay", "15",
        "--connect-timeout", "30", "--speed-time", "180", "--speed-limit", "1024",
        "--continue-at", "-", "-o", str(target), url,
    ]
    rp.run(cmd)


def fixed_download_core_rasters() -> dict[str, Path]:
    cutline = rp.INPUT / "study_area.shp"
    rasters: dict[str, Path] = {}

    dem_url = "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N39_00_E116_00_DEM/Copernicus_DSM_COG_10_N39_00_E116_00_DEM.tif"
    dem = rp.OUTPUT / "01_DEM" / "copernicus_dem_glo30_30m_32650.tif"
    try:
        rp.gdal_clip_cog(dem_url, dem, cutline, 30, 30, "bilinear", "Float32", None, "-9999")
        slope = rp.OUTPUT / "01_DEM" / "slope_degrees_30m_32650.tif"
        rp.run(["gdaldem", "slope", str(dem), str(slope), "-compute_edges", "-of", "GTiff", "-co", "TILED=YES", "-co", "COMPRESS=DEFLATE"])
        rasters["dem"] = dem
        rasters["slope"] = slope
        rp.add_status("Copernicus DEM GLO-30", "downloaded", dem_url, str(dem.relative_to(rp.ROOT)))
    except Exception as exc:
        rp.add_status("Copernicus DEM GLO-30", "failed", dem_url, note=repr(exc))

    wc_url = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_N39E114_Map.tif"
    wc = rp.OUTPUT / "02_landcover" / "esa_worldcover_2021_10m_32650.tif"
    try:
        rp.gdal_clip_cog(wc_url, wc, cutline, 10, 10, "near", "Byte", "0", "0")
        built = rp.OUTPUT / "02_landcover" / "built_up_binary_10m.tif"
        crop = rp.OUTPUT / "02_landcover" / "cropland_binary_10m.tif"
        make_binary_raster(wc, built, 50)
        make_binary_raster(wc, crop, 40)
        rasters.update(worldcover=wc, built=built, cropland=crop)
        rp.add_status("ESA WorldCover 2021", "downloaded", wc_url, str(wc.relative_to(rp.ROOT)), "Binary class rasters retain 0 as valid data; NoData=255")
    except Exception as exc:
        rp.add_status("ESA WorldCover 2021", "failed", wc_url, note=repr(exc))

    population_urls = [
        "https://worldpop-public-data.soton.ac.uk/GIS/Population/Global_2015_2030/R2025A/2025/CHN/v1/100m/constrained/chn_pop_2025_CN_100m_R2025A_v1.tif",
        "https://data.worldpop.org/GIS/Population/Global_2015_2030/R2025A/2025/CHN/v1/100m/constrained/chn_pop_2025_CN_100m_R2025A_v1.tif",
    ]
    raw_pop = rp.WORK / "chn_pop_2025_CN_100m_R2025A_v1.tif"
    pop = rp.OUTPUT / "03_population" / "worldpop_2025_constrained_100m_32650.tif"
    population_error: Exception | None = None
    used_url = population_urls[0]
    for url in population_urls:
        used_url = url
        try:
            if raw_pop.exists() and raw_pop.stat().st_size < 100_000_000:
                raw_pop.unlink()
            download_file(url, raw_pop)
            with rasterio.open(raw_pop) as src:
                if src.width < 1000 or src.height < 1000:
                    raise RuntimeError("Downloaded WorldPop file is unexpectedly small or invalid")
            pop.parent.mkdir(parents=True, exist_ok=True)
            rp.run([
                "gdalwarp", "-overwrite", "-multi", "-wo", "NUM_THREADS=ALL_CPUS",
                "-t_srs", rp.TARGET_CRS, "-cutline", str(cutline), "-crop_to_cutline",
                "-tr", "100", "100", "-r", "sum", "-ot", "Float32", "-dstnodata", "-9999",
                "-co", "TILED=YES", "-co", "COMPRESS=DEFLATE", "-co", "BIGTIFF=IF_SAFER",
                str(raw_pop), str(pop),
            ])
            rasters["population"] = pop
            rp.add_status("WorldPop 2025 constrained population", "downloaded", url, str(pop.relative_to(rp.ROOT)), "Modelled population count, not census; sum resampling used during reprojection")
            population_error = None
            break
        except Exception as exc:
            population_error = exc
            rp.log(f"WorldPop source failed: {url}: {exc!r}")
            if raw_pop.exists():
                raw_pop.unlink()
    if population_error is not None:
        rp.add_status("WorldPop 2025 constrained population", "failed", used_url, note=repr(population_error))
    if raw_pop.exists():
        raw_pop.unlink()

    return rasters


def item_datetime(item: Any) -> datetime:
    dt = item.datetime
    if dt is not None:
        return dt
    raw = (
        item.properties.get("start_datetime")
        or item.properties.get("end_datetime")
        or item.properties.get("datetime")
    )
    if not raw:
        raise RuntimeError(f"STAC item has no usable datetime: {item.id}")
    return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))


def apply_pixel_reliability(ndvi_path: Path, qa_path: Path) -> None:
    with rasterio.open(ndvi_path, "r+") as ndvi, rasterio.open(qa_path) as qa:
        if ndvi.width != qa.width or ndvi.height != qa.height or ndvi.transform != qa.transform:
            raise RuntimeError("NDVI and QA clips are not aligned")
        data = ndvi.read(1)
        quality = qa.read(1)
        invalid = (~np.isfinite(quality)) | (quality < 0) | (quality > 1)
        if qa.nodata is not None:
            invalid |= quality == qa.nodata
        data[invalid] = ndvi.nodata if ndvi.nodata is not None else -9999.0
        ndvi.write(data, 1)


def fixed_download_modis_ndvi(study: gpd.GeoDataFrame) -> list[Path]:
    paths: list[Path] = []
    source = "Microsoft Planetary Computer / NASA MOD13Q1 V6.1"
    try:
        catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
        candidates = [
            collection.id
            for collection in catalog.get_collections()
            if "13Q1" in (collection.id + " " + (collection.title or "")).upper()
        ]
        if not candidates:
            raise RuntimeError("MOD13Q1 collection not found in Planetary Computer catalog")
        collection_id = candidates[0]
        rp.log(f"Using Planetary Computer collection: {collection_id}")
        items = list(
            catalog.search(
                collections=[collection_id],
                bbox=list(study.total_bounds),
                datetime=f"{rp.START_DATE}/{rp.END_DATE}",
            ).items()
        )
        if not items:
            raise RuntimeError("No MOD13Q1 items found")
        geom = mapping(study.geometry.union_all())
        records: list[dict[str, Any]] = []
        dated_items = sorted(((item_datetime(item), item) for item in items), key=lambda pair: pair[0])
        for index, (dt, item) in enumerate(dated_items, start=1):
            signed = planetary_computer.sign(item)
            keys = list(signed.assets)
            ndvi_key = next((key for key in keys if "NDVI" in key.upper()), None)
            qa_key = next((key for key in keys if "PIXEL_RELIABILITY" in key.upper() or "RELIABILITY" in key.upper()), None)
            if ndvi_key is None:
                rp.log(f"Skip item without NDVI asset: {item.id}; keys={keys}")
                continue
            date_text = dt.astimezone(timezone.utc).strftime("%Y%m%d")
            ndvi_path = rp.OUTPUT / "05_ndvi" / "ndvi_qc" / f"MOD13Q1_NDVI_QC_{date_text}.tif"
            asset = signed.assets[ndvi_key]
            scale = 0.0001
            raster_bands = asset.extra_fields.get("raster:bands", [])
            if raster_bands and isinstance(raster_bands, list):
                scale = raster_bands[0].get("scale", scale) or scale
            with rasterio.open(asset.href) as src:
                rp.write_raster_clip(src, geom, ndvi_path, float(scale), resampling=Resampling.nearest)
            qa_relative = ""
            quality_applied = False
            if qa_key:
                qa_path = rp.OUTPUT / "05_ndvi" / "qa" / f"MOD13Q1_pixel_reliability_{date_text}.tif"
                with rasterio.open(signed.assets[qa_key].href) as src:
                    rp.write_raster_clip(src, geom, qa_path, 1.0, resampling=Resampling.nearest)
                apply_pixel_reliability(ndvi_path, qa_path)
                qa_relative = str(qa_path.relative_to(rp.ROOT))
                quality_applied = True
            paths.append(ndvi_path)
            records.append({
                "date": date_text,
                "item_id": item.id,
                "ndvi_file": str(ndvi_path.relative_to(rp.ROOT)),
                "qa_file": qa_relative,
                "pixel_reliability_0_or_1_applied": quality_applied,
            })
            rp.log(f"NDVI {index}/{len(dated_items)}: {date_text}")
        if not paths:
            raise RuntimeError("No NDVI clips were produced")
        inventory = rp.OUTPUT / "05_ndvi" / "ndvi_inventory.csv"
        inventory.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(inventory, index=False, encoding="utf-8-sig")
        rp.add_status("MODIS MOD13Q1 NDVI", "downloaded", source, str((rp.OUTPUT / "05_ndvi").relative_to(rp.ROOT)), f"{len(paths)} quality-screened 16-day clips from {rp.START_DATE} to {rp.END_DATE}")
    except Exception as exc:
        rp.add_status("MODIS MOD13Q1 NDVI", "failed", source, note=repr(exc))
    return paths


def validate_outputs(feature_file: Path, rasters: dict[str, Path], ndvi_paths: list[Path]) -> None:
    features = pd.read_csv(feature_file)
    if len(features) != 22:
        raise RuntimeError(f"Feature table has {len(features)} rows instead of 22")
    for column in ["built_ratio_500m", "built_ratio_1km", "cropland_ratio_1km"]:
        if column in features:
            values = pd.to_numeric(features[column], errors="coerce")
            if ((values < 0) | (values > 1)).any():
                raise RuntimeError(f"{column} contains values outside 0-1")
            if values.nunique(dropna=True) <= 1:
                raise RuntimeError(f"{column} is constant across all stations; land-cover feature generation is invalid")
    if "population" not in rasters:
        raise RuntimeError("WorldPop population was not downloaded successfully")
    if not ndvi_paths:
        raise RuntimeError("No MOD13Q1 NDVI files were produced")
    required = ["population_sum_1km", "ndvi_mean_1km_202405_202606"]
    missing = [column for column in required if column not in features]
    if missing:
        raise RuntimeError(f"Required feature columns are missing: {missing}")
    if features[required].isna().all().any():
        raise RuntimeError("Population or NDVI features are entirely missing")
    rp.add_status("Automated quality checks", "passed", "Pipeline validation", str(feature_file.relative_to(rp.ROOT)), "22 rows; nonconstant 0-1 land-cover ratios; population and NDVI present")


def main() -> None:
    if rp.OUTPUT.exists():
        shutil.rmtree(rp.OUTPUT)
    if rp.RELEASE.exists():
        shutil.rmtree(rp.RELEASE)
    rp.OUTPUT.mkdir(exist_ok=True)
    rp.RELEASE.mkdir(exist_ok=True)
    rp.STATUSES.clear()

    study, stations = rp.ensure_inputs()
    rp.log(f"Study bounds: {study.total_bounds.tolist()}")
    rasters = fixed_download_core_rasters()
    vectors = rp.download_osm(study)
    ndvi_paths = fixed_download_modis_ndvi(study)
    feature_file = rp.build_features(stations, rasters, vectors, ndvi_paths)
    validate_outputs(feature_file, rasters, ndvi_paths)
    rp.add_status("CAMS 2024-2026 emissions", "not_downloaded", "Copernicus ADS", note="Requires CDS/ADS API credentials; later years are forecast-oriented extensions rather than local measured inventory")
    rp.add_status("EDGAR 2022 emissions", "deferred", "JRC EDGAR", note="Coarse 0.1 degree background; excluded from core package to avoid block artefacts")
    rp.add_status("Local industrial emission sources", "not_downloaded", "National pollution permit platform / local departments", note="Requires manual verification or departmental data")
    rp.write_documentation()
    package = rp.package()
    rp.log(f"PACKAGE: {package} size={package.stat().st_size}")


if __name__ == "__main__":
    main()
