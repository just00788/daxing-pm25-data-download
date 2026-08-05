from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_geom
from shapely.geometry import LineString, Polygon, mapping

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "inputs"
OUTPUT = ROOT / "output"
RELEASE = ROOT / "release"
WORK = ROOT / "work"
TARGET_CRS = "EPSG:32650"
START_DATE = "2024-05-01"
END_DATE = "2026-06-15"

OUTPUT.mkdir(exist_ok=True)
RELEASE.mkdir(exist_ok=True)
WORK.mkdir(exist_ok=True)

@dataclass
class DatasetStatus:
    dataset: str
    status: str
    source: str
    output: str = ""
    note: str = ""

STATUSES: list[DatasetStatus] = []


def log(msg: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}", flush=True)


def run(cmd: list[str], env: dict[str, str] | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    log("RUN: " + " ".join(cmd))
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    return subprocess.run(cmd, text=True, check=check, env=full_env)


def add_status(dataset: str, status: str, source: str, output: str = "", note: str = "") -> None:
    STATUSES.append(DatasetStatus(dataset, status, source, output, note))
    log(f"STATUS {dataset}: {status} {note}")


def ensure_inputs() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    study = gpd.read_file(INPUT / "study_area.shp")
    stations = gpd.read_file(INPUT / "air_quality_points_22stations.shp")
    if study.crs is None:
        study = study.set_crs(4326)
    if stations.crs is None:
        stations = stations.set_crs(4326)
    study = study.to_crs(4326)
    stations = stations.to_crs(4326)
    if study.empty or stations.empty:
        raise RuntimeError("Study area or station input is empty")
    study.to_file(OUTPUT / "study_area.gpkg", layer="study_area", driver="GPKG")
    stations.to_file(OUTPUT / "stations_22.gpkg", layer="stations", driver="GPKG")
    return study, stations


def gdal_clip_cog(url: str, out_path: Path, cutline: Path, xres: float, yres: float, resampling: str = "bilinear", dtype: str | None = None, src_nodata: str | None = None, dst_nodata: str | None = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "gdalwarp", "-overwrite", "-multi", "-wo", "NUM_THREADS=ALL_CPUS",
        "-t_srs", TARGET_CRS, "-cutline", str(cutline), "-crop_to_cutline",
        "-tr", str(xres), str(yres), "-r", resampling,
        "-co", "TILED=YES", "-co", "COMPRESS=DEFLATE", "-co", "BIGTIFF=IF_SAFER",
    ]
    if dtype:
        cmd += ["-ot", dtype]
    if src_nodata is not None:
        cmd += ["-srcnodata", src_nodata]
    if dst_nodata is not None:
        cmd += ["-dstnodata", dst_nodata]
    cmd += [f"/vsicurl/{url}", str(out_path)]
    run(cmd, env={"CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff", "GDAL_HTTP_MAX_RETRY": "5", "GDAL_HTTP_RETRY_DELAY": "5"})


def download_core_rasters() -> dict[str, Path]:
    cutline = INPUT / "study_area.shp"
    rasters: dict[str, Path] = {}

    dem_url = "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N39_00_E116_00_DEM/Copernicus_DSM_COG_10_N39_00_E116_00_DEM.tif"
    dem = OUTPUT / "01_DEM" / "copernicus_dem_glo30_30m_32650.tif"
    try:
        gdal_clip_cog(dem_url, dem, cutline, 30, 30, "bilinear", "Float32", None, "-9999")
        slope = OUTPUT / "01_DEM" / "slope_degrees_30m_32650.tif"
        run(["gdaldem", "slope", str(dem), str(slope), "-compute_edges", "-of", "GTiff", "-co", "TILED=YES", "-co", "COMPRESS=DEFLATE"])
        rasters["dem"] = dem
        rasters["slope"] = slope
        add_status("Copernicus DEM GLO-30", "downloaded", dem_url, str(dem.relative_to(ROOT)))
    except Exception as e:
        add_status("Copernicus DEM GLO-30", "failed", dem_url, note=repr(e))

    wc_url = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_N39E114_Map.tif"
    wc = OUTPUT / "02_landcover" / "esa_worldcover_2021_10m_32650.tif"
    try:
        gdal_clip_cog(wc_url, wc, cutline, 10, 10, "near", "Byte", "0", "0")
        built = OUTPUT / "02_landcover" / "built_up_binary_10m.tif"
        crop = OUTPUT / "02_landcover" / "cropland_binary_10m.tif"
        run(["gdal_calc.py", "-A", str(wc), "--outfile", str(built), "--calc", "1*(A==50)", "--type", "Byte", "--NoDataValue", "0", "--co", "COMPRESS=DEFLATE", "--overwrite"])
        run(["gdal_calc.py", "-A", str(wc), "--outfile", str(crop), "--calc", "1*(A==40)", "--type", "Byte", "--NoDataValue", "0", "--co", "COMPRESS=DEFLATE", "--overwrite"])
        rasters["worldcover"] = wc
        rasters["built"] = built
        rasters["cropland"] = crop
        add_status("ESA WorldCover 2021", "downloaded", wc_url, str(wc.relative_to(ROOT)))
    except Exception as e:
        add_status("ESA WorldCover 2021", "failed", wc_url, note=repr(e))

    pop_url = "https://data.worldpop.org/GIS/Population/Global_2015_2030/R2025A/2025/CHN/v1/100m/constrained/chn_pop_2025_CN_100m_R2025A_v1.tif"
    pop = OUTPUT / "03_population" / "worldpop_2025_constrained_100m_32650.tif"
    try:
        gdal_clip_cog(pop_url, pop, cutline, 100, 100, "bilinear", "Float32", None, "-9999")
        rasters["population"] = pop
        add_status("WorldPop 2025 constrained population", "downloaded", pop_url, str(pop.relative_to(ROOT)), "Modelled population count, not census")
    except Exception as e:
        add_status("WorldPop 2025 constrained population", "failed", pop_url, note=repr(e))

    return rasters


def overpass_request(query: str, retries: int = 4) -> dict[str, Any]:
    endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass.nchc.org.tw/api/interpreter",
    ]
    last: Exception | None = None
    for attempt in range(retries):
        endpoint = endpoints[attempt % len(endpoints)]
        try:
            r = requests.post(endpoint, data={"data": query}, timeout=600, headers={"User-Agent": "daxing-pm25-research/1.0"})
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            log(f"Overpass attempt {attempt+1} failed at {endpoint}: {e}")
            time.sleep(15 * (attempt + 1))
    raise RuntimeError(f"Overpass failed: {last}")


def download_osm(study: gpd.GeoDataFrame) -> dict[str, Path]:
    minx, miny, maxx, maxy = study.total_bounds
    bbox = f"{miny},{minx},{maxy},{maxx}"
    out: dict[str, Path] = {}

    road_query = f"[out:json][timeout:600];way[\"highway\"]({bbox});out tags geom;"
    try:
        data = overpass_request(road_query)
        rows = []
        for el in data.get("elements", []):
            coords = [(p["lon"], p["lat"]) for p in el.get("geometry", [])]
            if len(coords) < 2:
                continue
            tags = el.get("tags", {})
            rows.append({
                "osm_id": el.get("id"),
                "highway": tags.get("highway"),
                "name": tags.get("name") or tags.get("name:zh"),
                "lanes": tags.get("lanes"),
                "maxspeed": tags.get("maxspeed"),
                "surface": tags.get("surface"),
                "geometry": LineString(coords),
            })
        roads = gpd.GeoDataFrame(rows, crs=4326).to_crs(TARGET_CRS)
        study_p = study.to_crs(TARGET_CRS)
        roads = gpd.clip(roads, study_p)
        p = OUTPUT / "04_osm" / "osm_roads_study_area.gpkg"
        p.parent.mkdir(parents=True, exist_ok=True)
        roads.to_file(p, layer="roads", driver="GPKG")
        out["roads"] = p
        add_status("OpenStreetMap roads", "downloaded", "Overpass API", str(p.relative_to(ROOT)), f"{len(roads)} road features")
    except Exception as e:
        add_status("OpenStreetMap roads", "failed", "Overpass API", note=repr(e))

    building_query = f"[out:json][timeout:900];way[\"building\"]({bbox});out tags geom;"
    try:
        data = overpass_request(building_query)
        rows = []
        for el in data.get("elements", []):
            coords = [(p["lon"], p["lat"]) for p in el.get("geometry", [])]
            if len(coords) < 4:
                continue
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            poly = Polygon(coords)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty:
                continue
            tags = el.get("tags", {})
            rows.append({
                "osm_id": el.get("id"),
                "building": tags.get("building"),
                "levels": tags.get("building:levels"),
                "height": tags.get("height"),
                "geometry": poly,
            })
        buildings = gpd.GeoDataFrame(rows, crs=4326).to_crs(TARGET_CRS)
        study_p = study.to_crs(TARGET_CRS)
        buildings = gpd.clip(buildings, study_p)
        p = OUTPUT / "04_osm" / "osm_buildings_study_area.gpkg"
        p.parent.mkdir(parents=True, exist_ok=True)
        buildings.to_file(p, layer="buildings", driver="GPKG")
        out["buildings"] = p
        add_status("OpenStreetMap buildings", "downloaded", "Overpass API", str(p.relative_to(ROOT)), f"{len(buildings)} building features; completeness varies")
    except Exception as e:
        add_status("OpenStreetMap buildings", "failed", "Overpass API", note=repr(e))

    return out


def write_raster_clip(src: rasterio.io.DatasetReader, geom4326: dict[str, Any], out_path: Path, scale: float = 1.0, dst_crs: str = TARGET_CRS, nodata_out: float = -9999.0, resampling: Resampling = Resampling.nearest) -> None:
    geom_src = transform_geom("EPSG:4326", src.crs, geom4326, precision=6)
    arr, src_transform = mask(src, [geom_src], crop=True, filled=True)
    arr = arr.astype("float32")
    src_nodata = src.nodata
    invalid = np.zeros(arr.shape, dtype=bool)
    if src_nodata is not None:
        invalid |= arr == src_nodata
    invalid |= ~np.isfinite(arr)
    arr *= scale
    arr[invalid] = nodata_out

    dst_transform, width, height = calculate_default_transform(
        src.crs, dst_crs, arr.shape[2], arr.shape[1], *rasterio.transform.array_bounds(arr.shape[1], arr.shape[2], src_transform), resolution=250
    )
    dest = np.full((arr.shape[0], height, width), nodata_out, dtype="float32")
    for i in range(arr.shape[0]):
        reproject(
            source=arr[i], destination=dest[i], src_transform=src_transform, src_crs=src.crs,
            src_nodata=nodata_out, dst_transform=dst_transform, dst_crs=dst_crs,
            dst_nodata=nodata_out, resampling=resampling,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff", "height": height, "width": width, "count": dest.shape[0],
        "dtype": "float32", "crs": dst_crs, "transform": dst_transform, "nodata": nodata_out,
        "compress": "deflate", "tiled": True,
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(dest)


def download_modis_ndvi(study: gpd.GeoDataFrame) -> list[Path]:
    paths: list[Path] = []
    source = "Microsoft Planetary Computer / NASA MOD13Q1 V6.1"
    try:
        from pystac_client import Client
        import planetary_computer

        catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
        collections = list(catalog.get_collections())
        candidates = [c.id for c in collections if "13Q1" in (c.id + " " + (c.title or "")).upper()]
        if not candidates:
            raise RuntimeError("MOD13Q1 collection not found in Planetary Computer catalog")
        collection_id = candidates[0]
        log(f"Using Planetary Computer collection: {collection_id}")
        bbox = list(study.total_bounds)
        search = catalog.search(collections=[collection_id], bbox=bbox, datetime=f"{START_DATE}/{END_DATE}")
        items = list(search.items())
        if not items:
            raise RuntimeError("No MOD13Q1 items found")
        geom = mapping(study.geometry.unary_union)
        records: list[dict[str, Any]] = []
        for idx, item in enumerate(sorted(items, key=lambda x: x.datetime or datetime.min.replace(tzinfo=timezone.utc))):
            signed = planetary_computer.sign(item)
            keys = list(signed.assets)
            ndvi_key = next((k for k in keys if "NDVI" in k.upper()), None)
            qa_key = next((k for k in keys if "PIXEL_RELIABILITY" in k.upper() or "RELIABILITY" in k.upper()), None)
            if ndvi_key is None:
                log(f"Skip item without NDVI asset: {item.id} keys={keys}")
                continue
            date_str = (item.datetime or datetime.fromisoformat(item.properties["datetime"].replace("Z", "+00:00"))).strftime("%Y%m%d")
            ndvi_path = OUTPUT / "05_ndvi" / "ndvi" / f"MOD13Q1_NDVI_{date_str}.tif"
            with rasterio.open(signed.assets[ndvi_key].href) as src:
                scale = 0.0001
                rb = signed.assets[ndvi_key].extra_fields.get("raster:bands", [])
                if rb and isinstance(rb, list):
                    scale = rb[0].get("scale", scale) or scale
                write_raster_clip(src, geom, ndvi_path, float(scale), resampling=Resampling.nearest)
            paths.append(ndvi_path)
            qa_path = ""
            if qa_key:
                qpath = OUTPUT / "05_ndvi" / "qa" / f"MOD13Q1_pixel_reliability_{date_str}.tif"
                with rasterio.open(signed.assets[qa_key].href) as src:
                    write_raster_clip(src, geom, qpath, 1.0, resampling=Resampling.nearest)
                qa_path = str(qpath.relative_to(ROOT))
            records.append({"date": date_str, "item_id": item.id, "ndvi_file": str(ndvi_path.relative_to(ROOT)), "qa_file": qa_path})
            log(f"NDVI {idx+1}/{len(items)}: {date_str}")
        pd.DataFrame(records).to_csv(OUTPUT / "05_ndvi" / "ndvi_inventory.csv", index=False, encoding="utf-8-sig")
        add_status("MODIS MOD13Q1 NDVI", "downloaded", source, str((OUTPUT / "05_ndvi").relative_to(ROOT)), f"{len(paths)} 16-day clips from {START_DATE} to {END_DATE}")
    except Exception as e:
        add_status("MODIS MOD13Q1 NDVI", "failed", source, note=repr(e))
    return paths


def zonal_values(raster_path: Path, geoms: Iterable[Any], stat: str = "mean") -> list[float]:
    vals: list[float] = []
    with rasterio.open(raster_path) as src:
        for geom in geoms:
            geom_src = transform_geom(TARGET_CRS, src.crs, mapping(geom), precision=6)
            try:
                arr, _ = mask(src, [geom_src], crop=True, filled=False)
                data = arr[0].compressed()
                data = data[np.isfinite(data)]
                if src.nodata is not None:
                    data = data[data != src.nodata]
                if data.size == 0:
                    vals.append(float("nan"))
                elif stat == "sum":
                    vals.append(float(data.sum()))
                elif stat == "median":
                    vals.append(float(np.median(data)))
                else:
                    vals.append(float(data.mean()))
            except ValueError:
                vals.append(float("nan"))
    return vals


def build_features(stations: gpd.GeoDataFrame, rasters: dict[str, Path], vectors: dict[str, Path], ndvi_paths: list[Path]) -> Path:
    st = stations.to_crs(TARGET_CRS).copy()
    st["station_id"] = np.arange(1, len(st) + 1)
    st["lon"] = stations.geometry.x.values
    st["lat"] = stations.geometry.y.values
    result = pd.DataFrame({"station_id": st["station_id"], "lon": st["lon"], "lat": st["lat"]})

    points = st.geometry
    b500 = points.buffer(500)
    b1000 = points.buffer(1000)

    if "dem" in rasters:
        result["elevation_m"] = zonal_values(rasters["dem"], points, "mean")
        result["elevation_mean_1km_m"] = zonal_values(rasters["dem"], b1000, "mean")
    if "slope" in rasters:
        result["slope_mean_1km_deg"] = zonal_values(rasters["slope"], b1000, "mean")
    if "built" in rasters:
        result["built_ratio_500m"] = zonal_values(rasters["built"], b500, "mean")
        result["built_ratio_1km"] = zonal_values(rasters["built"], b1000, "mean")
    if "cropland" in rasters:
        result["cropland_ratio_1km"] = zonal_values(rasters["cropland"], b1000, "mean")
    if "population" in rasters:
        result["population_sum_1km"] = zonal_values(rasters["population"], b1000, "sum")
        result["population_density_1km_per_km2"] = result["population_sum_1km"] / math.pi

    if "roads" in vectors:
        roads = gpd.read_file(vectors["roads"]).to_crs(TARGET_CRS)
        major_types = {"motorway", "trunk", "primary", "motorway_link", "trunk_link", "primary_link"}
        backbone_types = major_types | {"secondary", "secondary_link"}
        for i, (g500, g1000, pt) in enumerate(zip(b500, b1000, points)):
            cand500 = roads[roads.intersects(g500)]
            cand1000 = roads[roads.intersects(g1000)]
            total500 = cand500.geometry.intersection(g500).length.sum()
            major500 = cand500[cand500["highway"].isin(major_types)].geometry.intersection(g500).length.sum()
            back1000 = cand1000[cand1000["highway"].isin(backbone_types)].geometry.intersection(g1000).length.sum()
            back_all = roads[roads["highway"].isin(backbone_types)]
            dist = back_all.distance(pt).min() if not back_all.empty else np.nan
            result.loc[i, "road_density_all_500m_km_per_km2"] = total500 / 1000.0 / (math.pi * 0.5**2)
            result.loc[i, "road_density_major_500m_km_per_km2"] = major500 / 1000.0 / (math.pi * 0.5**2)
            result.loc[i, "road_density_backbone_1km_km_per_km2"] = back1000 / 1000.0 / math.pi
            result.loc[i, "distance_to_backbone_road_m"] = dist

    if "buildings" in vectors:
        buildings = gpd.read_file(vectors["buildings"]).to_crs(TARGET_CRS)
        for i, g500 in enumerate(b500):
            cand = buildings[buildings.intersects(g500)]
            area = cand.geometry.intersection(g500).area.sum()
            result.loc[i, "osm_building_coverage_500m"] = area / g500.area

    if ndvi_paths:
        all_vals = np.array([zonal_values(p, b1000, "mean") for p in ndvi_paths], dtype=float)
        result["ndvi_mean_1km_202405_202606"] = np.nanmean(all_vals, axis=0)
        result["ndvi_latest_1km"] = all_vals[-1]
        result["ndvi_valid_observation_count"] = np.sum(np.isfinite(all_vals), axis=0)

    out = OUTPUT / "06_features" / "station_spatial_features.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False, encoding="utf-8-sig")
    add_status("Station spatial feature table", "generated", "Derived from downloaded datasets", str(out.relative_to(ROOT)), f"{len(result)} stations, {len(result.columns)} columns")
    return out


def write_documentation() -> None:
    docs = OUTPUT / "07_docs"
    docs.mkdir(parents=True, exist_ok=True)
    with open(docs / "DATA_MANIFEST.csv", "w", newline="", encoding="utf-8-sig") as f:
        fields = ["dataset", "status", "source", "output", "note"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for s in STATUSES:
            writer.writerow(asdict(s))
    (docs / "DATA_MANIFEST.json").write_text(json.dumps([asdict(s) for s in STATUSES], ensure_ascii=False, indent=2), encoding="utf-8")
    text = [
        "# 大兴PM2.5预报空间辅助数据包\n",
        f"生成时间：{datetime.now(timezone.utc).isoformat()}\n",
        "## 使用范围\n",
        "22个空气质量站未来72小时PM2.5预报及约1 km连续栅格空间增强。\n",
        "## 已处理数据\n",
    ]
    for s in STATUSES:
        text.append(f"- **{s.dataset}**：{s.status}；来源：{s.source}；输出：{s.output or '-'}；说明：{s.note or '-'}\n")
    text += [
        "\n## 重要限制\n",
        "- WorldPop为模型估算人口，不是人口普查。\n",
        "- WorldCover 2021为静态土地覆盖。\n",
        "- OSM道路和建筑完整性取决于志愿者标注。\n",
        "- MOD13Q1为16天合成产品；进入业务模型时必须使用预测发布时点之前已发布的数据。\n",
        "- 本包暂不把CAMS、EDGAR粗网格排放作为核心空间变量，也不将其称为2024—2026本地实测排放。\n",
        "\n## 模型接入建议\n",
        "优先从 `06_features/station_spatial_features.csv` 选取不超过8个空间变量，并进行留一站空间交叉验证。\n",
    ]
    (docs / "README_DATA.md").write_text("".join(text), encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def package() -> Path:
    checks = []
    for p in sorted(OUTPUT.rglob("*")):
        if p.is_file():
            checks.append(f"{sha256_file(p)}  {p.relative_to(OUTPUT).as_posix()}")
    (OUTPUT / "07_docs" / "SHA256SUMS.txt").write_text("\n".join(checks) + "\n", encoding="utf-8")
    archive_base = RELEASE / "daxing_pm25_auxiliary_data_core"
    if archive_base.with_suffix(".zip").exists():
        archive_base.with_suffix(".zip").unlink()
    shutil.make_archive(str(archive_base), "zip", OUTPUT)
    return archive_base.with_suffix(".zip")


def main() -> None:
    study, stations = ensure_inputs()
    log(f"Study bounds: {study.total_bounds.tolist()}")
    rasters = download_core_rasters()
    vectors = download_osm(study)
    ndvi = download_modis_ndvi(study)
    build_features(stations, rasters, vectors, ndvi)
    add_status("CAMS 2024-2026 emissions", "not_downloaded", "Copernicus ADS", note="Requires CDS/ADS API credentials; later years are forecast-oriented extensions rather than local measured inventory")
    add_status("EDGAR 2022 emissions", "deferred", "JRC EDGAR", note="Coarse 0.1 degree background; excluded from core run to avoid block artefacts and oversized downloads")
    add_status("Local industrial emission sources", "not_downloaded", "National pollution permit platform / local departments", note="Requires manual verification or departmental data")
    write_documentation()
    z = package()
    log(f"PACKAGE: {z} size={z.stat().st_size}")


if __name__ == "__main__":
    main()
