"""Fetch four Helsinki WFS map sheets as individual WCS DEM and PLY tiles."""

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlencode

import terrio

from helsinki import CRS, WFS_URL, MeshStats, fetch_wcs_raster, write_raster_mesh

MAP_SHEET_LAYER = "avoindata:Karttalehtijako_05x05_km"
TILE_IDS = ("672496d", "672497b", "673496c", "673497a")
TILE_SIZE = 500
STATION_GK25 = (25496758.2, 6673226.3)


@dataclass(frozen=True)
class MapSheet:
    tunnus: str
    bounds: tuple[int, int, int, int]
    ring: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class TileResult:
    sheet: MapSheet
    stats: MeshStats


def integer_coordinate(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Map-sheet coordinate is not numeric: {value!r}")
    if not math.isfinite(value) or value != int(value):
        raise ValueError(f"Map-sheet coordinate is not an integer metre: {value!r}")
    return int(value)


def parse_map_sheet(feature: object) -> MapSheet:
    if not isinstance(feature, dict) or feature.get("type") != "Feature":
        raise ValueError("WFS map-sheet response contains an invalid feature")
    properties = feature.get("properties")
    geometry = feature.get("geometry")
    if not isinstance(properties, dict) or not isinstance(geometry, dict):
        raise ValueError("WFS map-sheet feature is missing properties or geometry")
    tunnus = properties.get("tunnus")
    if not isinstance(tunnus, str) or tunnus not in TILE_IDS:
        raise ValueError(f"Unexpected map-sheet tunnus: {tunnus!r}")
    coordinates = geometry.get("coordinates")
    if (
        geometry.get("type") != "Polygon"
        or not isinstance(coordinates, list)
        or len(coordinates) != 1
        or not isinstance(coordinates[0], list)
        or len(coordinates[0]) < 5
    ):
        raise ValueError(f"Map sheet {tunnus} must be a single-ring polygon")

    points = []
    for point in coordinates[0]:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f"Map sheet {tunnus} has an invalid coordinate")
        points.append((integer_coordinate(point[0]), integer_coordinate(point[1])))
    if points[0] != points[-1]:
        raise ValueError(f"Map sheet {tunnus} has an unclosed polygon")
    min_x = min(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_x = max(point[0] for point in points)
    max_y = max(point[1] for point in points)
    corners = {(min_x, min_y), (min_x, max_y), (max_x, min_y), (max_x, max_y)}
    area_twice = abs(
        sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(points, points[1:]))
    )
    if (
        max_x - min_x != TILE_SIZE
        or max_y - min_y != TILE_SIZE
        or not corners.issubset(points)
        or any(x not in (min_x, max_x) and y not in (min_y, max_y) for x, y in points)
        or any(x1 != x2 and y1 != y2 for (x1, y1), (x2, y2) in zip(points, points[1:]))
        or area_twice != 2 * TILE_SIZE * TILE_SIZE
    ):
        raise ValueError(f"Map sheet {tunnus} is not a 500 x 500 m rectangle")
    if (
        properties.get("koordinaatti_e") != min_x
        or properties.get("koordinaatti_n") != min_y
        or properties.get("lehtikoko_e") != TILE_SIZE
        or properties.get("lehtikoko_n") != TILE_SIZE
    ):
        raise ValueError(f"Map sheet {tunnus} attributes do not match its polygon")
    return MapSheet(tunnus, (min_x, min_y, max_x, max_y), tuple(points))


def select_map_sheets(features: Sequence[object]) -> list[MapSheet]:
    sheets = [parse_map_sheet(feature) for feature in features]
    if len(sheets) != 4 or {sheet.tunnus for sheet in sheets} != set(TILE_IDS):
        raise ValueError("WFS did not return exactly the four selected map sheets")
    xs = sorted({sheet.bounds[0] for sheet in sheets})
    ys = sorted({sheet.bounds[1] for sheet in sheets})
    origins = {(sheet.bounds[0], sheet.bounds[1]) for sheet in sheets}
    if (
        len(xs) != 2
        or len(ys) != 2
        or xs[1] - xs[0] != TILE_SIZE
        or ys[1] - ys[0] != TILE_SIZE
        or origins != {(x, y) for x in xs for y in ys}
    ):
        raise ValueError("Selected WFS map sheets do not form a contiguous 2 x 2 grid")
    if not (
        xs[0] <= STATION_GK25[0] < xs[1] + TILE_SIZE
        and ys[0] <= STATION_GK25[1] < ys[1] + TILE_SIZE
    ):
        raise ValueError("Selected WFS map sheets do not contain the railway station")
    return sorted(sheets, key=lambda sheet: (sheet.bounds[1], sheet.bounds[0]))


def fetch_map_sheets() -> list[MapSheet]:
    cql = "tunnus IN (" + ",".join(f"'{name}'" for name in TILE_IDS) + ")"
    params = {
        "service": "WFS",
        "version": "1.0.0",
        "request": "GetFeature",
        "typeName": MAP_SHEET_LAYER,
        "outputFormat": "application/json",
        "srsName": CRS,
        "CQL_FILTER": cql,
        "maxFeatures": 20,
    }
    dataset = terrio.open_vector(f"{WFS_URL}?{urlencode(params)}")
    if len(dataset.layers) != 1:
        raise ValueError("WFS map-sheet response must contain one layer")
    layer = dataset.layers[0]
    if layer.crs != CRS:
        raise ValueError(f"WFS map sheets have unexpected CRS: {layer.crs}")
    collection = layer.to_geojson()
    features = collection.get("features")
    if collection.get("type") != "FeatureCollection" or not isinstance(features, list):
        raise ValueError("WFS map-sheet response is not a GeoJSON FeatureCollection")
    return select_map_sheets(features)


def write_map_index(output_dir: Path, results: list[TileResult]) -> Path:
    features = [
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[list(point) for point in result.sheet.ring]],
            },
            "properties": {
                "tunnus": result.sheet.tunnus,
                "vertices": result.stats.vertices,
                "faces": result.stats.faces,
                "zmin": result.stats.minimum,
                "zmax": result.stats.maximum,
            },
        }
        for result in results
    ]
    collection = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::3879"}},
        "features": features,
    }
    destination = output_dir / "map_sheets.gpkg"
    with TemporaryDirectory() as tmp:
        source = Path(tmp) / "map_sheets.geojson"
        source.write_text(json.dumps(collection), encoding="utf-8")
        terrio.convert(str(source), str(destination), nln="map_sheets", overwrite=True)
    layers = terrio.open_vector(str(destination)).layers
    if len(layers) != 1 or layers[0].crs != CRS or len(layers[0]) != 4:
        raise ValueError(f"Map-sheet index has unexpected features or CRS: {destination}")
    saved = layers[0].to_geojson()["features"]
    if {feature["properties"]["tunnus"] for feature in saved} != set(TILE_IDS):
        raise ValueError(f"Map-sheet index has unexpected tunnus values: {destination}")
    return destination


def run_map_sheet_test(output_dir: Path) -> list[TileResult]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sheets = fetch_map_sheets()
    results = []
    for sheet in sheets:
        raster_path = output_dir / f"{sheet.tunnus}.tif"
        mesh_path = output_dir / f"{sheet.tunnus}.ply"
        raster = fetch_wcs_raster(sheet.bounds, raster_path)
        stats = write_raster_mesh(raster, mesh_path)
        results.append(TileResult(sheet, stats))
        print(
            f"{sheet.tunnus}: {raster_path}  bounds={sheet.bounds}  "
            f"{mesh_path}  {stats.vertices:,} vertices  {stats.faces:,} triangles"
        )
    index = write_map_index(output_dir, results)
    print(f"WFS map-sheet index: {index}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output") / "wcs_map_tiles")
    args = parser.parse_args()
    run_map_sheet_test(args.output_dir)


if __name__ == "__main__":
    main()
