"""Small terrio examples using Helsinki's public geospatial services."""

import argparse
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import terrio
from numpy.typing import NDArray

BASE_URL = "https://kartta.hel.fi/ws/geoserver"
WMS_URL = f"{BASE_URL}/avoindata/wms"
WMTS_URL = f"{BASE_URL}/gwc/service/wmts"
WFS_URL = f"{BASE_URL}/avoindata/wfs"
WCS_URL = f"{BASE_URL}/avoindata/wcs"
CRS = "EPSG:3879"
WCS_LAYER = "avoindata:Korkeusmalli_2021_1m"
WCS_BOUNDS = (25496200, 6672700, 25497200, 6673700)

WMS_LAYER = "avoindata:Seutukartta_yhdistetty_maankaytto_tiesto"
WMTS_LAYER = "avoindata:Opaskartta_Helsinki"
WMTS_MATRIX_SET = "ETRS-GK25"
WMTS_MATRIX = "ETRS-GK25:10"
WMTS_ROW = 837
WMTS_COL = 510
WFS_POINT_LAYER = "avoindata:Seutukartta_nimisto_kaupunginosanimet_suomi"
WFS_DISTRICT_LAYER = "avoindata:Kaupunginosajako"
WFS_FEATURE_LIMIT = 2

NAMESPACES = {
    "wmts": "http://www.opengis.net/wmts/1.0",
    "ows": "http://www.opengis.net/ows/1.1",
}
MAX_RESPONSE_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class MeshStats:
    vertices: int
    faces: int
    minimum: float
    maximum: float


def service_bytes(
    endpoint: str, params: dict[str, str | int], content_types: set[str]
) -> bytes:
    url = f"{endpoint}?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": "terrio-helsinki-testing/1.0"})
    with urlopen(request, timeout=60) as response:
        content_type = response.headers.get_content_type()
        data = response.read(MAX_RESPONSE_BYTES + 1)
    if len(data) > MAX_RESPONSE_BYTES:
        raise ValueError(f"Response from {endpoint} exceeds {MAX_RESPONSE_BYTES} bytes")
    if content_type not in content_types:
        detail = data[:400].decode("utf-8", errors="replace")
        raise ValueError(f"Unexpected content type {content_type!r} from {endpoint}: {detail}")
    return data


def required_text(element: ET.Element, path: str) -> str:
    text = element.findtext(path, namespaces=NAMESPACES)
    if text is None or not text.strip():
        raise ValueError(f"WMTS capabilities are missing {path}")
    return text.strip()


def by_identifier(elements: list[ET.Element], identifier: str) -> ET.Element:
    for element in elements:
        if element.findtext("ows:Identifier", namespaces=NAMESPACES) == identifier:
            return element
    raise ValueError(f"WMTS capabilities do not contain {identifier!r}")


def wmts_georeferencing(
    capabilities: bytes, matrix_set: str, matrix_name: str, row: int, col: int
) -> tuple[tuple[float, float, float, float, float, float], tuple[int, int]]:
    """Derive a GDAL geotransform and tile dimensions for Helsinki's EPSG:3879 grid."""
    root = ET.fromstring(capabilities)
    grid = by_identifier(
        root.findall("./wmts:Contents/wmts:TileMatrixSet", NAMESPACES), matrix_set
    )
    grid_crs = required_text(grid, "ows:SupportedCRS")
    if grid_crs != "urn:ogc:def:crs:EPSG::3879":
        raise ValueError(f"This example expects EPSG:3879, not {grid_crs!r}")

    matrix = by_identifier(grid.findall("wmts:TileMatrix", NAMESPACES), matrix_name)
    resolution = float(required_text(matrix, "wmts:ScaleDenominator")) * 0.00028
    # EPSG:3879 advertises its coordinates in northing, easting axis order.
    northing, easting = map(float, required_text(matrix, "wmts:TopLeftCorner").split())
    width = int(required_text(matrix, "wmts:TileWidth"))
    height = int(required_text(matrix, "wmts:TileHeight"))
    matrix_width = int(required_text(matrix, "wmts:MatrixWidth"))
    matrix_height = int(required_text(matrix, "wmts:MatrixHeight"))
    if (
        not all(map(math.isfinite, (resolution, northing, easting)))
        or resolution <= 0
        or width <= 0
        or height <= 0
        or not 0 <= row < matrix_height
        or not 0 <= col < matrix_width
    ):
        raise ValueError("Invalid WMTS tile grid or tile coordinates")
    x_origin = easting + col * width * resolution
    y_origin = northing - row * height * resolution
    return (x_origin, resolution, 0.0, y_origin, 0.0, -resolution), (width, height)


def run_wms(output_dir: Path) -> None:
    min_x, min_y, max_x, max_y = 25496000, 6673000, 25498000, 6675000
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetMap",
        "LAYERS": WMS_LAYER,
        "STYLES": "",
        "CRS": CRS,
        "BBOX": f"{min_y},{min_x},{max_y},{max_x}",
        "WIDTH": 256,
        "HEIGHT": 256,
        "FORMAT": "image/geotiff",
    }
    path = output_dir / "wms_seutukartta.tif"
    path.write_bytes(service_bytes(WMS_URL, params, {"image/geotiff"}))
    raster = terrio.open_raster(str(path))
    if raster.crs != CRS or raster.read(1).shape != (256, 256):
        raise ValueError(f"Unexpected WMS raster metadata in {path}")
    expected_bounds = (min_x, min_y, max_x, max_y)
    if any(
        abs(actual - expected) > 1e-6 for actual, expected in zip(raster.bounds, expected_bounds)
    ):
        raise ValueError(f"Unexpected WMS raster bounds in {path}: {raster.bounds}")
    print(f"WMS: {path}  {raster.width}x{raster.height}  {raster.crs}  bounds={raster.bounds}")


def run_wmts(output_dir: Path) -> None:
    capabilities = service_bytes(
        WMTS_URL,
        {"SERVICE": "WMTS", "VERSION": "1.0.0", "REQUEST": "GetCapabilities"},
        {"text/xml", "application/xml"},
    )
    transform, expected_size = wmts_georeferencing(
        capabilities, WMTS_MATRIX_SET, WMTS_MATRIX, WMTS_ROW, WMTS_COL
    )
    params = {
        "SERVICE": "WMTS",
        "VERSION": "1.0.0",
        "REQUEST": "GetTile",
        "LAYER": WMTS_LAYER,
        "STYLE": "",
        "TILEMATRIXSET": WMTS_MATRIX_SET,
        "TILEMATRIX": WMTS_MATRIX,
        "TILEROW": WMTS_ROW,
        "TILECOL": WMTS_COL,
        "FORMAT": "image/png",
    }
    png = output_dir / "wmts_opaskartta.png"
    png.write_bytes(service_bytes(WMTS_URL, params, {"image/png"}))
    source = terrio.open_raster(str(png))
    if (source.width, source.height) != expected_size:
        raise ValueError(f"WMTS tile size does not match the advertised grid: {png}")

    tiff = output_dir / "wmts_opaskartta.tif"
    raster = terrio.Raster.from_array(
        source.read(),
        transform,
        crs=CRS,
        metadata={
            "SOURCE": "City of Helsinki WMTS",
            "LAYER": WMTS_LAYER,
            "TILE_MATRIX_SET": WMTS_MATRIX_SET,
            "TILE_MATRIX": WMTS_MATRIX,
            "TILE_ROW": str(WMTS_ROW),
            "TILE_COL": str(WMTS_COL),
        },
    )
    raster.write(str(tiff), band_metadata=source.band_metadata, colormap=source.color_table)
    result = terrio.open_raster(str(tiff))
    if result.crs != CRS or result.transform != transform:
        raise ValueError(f"WMTS GeoTIFF georeferencing was not preserved: {tiff}")
    print(f"WMTS: {png} (unreferenced PNG)")
    print(f"WMTS: {tiff}  {result.width}x{result.height}  {result.crs}  bounds={result.bounds}")


def run_wfs(output_dir: Path) -> None:
    source = f"WFS:{WFS_URL}"
    names = terrio.list_layers(source)
    for name in (WFS_POINT_LAYER, WFS_DISTRICT_LAYER):
        if name not in names:
            raise ValueError(f"Helsinki WFS no longer advertises {name}")
    print(f"WFS: {len(names)} advertised layers")

    native = terrio.open_vector(
        source,
        layers=[WFS_POINT_LAYER],
        options={"WFS_PAGING": "NO", "WFS_PAGE_SIZE": "250"},
    ).layers[0]
    if native.crs != CRS or not len(native):
        raise ValueError("The native WFS example returned no features or an unexpected CRS")
    print(f"WFS native: {len(native)} features (up to 250, paging disabled), {native.crs}")
    print("WFS native: check EPSG:3879 axis order before spatial use; see README.")

    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeNames": WFS_DISTRICT_LAYER,
        "outputFormat": "application/json",
        "srsName": CRS,
        "count": WFS_FEATURE_LIMIT,
    }
    geojson_url = f"{WFS_URL}?{urlencode(params)}"
    path = output_dir / "wfs_kaupunginosat.gpkg"
    terrio.convert(geojson_url, str(path), overwrite=True)
    layer = terrio.open_vector(str(path)).layers[0]
    if layer.crs != CRS or not 0 < len(layer) <= WFS_FEATURE_LIMIT:
        raise ValueError(f"Unexpected WFS feature count or CRS in {path}")
    print(f"WFS GeoJSON -> GeoPackage: {path}  {len(layer)} features  {layer.crs}")


def terrain_mesh(
    heights: NDArray[np.floating],
    transform: tuple[float, float, float, float, float, float],
    nodata: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Triangulate pixel centers without bridging invalid DEM samples."""
    if heights.ndim != 2 or min(heights.shape) < 2:
        raise ValueError("A terrain mesh needs a two-dimensional grid of at least 2 x 2")
    x0, dx, rx, y0, ry, dy = transform
    if not all(map(math.isfinite, transform)) or dx <= 0 or dy >= 0 or rx != 0 or ry != 0:
        raise ValueError("A terrain mesh needs a finite, north-up raster geotransform")
    valid = np.isfinite(heights)
    if nodata is not None:
        valid &= heights != nodata

    rows, cols = heights.shape
    indices = np.arange(rows * cols, dtype=np.int64).reshape(rows, cols)
    a = indices[:-1, :-1].ravel()
    b = indices[:-1, 1:].ravel()
    c = indices[1:, :-1].ravel()
    d = indices[1:, 1:].ravel()
    faces = np.concatenate((np.column_stack((a, c, b)), np.column_stack((b, c, d))))
    faces = faces[np.all(valid.ravel()[faces], axis=1)]
    if not len(faces):
        raise ValueError("The DEM contains no valid terrain triangles")
    used = np.unique(faces)
    remap = np.full(rows * cols, -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    row, col = np.divmod(used, cols)
    vertices = np.asarray(
        np.column_stack(
            (x0 + (col + 0.5) * dx, y0 + (row + 0.5) * dy, heights.ravel()[used])
        ),
        dtype=np.float64,
    )
    return vertices, remap[faces]


def write_ply(
    path: Path, vertices: NDArray[np.float64], faces: NDArray[np.int64], crs: str
) -> None:
    """Write projected XYZ vertices and counterclockwise triangles as ASCII PLY."""
    if "\n" in crs or "\r" in crs:
        raise ValueError("PLY CRS metadata must fit on one line")
    with path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write(
            "ply\n"
            "format ascii 1.0\n"
            f"comment crs {crs}\n"
            f"element vertex {len(vertices)}\n"
            "property double x\n"
            "property double y\n"
            "property double z\n"
            f"element face {len(faces)}\n"
            "property list uchar int vertex_indices\n"
            "end_header\n"
        )
        for x, y, z in vertices:
            stream.write(f"{x:.3f} {y:.3f} {z:.6f}\n")
        for a, b, c in faces:
            stream.write(f"3 {a} {b} {c}\n")


def fetch_wcs_raster(
    bounds: tuple[int, int, int, int], path: Path
) -> terrio.Raster:
    min_x, min_y, max_x, max_y = bounds
    width, height = max_x - min_x, max_y - min_y
    if width <= 0 or height <= 0:
        raise ValueError(f"WCS bounds must have positive width and height: {bounds}")
    params = {
        "service": "WCS",
        "version": "1.0.0",
        "request": "GetCoverage",
        "coverage": WCS_LAYER,
        "crs": CRS,
        "bbox": ",".join(map(str, bounds)),
        "resx": 1,
        "resy": 1,
        "format": "GeoTIFF",
    }
    path.write_bytes(service_bytes(WCS_URL, params, {"image/tiff", "image/geotiff"}))
    raster = terrio.open_raster(str(path))
    expected_transform = (min_x, 1.0, 0.0, max_y, 0.0, -1.0)
    if (
        raster.crs != CRS
        or raster.count != 1
        or (raster.width, raster.height) != (width, height)
        or raster.transform != expected_transform
        or raster.bounds != bounds
    ):
        raise ValueError(f"WCS did not return the requested bounds at native 1 m: {path}")
    return raster


def write_raster_mesh(raster: terrio.Raster, mesh_path: Path) -> MeshStats:
    wkt = raster.crs_wkt
    if wkt is None:
        raise ValueError(f"WCS raster is missing its CRS definition: {mesh_path}")
    heights = raster.read(1)
    mask = raster.read_mask()
    if mask is not None:
        heights = np.where(mask != 0, heights, np.nan)
    vertices, faces = terrain_mesh(heights, raster.transform, raster.nodata)
    write_ply(mesh_path, vertices, faces, CRS)
    mesh_path.with_suffix(".prj").write_text(wkt, encoding="ascii")
    return MeshStats(
        vertices=len(vertices),
        faces=len(faces),
        minimum=float(vertices[:, 2].min()),
        maximum=float(vertices[:, 2].max()),
    )


def run_wcs(output_dir: Path) -> None:
    path = output_dir / "wcs_korkeusmalli_2021_1m.tif"
    raster = fetch_wcs_raster(WCS_BOUNDS, path)
    mesh_path = output_dir / "wcs_korkeusmalli_2021_1m.ply"
    stats = write_raster_mesh(raster, mesh_path)
    print(f"WCS: {path}  1000x1000  {raster.crs}  bounds={raster.bounds}")
    print(f"Mesh: {mesh_path}  {stats.vertices:,} vertices  {stats.faces:,} triangles")
    print(f"Mesh: elevation range {stats.minimum:.3f} to {stats.maximum:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "service", nargs="?", choices=("all", "wms", "wmts", "wfs", "wcs"), default="all"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, run in (("wms", run_wms), ("wmts", run_wmts), ("wfs", run_wfs), ("wcs", run_wcs)):
        if args.service in ("all", name):
            run(args.output_dir)


if __name__ == "__main__":
    main()
