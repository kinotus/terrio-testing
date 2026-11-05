import contextlib
import copy
import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch, sentinel
from urllib.parse import parse_qs, urlparse

import numpy as np
import terrio

from helsinki import CRS, WCS_LAYER, WCS_URL, MeshStats, fetch_wcs_raster, write_raster_mesh
from wcs_map_tiles import (
    MAP_SHEET_LAYER,
    TILE_IDS,
    TileResult,
    fetch_map_sheets,
    run_map_sheet_test,
    select_map_sheets,
    write_map_index,
)


def feature(tunnus, x, y):
    return {
        "type": "Feature",
        "properties": {
            "tunnus": tunnus,
            "koordinaatti_e": x,
            "koordinaatti_n": y,
            "lehtikoko_e": 500,
            "lehtikoko_n": 500,
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[x, y], [x + 500, y], [x + 500, y + 500], [x, y + 500], [x, y]]],
        },
    }


def valid_features():
    return [
        feature("673497a", 25497000, 6673000),
        feature("672496d", 25496500, 6672500),
        feature("673496c", 25496500, 6673000),
        feature("672497b", 25497000, 6672500),
    ]


class MapSheetTests(unittest.TestCase):
    def test_selects_exact_contiguous_wfs_polygons(self):
        sheets = select_map_sheets(valid_features())
        self.assertEqual([sheet.tunnus for sheet in sheets], list(TILE_IDS))
        self.assertEqual(
            [sheet.bounds for sheet in sheets],
            [
                (25496500, 6672500, 25497000, 6673000),
                (25497000, 6672500, 25497500, 6673000),
                (25496500, 6673000, 25497000, 6673500),
                (25497000, 6673000, 25497500, 6673500),
            ],
        )

    def test_rejects_missing_duplicate_or_unsafe_tunnus(self):
        for features in (
            valid_features()[:-1],
            valid_features()[:3] + [valid_features()[0]],
        ):
            with self.subTest(features=features), self.assertRaises(ValueError):
                select_map_sheets(features)
        features = valid_features()
        features[0]["properties"]["tunnus"] = "../unsafe"
        with self.assertRaisesRegex(ValueError, "tunnus"):
            select_map_sheets(features)

    def test_rejects_invalid_polygon_or_mismatched_attributes(self):
        for mutate in ("size", "polygon_size", "attributes", "unclosed", "diagonal"):
            features = copy.deepcopy(valid_features())
            item = features[0]
            if mutate == "size":
                item["properties"]["lehtikoko_e"] = 1000
            elif mutate == "polygon_size":
                item["geometry"]["coordinates"][0][1][0] += 1
                item["geometry"]["coordinates"][0][2][0] += 1
            elif mutate == "attributes":
                item["properties"]["koordinaatti_e"] += 1
            elif mutate == "unclosed":
                item["geometry"]["coordinates"][0][-1][0] += 1
            else:
                item["geometry"]["coordinates"][0][1][1] += 1
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                select_map_sheets(features)

    def test_rejects_noncontiguous_sheets(self):
        features = valid_features()
        features[3] = feature("672497b", 25497500, 6672500)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            select_map_sheets(features)

    def test_rejects_grid_that_does_not_contain_station(self):
        features = [
            feature(
                item["properties"]["tunnus"],
                item["properties"]["koordinaatti_e"] + 2000,
                item["properties"]["koordinaatti_n"],
            )
            for item in valid_features()
        ]
        with self.assertRaisesRegex(ValueError, "railway station"):
            select_map_sheets(features)

    def test_fetches_the_four_ids_in_projected_geojson(self):
        class FakeLayer:
            crs = CRS

            def to_geojson(self):
                return {"type": "FeatureCollection", "features": valid_features()}

        class FakeDataset:
            layers = [FakeLayer()]

        with patch("wcs_map_tiles.terrio.open_vector", return_value=FakeDataset()) as opening:
            self.assertEqual(len(fetch_map_sheets()), 4)
        query = parse_qs(urlparse(opening.call_args.args[0]).query)
        self.assertEqual(query["typeName"], [MAP_SHEET_LAYER])
        self.assertEqual(query["srsName"], [CRS])
        self.assertEqual(query["outputFormat"], ["application/json"])
        self.assertTrue(all(f"'{name}'" in query["CQL_FILTER"][0] for name in TILE_IDS))

    def test_fetch_rejects_wrong_wfs_crs(self):
        class FakeLayer:
            crs = "EPSG:4326"

        class FakeDataset:
            layers = [FakeLayer()]

        with patch("wcs_map_tiles.terrio.open_vector", return_value=FakeDataset()):
            with self.assertRaisesRegex(ValueError, "unexpected CRS"):
                fetch_map_sheets()

    def test_tile_fetch_uses_source_bounds_and_preserves_nodata_holes(self):
        bounds = (25496500, 6672500, 25497000, 6673000)
        data = np.zeros((500, 500), dtype=np.float32)
        data[0, 0] = -32767
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = root / "fixture.tif"
            terrio.Raster.from_array(
                data, (bounds[0], 1, 0, bounds[3], 0, -1), crs=CRS, nodata=-32767
            ).write(str(fixture))
            with (
                patch("helsinki.service_bytes", return_value=fixture.read_bytes()) as request,
                patch("helsinki.write_ply") as writer,
            ):
                raster = fetch_wcs_raster(bounds, root / "672496d.tif")
                stats = write_raster_mesh(raster, root / "672496d.ply")
            endpoint, params, content_types = request.call_args.args
            self.assertEqual(endpoint, WCS_URL)
            self.assertEqual(params["coverage"], WCS_LAYER)
            self.assertEqual(params["bbox"], ",".join(map(str, bounds)))
            self.assertEqual((params["resx"], params["resy"]), (1, 1))
            self.assertIn("image/tiff", content_types)
            self.assertEqual(stats.vertices, 249_999)
            self.assertEqual(stats.faces, 498_001)
            path, vertices, faces, crs = writer.call_args.args
            self.assertEqual(path.name, "672496d.ply")
            self.assertEqual(crs, CRS)
            self.assertEqual(vertices.shape, (249_999, 3))
            self.assertEqual(faces.shape, (498_001, 3))
            self.assertTrue(path.with_suffix(".prj").exists())

    def test_tunnus_is_used_unchanged_for_each_tile_filename(self):
        sheets = select_map_sheets(valid_features())
        stats = MeshStats(250_000, 498_002, -2.0, 23.0)
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "tiles"
            with (
                patch("wcs_map_tiles.fetch_map_sheets", return_value=sheets),
                patch("wcs_map_tiles.fetch_wcs_raster", return_value=sentinel.raster) as fetch,
                patch("wcs_map_tiles.write_raster_mesh", return_value=stats) as writer,
                patch("wcs_map_tiles.write_map_index", return_value=output / "map_sheets.gpkg"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                results = run_map_sheet_test(output)
            self.assertEqual([item.sheet.tunnus for item in results], list(TILE_IDS))
            self.assertEqual(
                [call.args[1].name for call in fetch.call_args_list],
                [f"{name}.tif" for name in TILE_IDS],
            )
            self.assertEqual(
                [call.args[1].name for call in writer.call_args_list],
                [f"{name}.ply" for name in TILE_IDS],
            )
            self.assertEqual(
                [call.args[0] for call in fetch.call_args_list],
                [sheet.bounds for sheet in sheets],
            )

    def test_geopackage_preserves_wfs_polygons_names_and_crs(self):
        sheets = select_map_sheets(valid_features())
        stats = MeshStats(250_000, 498_002, -2.0, 23.0)
        with TemporaryDirectory() as tmp:
            path = write_map_index(Path(tmp), [TileResult(sheet, stats) for sheet in sheets])
            layer = terrio.open_vector(str(path)).layers[0]
            self.assertEqual(layer.crs, CRS)
            self.assertEqual(len(layer), 4)
            features = layer.to_geojson()["features"]
            self.assertEqual({item["properties"]["tunnus"] for item in features}, set(TILE_IDS))
            for item in features:
                self.assertEqual(item["properties"]["vertices"], 250_000)
                self.assertEqual(item["properties"]["faces"], 498_002)
                self.assertEqual(item["geometry"]["type"], "Polygon")


if __name__ == "__main__":
    unittest.main()
