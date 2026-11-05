import contextlib
import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import terrio

from helsinki import CRS, WCS_BOUNDS, WCS_LAYER, WCS_URL, run_wcs, terrain_mesh, write_ply


TRANSFORM = (25496200.0, 1.0, 0.0, 6673700.0, 0.0, -1.0)


class TerrainMeshTests(unittest.TestCase):
    def test_pixel_centers_elevations_and_upward_winding(self):
        heights = np.array([[1.0, 2.0], [3.0, 4.0]])
        vertices, faces = terrain_mesh(heights, TRANSFORM)
        np.testing.assert_array_equal(
            vertices,
            [
                [25496200.5, 6673699.5, 1.0],
                [25496201.5, 6673699.5, 2.0],
                [25496200.5, 6673698.5, 3.0],
                [25496201.5, 6673698.5, 4.0],
            ],
        )
        self.assertEqual(faces.shape, (2, 3))
        triangles = vertices[faces]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        self.assertTrue(np.all(normals[:, 2] > 0))
        self.assertEqual(vertices.dtype, np.dtype("float64"))

    def test_invalid_samples_leave_holes_not_nodata_spikes(self):
        for invalid in (-32767.0, np.nan, np.inf):
            with self.subTest(invalid=invalid):
                heights = np.array([[invalid, 2.0], [3.0, 4.0]])
                vertices, faces = terrain_mesh(heights, TRANSFORM, -32767.0)
                self.assertEqual(vertices.shape, (3, 3))
                self.assertEqual(faces.shape, (1, 3))
                np.testing.assert_array_equal(vertices[:, 2], [2.0, 3.0, 4.0])
                self.assertEqual(set(faces.ravel()), {0, 1, 2})

    def test_grid_counts_and_extent(self):
        vertices, faces = terrain_mesh(np.zeros((3, 4)), TRANSFORM)
        self.assertEqual(len(vertices), 12)
        self.assertEqual(len(faces), 2 * (3 - 1) * (4 - 1))
        np.testing.assert_array_equal(np.ptp(vertices[:, :2], axis=0), [3.0, 2.0])

    def test_rejects_empty_or_non_triangulatable_grids(self):
        for heights in (np.zeros((1, 2)), np.zeros(3), np.full((2, 2), np.nan)):
            with self.subTest(shape=heights.shape), self.assertRaises(ValueError):
                terrain_mesh(heights, TRANSFORM)

    def test_rejects_rotated_or_invalid_transform(self):
        for transform in (
            (0.0, 0.0, 0.0, 0.0, 0.0, -1.0),
            (0.0, 1.0, 0.5, 0.0, 0.0, -1.0),
            (0.0, 1.0, 0.0, np.nan, 0.0, -1.0),
        ):
            with self.subTest(transform=transform), self.assertRaises(ValueError):
                terrain_mesh(np.zeros((2, 2)), transform)

    def test_ply_file_uses_double_precision_vertices_and_zero_based_faces(self):
        vertices, faces = terrain_mesh(np.array([[1.0, 2.0], [3.0, 4.0]]), TRANSFORM)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "terrain.ply"
            write_ply(path, vertices, faces, CRS)
            lines = path.read_text(encoding="ascii").splitlines()
            points = terrio.read_points(str(path))
            np.testing.assert_array_equal(points["x"], vertices[:, 0])
            np.testing.assert_array_equal(points["y"], vertices[:, 1])
            np.testing.assert_array_equal(points["z"], vertices[:, 2])
        self.assertEqual(
            lines[:10],
            [
                "ply",
                "format ascii 1.0",
                "comment crs EPSG:3879",
                "element vertex 4",
                "property double x",
                "property double y",
                "property double z",
                "element face 2",
                "property list uchar int vertex_indices",
                "end_header",
            ],
        )
        self.assertEqual(lines[10], "25496200.500 6673699.500 1.000000")
        self.assertEqual(lines[14:], ["3 0 2 1", "3 1 2 3"])

    def test_ply_serializes_nodata_holes_with_remapped_indices(self):
        heights = np.array([[-32767.0, 2.0], [3.0, 4.0]])
        vertices, faces = terrain_mesh(heights, TRANSFORM, -32767.0)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "terrain.ply"
            write_ply(path, vertices, faces, CRS)
            lines = path.read_text(encoding="ascii").splitlines()
        self.assertIn("element vertex 3", lines)
        self.assertIn("element face 1", lines)
        self.assertEqual(lines[-1], "3 0 1 2")
        self.assertNotIn("-32767", "\n".join(lines))

    def test_ply_rejects_multiline_crs_metadata(self):
        vertices, faces = terrain_mesh(np.ones((2, 2)), TRANSFORM)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "terrain.ply"
            with self.assertRaisesRegex(ValueError, "one line"):
                write_ply(path, vertices, faces, "EPSG:3879\ninvalid")
            self.assertFalse(path.exists())

    def test_wcs_request_and_full_resolution_mesh_pipeline(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp)
            fixture = output / "fixture.tif"
            terrio.Raster.from_array(
                np.zeros((1000, 1000), dtype=np.float32), TRANSFORM, crs=CRS
            ).write(str(fixture))
            with (
                patch("helsinki.service_bytes", return_value=fixture.read_bytes()) as request,
                patch("helsinki.write_ply") as writer,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                run_wcs(output)
            endpoint, params, content_types = request.call_args.args
            self.assertEqual(endpoint, WCS_URL)
            self.assertEqual(params["coverage"], WCS_LAYER)
            self.assertEqual(params["bbox"], ",".join(map(str, WCS_BOUNDS)))
            self.assertEqual((params["resx"], params["resy"]), (1, 1))
            self.assertIn("image/tiff", content_types)
            path, vertices, faces, crs = writer.call_args.args
            self.assertEqual(path.suffix, ".ply")
            self.assertEqual(crs, CRS)
            self.assertEqual(vertices.shape, (1_000_000, 3))
            self.assertEqual(faces.shape, (1_996_002, 3))
            self.assertTrue(path.with_suffix(".prj").exists())

    def test_wcs_rejects_wrong_raster_geometry(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp)
            fixture = output / "fixture.tif"
            terrio.Raster.from_array(
                np.zeros((2, 2), dtype=np.float32), TRANSFORM, crs=CRS
            ).write(str(fixture))
            with patch("helsinki.service_bytes", return_value=fixture.read_bytes()):
                with self.assertRaisesRegex(ValueError, "native 1 m"):
                    run_wcs(output)
            self.assertFalse((output / "wcs_korkeusmalli_2021_1m.ply").exists())


if __name__ == "__main__":
    unittest.main()
