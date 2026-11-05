import contextlib
import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import terrio

from helsinki import WMTS_MATRIX, WMTS_MATRIX_SET, run_wmts, wmts_georeferencing


CAPABILITIES = b"""\
<Capabilities xmlns="http://www.opengis.net/wmts/1.0"
              xmlns:ows="http://www.opengis.net/ows/1.1">
  <Contents>
    <TileMatrixSet>
      <ows:Identifier>ETRS-GK25</ows:Identifier>
      <ows:SupportedCRS>urn:ogc:def:crs:EPSG::3879</ows:SupportedCRS>
      <TileMatrix>
        <ows:Identifier>ETRS-GK25:10</ows:Identifier>
        <ScaleDenominator>28571.428571428572</ScaleDenominator>
        <TopLeftCorner>8388608.0 24451424.0</TopLeftCorner>
        <TileWidth>256</TileWidth>
        <TileHeight>256</TileHeight>
        <MatrixWidth>1024</MatrixWidth>
        <MatrixHeight>1024</MatrixHeight>
      </TileMatrix>
    </TileMatrixSet>
  </Contents>
</Capabilities>
"""


class WmtsGeoreferencingTests(unittest.TestCase):
    def test_helsinki_tile_transform_and_dimensions(self):
        transform, size = wmts_georeferencing(CAPABILITIES, WMTS_MATRIX_SET, WMTS_MATRIX, 837, 510)
        self.assertEqual(transform, (25495904.0, 8.0, 0.0, 6674432.0, 0.0, -8.0))
        self.assertEqual(size, (256, 256))

    def test_rejects_another_crs(self):
        capabilities = CAPABILITIES.replace(b"EPSG::3879", b"EPSG::3857")
        with self.assertRaisesRegex(ValueError, "expects EPSG:3879"):
            wmts_georeferencing(capabilities, WMTS_MATRIX_SET, WMTS_MATRIX, 837, 510)

    def test_rejects_missing_matrix(self):
        with self.assertRaisesRegex(ValueError, "do not contain"):
            wmts_georeferencing(CAPABILITIES, WMTS_MATRIX_SET, "ETRS-GK25:99", 837, 510)

    def test_rejects_out_of_range_tile(self):
        with self.assertRaisesRegex(ValueError, "Invalid WMTS"):
            wmts_georeferencing(CAPABILITIES, WMTS_MATRIX_SET, WMTS_MATRIX, 1024, 510)

    def test_rejects_zero_scale(self):
        capabilities = CAPABILITIES.replace(b"28571.428571428572", b"0")
        with self.assertRaisesRegex(ValueError, "Invalid WMTS"):
            wmts_georeferencing(capabilities, WMTS_MATRIX_SET, WMTS_MATRIX, 837, 510)

    def test_wmts_geotiff_preserves_rgba_pixels_and_georeferencing(self):
        pixels = np.arange(4 * 256 * 256, dtype=np.uint8).reshape(4, 256, 256)
        colors = ["Red", "Green", "Blue", "Alpha"]
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = root / "fixture.png"
            terrio.Raster.from_array(pixels, (0.0, 1.0, 0.0, 256.0, 0.0, -1.0)).write(
                str(fixture), band_metadata=[{"color_interp": color} for color in colors]
            )
            output = root / "output"
            output.mkdir()
            with patch("helsinki.service_bytes", side_effect=[CAPABILITIES, fixture.read_bytes()]):
                with contextlib.redirect_stdout(io.StringIO()):
                    run_wmts(output)
            result = terrio.open_raster(str(output / "wmts_opaskartta.tif"))
            self.assertEqual(result.crs, "EPSG:3879")
            self.assertEqual(result.transform, (25495904.0, 8.0, 0.0, 6674432.0, 0.0, -8.0))
            self.assertEqual([band["color_interp"] for band in result.band_metadata], colors)
            self.assertTrue(np.array_equal(result.read(), pixels))


if __name__ == "__main__":
    unittest.main()
