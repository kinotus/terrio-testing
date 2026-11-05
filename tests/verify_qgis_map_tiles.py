"""Verify the tiled outputs with QGIS and create a separate QGIS project."""

import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qgis.core import (
    Qgis,
    QgsApplication,
    QgsCoordinateReferenceSystem,
    QgsFillSymbol,
    QgsMesh,
    QgsMeshDatasetIndex,
    QgsMeshLayer,
    QgsPointXY,
    QgsProject,
    QgsRasterLayer,
    QgsRectangle,
    QgsReferencedRectangle,
    QgsVectorLayer,
)


TILE_IDS = ("672496d", "672497b", "673496c", "673497a")


def matching_layers(project, layer_type, path):
    return [
        layer
        for layer in project.mapLayers().values()
        if isinstance(layer, layer_type)
        and Path(layer.source().split("|", 1)[0]).resolve() == path
    ]


def extent_bounds(extent):
    return (extent.xMinimum(), extent.yMinimum(), extent.xMaximum(), extent.yMaximum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output") / "wcs_map_tiles")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    project_path = output / "helsinki_wcs_map_tiles.qgz"
    index_path = output / "map_sheets.gpkg"
    app = QgsApplication([], False)
    app.initQgis()
    project = QgsProject.instance()
    try:
        crs = QgsCoordinateReferenceSystem("EPSG:3879")
        save_project = not project_path.exists()
        if save_project:
            project.setCrs(crs)
            index = QgsVectorLayer(str(index_path), "WFS map sheets", "ogr")
        else:
            if not project.read(str(project_path)):
                raise OSError(f"QGIS could not read {project_path}")
            indexes = matching_layers(project, QgsVectorLayer, index_path)
            if not indexes:
                raise ValueError(f"Existing QGIS project is missing the WFS index: {project_path}")
            index = indexes[0]
        if not index.isValid() or index.crs().authid() != "EPSG:3879":
            raise ValueError(f"QGIS could not load the EPSG:3879 WFS index: {index_path}")
        features = list(index.getFeatures())
        by_name = {feature["tunnus"]: feature for feature in features}
        if len(features) != 4 or set(by_name) != set(TILE_IDS):
            raise ValueError("QGIS WFS index must contain the four selected map sheets")

        for name in TILE_IDS:
            mesh_path = output / f"{name}.ply"
            dem_path = output / f"{name}.tif"
            if save_project:
                mesh = QgsMeshLayer(str(mesh_path), f"Mesh {name}", "mdal")
                dem = QgsRasterLayer(str(dem_path), f"DEM {name}")
                mesh.setCrs(crs)
            else:
                meshes = matching_layers(project, QgsMeshLayer, mesh_path)
                rasters = matching_layers(project, QgsRasterLayer, dem_path)
                if not meshes or not rasters:
                    raise ValueError(f"Existing QGIS project is missing tile {name}")
                mesh, dem = meshes[0], rasters[0]
            if not mesh.isValid() or not dem.isValid():
                raise ValueError(f"QGIS could not load tile {name}")
            if mesh.crs().authid() != "EPSG:3879" or dem.crs().authid() != "EPSG:3879":
                raise ValueError(f"QGIS mesh and DEM must both use EPSG:3879: {name}")
            feature = by_name[name]
            sheet_extent = feature.geometry().boundingBox()
            if (
                (dem.width(), dem.height()) != (500, 500)
                or extent_bounds(dem.extent()) != extent_bounds(sheet_extent)
                or not sheet_extent.contains(mesh.extent())
            ):
                raise ValueError(f"QGIS tile extent does not match the WFS polygon: {name}")
            provider = mesh.dataProvider()
            vertices, faces = provider.vertexCount(), provider.faceCount()
            if vertices != feature["vertices"] or faces != feature["faces"]:
                raise ValueError(f"Unexpected QGIS mesh vertex or triangle count: {name}")
            if vertices < 3 or faces < 1 or provider.datasetGroupCount() < 1:
                raise ValueError(f"QGIS did not expose a valid elevation mesh: {name}")
            native_mesh = QgsMesh()
            provider.populateMesh(native_mesh)
            if native_mesh.vertexCount() != vertices or native_mesh.faceCount() != faces:
                raise ValueError(f"QGIS could not populate mesh geometry: {name}")
            for vertex_index in (0, vertices // 2, vertices - 1):
                vertex = native_mesh.vertex(vertex_index)
                elevation = provider.datasetValue(QgsMeshDatasetIndex(0, 0), vertex_index).scalar()
                source, valid = dem.dataProvider().sample(QgsPointXY(vertex.x(), vertex.y()), 1)
                if (
                    not valid
                    or not math.isfinite(elevation)
                    or not math.isfinite(source)
                    or abs(elevation - source) > 1e-6
                ):
                    raise ValueError(
                        f"QGIS elevation differs from the DEM at {name} vertex {vertex_index}: "
                        f"{elevation}, {source}"
                    )
            if save_project:
                settings = mesh.rendererSettings()
                settings.setActiveScalarDatasetGroup(0)
                mesh.setRendererSettings(settings)
                project.addMapLayer(dem)
                project.addMapLayer(mesh)
            print(f"{name}: QGIS loaded {vertices:,} vertices and {faces:,} triangles")

        if save_project:
            symbol = QgsFillSymbol.createSimple(
                {"style": "no", "outline_color": "255,90,20,255", "outline_width": "0.4"}
            )
            index.renderer().setSymbol(symbol)
            project.addMapLayer(index)
            extent = QgsRectangle(index.extent())
            extent.scale(1.05)
            project.viewSettings().setDefaultViewExtent(QgsReferencedRectangle(extent, crs))
            project.setFilePathStorage(Qgis.FilePathType.Relative)
            if project_path.exists():
                raise FileExistsError(f"QGIS project already exists: {project_path}")
            if not project.write(str(project_path)):
                raise OSError(f"QGIS could not save {project_path}")
            print(f"Saved QGIS project: {project_path}")
        else:
            print(f"Verified existing QGIS project without overwriting it: {project_path}")
        print(f"QGIS {Qgis.QGIS_VERSION}: sampled mesh elevations match each tile's DEM")
    finally:
        project.clear()
        app.exitQgis()


if __name__ == "__main__":
    main()
