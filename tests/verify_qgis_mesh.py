"""Verify the live WCS outputs and save a project using QGIS's own Python runtime."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qgis.core import (
    Qgis,
    QgsApplication,
    QgsCoordinateReferenceSystem,
    QgsMeshDatasetIndex,
    QgsMeshLayer,
    QgsPointXY,
    QgsProject,
    QgsRasterLayer,
    QgsReferencedRectangle,
)


def matching_layers(project, layer_type, path):
    return [
        layer
        for layer in project.mapLayers().values()
        if isinstance(layer, layer_type) and Path(layer.source()).resolve() == path
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    args = parser.parse_args()
    output = args.output_dir.resolve()
    app = QgsApplication([], False)
    app.initQgis()
    project = QgsProject.instance()
    try:
        stem = output / "wcs_korkeusmalli_2021_1m"
        mesh_path = stem.with_suffix(".ply")
        dem_path = stem.with_suffix(".tif")
        project_path = output / "helsinki_wcs_mesh_ply.qgz"
        legacy_path = output / "helsinki_wcs_mesh.qgz"
        crs = QgsCoordinateReferenceSystem("EPSG:3879")
        if project_path.exists():
            if not project.read(str(project_path)):
                raise OSError(f"QGIS could not read {project_path}")
            meshes = matching_layers(project, QgsMeshLayer, mesh_path)
            rasters = matching_layers(project, QgsRasterLayer, dem_path)
            if not meshes or not rasters:
                raise ValueError(
                    f"Existing QGIS project does not contain the PLY and DEM: {project_path}"
                )
            mesh, dem = meshes[0], rasters[0]
            save_project = False
        elif legacy_path.exists():
            if not project.read(str(legacy_path)):
                raise OSError(f"QGIS could not read {legacy_path}")
            meshes = matching_layers(project, QgsMeshLayer, stem.with_suffix(".2dm"))
            rasters = matching_layers(project, QgsRasterLayer, dem_path)
            if not meshes or not rasters:
                raise ValueError(
                    "Existing QGIS project does not contain the original mesh "
                    f"and DEM: {legacy_path}"
                )
            for layer in meshes:
                settings = layer.rendererSettings()
                layer.setDataSource(str(mesh_path), layer.name(), "mdal")
                layer.setCrs(crs)
                layer.setRendererSettings(settings)
            mesh, dem = meshes[0], rasters[0]
            save_project = True
        else:
            mesh = QgsMeshLayer(str(mesh_path), "Downtown Helsinki terrain", "mdal")
            dem = QgsRasterLayer(str(dem_path), "WCS DEM 2021, 1 m")
            if not mesh.isValid() or not dem.isValid():
                raise ValueError("QGIS could not load the generated PLY mesh and DEM")
            mesh.setCrs(crs)
            save_project = True
            project.setCrs(crs)
            project.addMapLayer(dem)
            project.addMapLayer(mesh)
            project.viewSettings().setDefaultViewExtent(QgsReferencedRectangle(mesh.extent(), crs))
            project.setFilePathStorage(Qgis.FilePathType.Relative)
            settings = mesh.rendererSettings()
            settings.setActiveScalarDatasetGroup(0)
            mesh.setRendererSettings(settings)

        if not mesh.isValid() or not dem.isValid():
            raise ValueError("QGIS could not load the generated PLY mesh and DEM")
        if mesh.crs().authid() != "EPSG:3879" or dem.crs().authid() != "EPSG:3879":
            raise ValueError("QGIS mesh and DEM must both use EPSG:3879")
        provider = mesh.dataProvider()
        if provider.vertexCount() != 1_000_000 or provider.faceCount() != 1_996_002:
            raise ValueError("Unexpected QGIS mesh vertex or triangle count")
        extent = mesh.extent()
        bounds = (extent.xMinimum(), extent.yMinimum(), extent.xMaximum(), extent.yMaximum())
        if bounds != (25496200.5, 6672700.5, 25497199.5, 6673699.5):
            raise ValueError(f"Unexpected QGIS mesh extent: {bounds}")
        if provider.datasetGroupCount() < 1:
            raise ValueError("QGIS did not expose mesh vertex elevations")
        samples = (
            (0, 25496200.5, 6673699.5),
            (500500, 25496700.5, 6673199.5),
            (999999, 25497199.5, 6672700.5),
        )
        for index, x, y in samples:
            value = provider.datasetValue(QgsMeshDatasetIndex(0, 0), index).scalar()
            source, valid = dem.dataProvider().sample(QgsPointXY(x, y), 1)
            if not valid or abs(value - source) > 1e-6:
                raise ValueError(
                    f"QGIS mesh elevation differs from DEM at vertex {index}: {value}, {source}"
                )
        if save_project:
            if project_path.exists():
                raise FileExistsError(f"QGIS project already exists: {project_path}")
            if not project.write(str(project_path)):
                raise OSError(f"QGIS could not save {project_path}")
            print(f"Saved QGIS project: {project_path}")
        else:
            print(f"Verified existing QGIS project without overwriting it: {project_path}")
        print(f"QGIS {Qgis.QGIS_VERSION}: valid MDAL mesh, 1,000,000 vertices, 1,996,002 faces")
        print("Mesh elevations match the DEM at the first, middle and last vertices")
    finally:
        project.clear()
        app.exitQgis()


if __name__ == "__main__":
    main()
