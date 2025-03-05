from xarray import Dataset
import numpy as np
from pystac import Item
from odc.stac import load


def make_indices(geomad: Dataset) -> Dataset:
    # Add some indices
    geomad["ndvi"] = (geomad.nir - geomad.red) / (geomad.nir + geomad.red)
    geomad["ndwi"] = (geomad.green - geomad.nir) / (geomad.green + geomad.nir)
    geomad["mndwi"] = (geomad.green - geomad.swir16) / (geomad.green + geomad.swir16)
    geomad["ndti"] = (geomad.red - geomad.green) / (geomad.red + geomad.green)

    # Stumpf variable
    geomad["stumpf"] = np.log(geomad.green - geomad.blue) / np.log(
        geomad.green + geomad.blue
    )
    # Lyzenga variable
    geomad["lyzenga"] = np.log(geomad.green - geomad.blue)

    return geomad


def mask_with_gebco(ds: Dataset, depth: float | int = 40, interpolate=True) -> Dataset:
    # Get GEBCO bathymetry for the aoi
    item = Item.from_file(
        "https://data.source.coop/alexgleith/gebco-2024/GEBCO_2024.stac-item.json"
    )
    gebco = load(
        [item],
        bbox=list(ds.odc.geobox.extent.boundingbox.to_crs("epsg:4326")),
        dtype="float32",
        chunks={},
    )
    
    resampling = "bilinear" if interpolate else "nearest"

    # This should be possible to do in one step above... but it isn't working
    gebco = (
        gebco.odc.reproject(ds.odc.geobox, resampling=resampling).squeeze().elevation
    )

    # Mask geomad by gebco where it's less than XXX
    gebco_mask = gebco > depth

    return ds.where(gebco_mask)
