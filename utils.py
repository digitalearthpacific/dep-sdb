import numpy as np
import xarray as xr
from odc.algo import mask_cleanup
from odc.stac import load
from pystac import Item
from sklearn.base import RegressorMixin
from xarray import Dataset, DataArray


def make_indices(geomad: Dataset) -> Dataset:
    scaled = (geomad / 10000).clip(0, 1)

    # Add some indices
    geomad["ndvi"] = (scaled.nir - scaled.red) / (scaled.nir + scaled.red)
    geomad["ndwi"] = (scaled.green - scaled.nir) / (scaled.green + scaled.nir)
    geomad["mndwi"] = (scaled.green - scaled.swir16) / (scaled.green + scaled.swir16)
    geomad["ndti"] = (scaled.red - scaled.green) / (scaled.red + scaled.green)

    # Stumpf, calculate off non-scaled data to remove nan/infinities
    geomad["stumpf"] = np.log(geomad.green - geomad.blue) / np.log(
        geomad.green + geomad.blue
    )
    # Blue over green index
    geomad["bg"] = scaled.blue / scaled.green
    # Blue over red index
    geomad["br"] = scaled.blue / scaled.red

    # # Lyzenga... seems problematic
    # geomad["lyzenga"] = np.log(scaled.green / scaled.blue)

    return geomad


def mask_with_gebco(
    ds: Dataset,
    depth: float | int = 40,
    interpolate: bool = True,
    return_mask: bool = False,
) -> Dataset:
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

    masked = ds.where(gebco_mask)

    if return_mask:
        return masked, gebco_mask
    else:
        return masked


def mask_deeps(
    ds: Dataset,
    ds_to_mask: Dataset | None = None,
    threshold: float | None = None,
    return_mask: bool = False,
) -> Dataset:
    """Masks out deep water pixels based on the Stumpf index. If a threshold is provided, the Stumpf index is used to
    create a mask. If no threshold is provided, the Stumpf index is used to create a mask where it is NaN.

    Args:
        ds (Dataset): Dataset to mask
        ds_to_mask (Dataset | None, optional): Dataset to mask. Defaults to None.
        threshold (float | None, optional): Threshold for the Stumpf index. Defaults to None.
        return_mask (bool, optional): If True, returns the mask as well. Defaults to False.

    Returns:
        Dataset: Masked dataset
    """

    if threshold is not None:
        stumpf = np.log(ds.green - ds.blue) / np.log(ds.green + ds.blue)
        mask = stumpf < threshold
        mask = mask_cleanup(mask, [["erosion", 20], ["dilation", 10]])
    else:
        mask = ds.stumpf.isnull()
        mask = mask_cleanup(mask, [["erosion", 20], ["dilation", 10]])
        mask = ~mask

    to_mask = ds if ds_to_mask is None else ds_to_mask
    masked = to_mask.where(mask)

    if return_mask:
        return masked, mask
    else:
        return masked


def mask_land(
    ds: Dataset, ds_to_mask: Dataset | None = None, return_mask: bool = False
) -> Dataset:
    """Masks out land pixels based on the NDWI and MNDWI indices.

    Args:
        ds (Dataset): Dataset to mask
        ds_to_mask (Dataset | None, optional): Dataset to mask. Defaults to None.
        return_mask (bool, optional): If True, returns the mask as well. Defaults to False.

    Returns:
        Dataset: Masked dataset
    """
    land = (ds.mndwi + ds.ndwi).squeeze() < 0
    mask = mask_cleanup(land, [["dilation", 5], ["erosion", 5]])

    to_mask = ds if ds_to_mask is None else ds_to_mask

    # Inverting the mask here
    masked = to_mask.where(~mask)

    if return_mask:
        return masked, mask
    else:
        return masked


def do_prediction(
    ds: Dataset, model: RegressorMixin, output_name: str | None = None
) -> Dataset | DataArray:
    """Predicts the model on the dataset and adds the prediction as a new variable.

    Args:
        ds (Dataset): Dataset to predict on
        model (RegressorMixin): Model to predict with

    Returns:
        Dataset: Dataset with the prediction as a new variable
    """
    mask = ds.red.isnull()  # Probably should check more bands

    # Convert to a stacked array of observations
    stacked_arrays = ds.to_array().stack(dims=["y", "x"])

    # Replace any infinities with NaN
    stacked_arrays = stacked_arrays.where(stacked_arrays != float("inf"))
    stacked_arrays = stacked_arrays.where(stacked_arrays != float("-inf"))

    # Replace any NaN values with 0
    # TODO: Make sure that each column is labelled with the correct band name
    stacked_arrays = stacked_arrays.squeeze().fillna(0).transpose()

    # Predict the classes
    predicted = model.predict(stacked_arrays)

    # Reshape back to the original 2D array
    array = predicted.reshape(ds.y.size, ds.x.size)

    # Convert to an xarray again, because it's easier to work with
    predicted_da = xr.DataArray(array, coords={"y": ds.y, "x": ds.x}, dims=["y", "x"])

    # Mask the prediction with the original mask
    predicted_da = predicted_da.where(~mask)

    # If we have a name, return dataset, else the dataarray
    if output_name is None:
        return predicted_da
    else:
        return predicted_da.to_dataset(name=output_name)
