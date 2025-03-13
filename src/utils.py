from concurrent.futures import ThreadPoolExecutor
import numpy as np
import xarray as xr
from dep_tools.processors import Processor
from odc.algo import mask_cleanup
from odc.stac import load
from pystac import Item
from sklearn.base import RegressorMixin
from xarray import DataArray, Dataset
from dep_tools.s2_utils import mask_clouds
from odc.algo import mask_cleanup


S2_BANDS = [
    "nir",
    "red",
    "blue",
    "green",
    "nir08",
    "nir09",
    "swir16",
    "swir22",
    "coastal",
    "scl",
]


class Location:
    def __init__(self, bbox, name):
        self.bbox = bbox
        self.name = name

    def __str__(self):
        return f"{self.bbox}"


locations_list = [
    Location([177.20, -17.85, 177.50, -17.65], "nadi"),
    Location([179.020, -8.665, 179.218, -8.413], "tuvalu"),
    Location([178.400, -18.200, 178.600, -18.000], "suva"),
    Location([177.05276, -17.80173, 177.27512, -17.64840], "malolo"),
    Location([-171.7, -13.9, -171.9, -13.7], "apia"),
    Location([-159.85, -21.3, -159.7, -21.15], "rarotonga"),
]


class Locations:
    def __init__(self):
        for location in locations_list:
            setattr(self, location.name, location)

    # Print locations
    def __str__(self):
        return "\n".join([str(location) for location in locations_list])


locations = Locations()


class SDBProcessor(Processor):
    send_area_to_processor = False

    def __init__(self, model, parallelism):
        self.model = model
        self.parallelism = parallelism

    def process(self, input: DataArray) -> Dataset:
        # Mask clouds from S-2
        data = mask_clouds(input)

        # Drop the SCL band, because the pre-processor should have masked clouds
        data = data.drop_vars(["scl"])

        # Add the fancy indices
        data = make_indices(data)

        # Mask land
        data = mask_land(data)

        # # Mask deep water
        data = mask_deeps(data)

        predictions_list = []

        def process_day(day):
            # Load day into memory
            day_data = data.sel(time=day).compute()
            # Do prediction on in-memory data
            return do_prediction(day_data, self.model)

        with ThreadPoolExecutor(max_workers=self.parallelism) as executor:
            predictions_list = list(executor.map(process_day, data.time))

        # Concatenate them all together again
        predictions = xr.concat(predictions_list, dim="time").to_dataset(
            name="elevation"
        )

        # Clean up the data by removing pixels that only had predictions sometimes
        output = predictions.elevation.count(dim="time").to_dataset(name="count")
        total = len(predictions.time)

        # At least X% of the time there was a prediction
        mask = output["count"] > (total * 0.20)
        # Clean up to try to remove single pixel noise
        mask = mask_cleanup(mask, [["dilation", 2], ["erosion", 2]])

        output["mean"] = predictions.elevation.mean(dim="time")
        output["stdev"] = predictions.elevation.std(dim="time")
        output["depth"] = output["mean"].where(mask)

        output["count"] = output["count"].astype("uint8")
        output["mean"] = output["mean"].astype("float32")
        output["stdev"] = output["stdev"].astype("float32")
        output["depth"] = output["depth"].astype("float32")

        # Set count to 255 if it's 0
        output["count"].attrs = {"nodata": 255}
        output["count"] = output["count"].where(output["count"] > 0, 255)

        return output


def make_indices(geomad: Dataset) -> Dataset:
    scaled = (geomad / 10000).clip(0, 1)

    # Add some indices
    geomad["ndvi"] = (scaled.nir - scaled.red) / (scaled.nir + scaled.red)
    geomad["ndwi"] = (scaled.green - scaled.nir) / (scaled.green + scaled.nir)
    geomad["mndwi"] = (scaled.green - scaled.swir16) / (scaled.green + scaled.swir16)
    geomad["ndti"] = (scaled.red - scaled.green) / (scaled.red + scaled.green)

    # Stumpf, calculate off non-scaled data to remove nan/infinities
    geomad["stumpf"] = np.log(np.abs(scaled.green - scaled.blue)) / np.log(
        scaled.green + scaled.blue
    )
    # Blue over green index
    geomad["bg"] = scaled.blue / scaled.green
    # Blue over red index
    geomad["br"] = scaled.blue / scaled.red

    # Natural log of blue/green
    geomad["ln_bg"] = np.log(scaled.blue / scaled.green)

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


def apply_mask(
    ds: Dataset,
    mask: DataArray,
    ds_to_mask: Dataset | None = None,
    return_mask: bool = False,
) -> Dataset:
    """Applies a mask to a dataset"""
    to_mask = ds if ds_to_mask is None else ds_to_mask
    masked = to_mask.where(mask)

    if return_mask:
        return masked, mask
    else:
        return masked


def mask_deeps_stumpf(
    ds: Dataset,
    ds_to_mask: Dataset | None = None,
    threshold: float = 2.0,
    return_mask: bool = False,
) -> Dataset:
    """Masks out deep water pixels based on the Stumpf index.

    Args:
        ds (Dataset): Dataset to mask
        ds_to_mask (Dataset | None, optional): Dataset to mask. Defaults to None.
        threshold (float | None, optional): Threshold for the Stumpf index. Defaults to 1.9.
        return_mask (bool, optional): If True, returns the mask as well. Defaults to False.

    Returns:
        Dataset: Masked dataset
    """
    mask = ds.stumpf > threshold
    mask = mask_cleanup(mask, [["erosion", 5], ["dilation", 5]])

    return apply_mask(ds, mask, ds_to_mask, return_mask)


def mask_deeps_ln_bg(
    ds: Dataset,
    ds_to_mask: Dataset | None = None,
    threshold: float = 0.2,
    return_mask: bool = False,
) -> Dataset:
    """Masks out deep water pixels based on the natural log of the blue/green

    Args:
        ds (Dataset): Dataset to mask
        ds_to_mask (Dataset | None, optional): Dataset to mask. Defaults to None.
        threshold (float, optional): Threshold for the natural log of the blue/green. Defaults to 0.2.
        return_mask (bool, optional): If True, returns the mask as well. Defaults to False.

    Returns:
        Dataset: Masked dataset
    """
    mask = ds.ln_bg < threshold
    mask = mask_cleanup(mask, [["erosion", 5], ["dilation", 5]])

    return apply_mask(ds, mask, ds_to_mask, return_mask)


def mask_deeps(
    ds: Dataset,
    ds_to_mask: Dataset | None = None,
    return_mask: bool = False,
    stumpf_threshold: float = 2.0,
    ln_bg_threshold: float = 0.2,
) -> Dataset:
    _, mask_stumpf = mask_deeps_stumpf(ds, threshold=stumpf_threshold, return_mask=True)
    _, mask_ln_bg = mask_deeps_ln_bg(ds, threshold=ln_bg_threshold, return_mask=True)

    mask = mask_stumpf | mask_ln_bg

    return apply_mask(ds, mask, ds_to_mask, return_mask)


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

    # Inverting the mask here
    mask = ~mask

    return apply_mask(ds, mask, ds_to_mask, return_mask)


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
    stacked_arrays = stacked_arrays.squeeze().fillna(0).transpose().to_pandas()

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
