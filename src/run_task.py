from concurrent.futures import ThreadPoolExecutor
from logging import INFO, Formatter, Logger, StreamHandler, getLogger
from pathlib import Path
from zipfile import ZipFile

import boto3
import joblib
import requests
import typer
import xarray as xr
from dask.distributed import Client
from dep_tools.aws import object_exists
from dep_tools.exceptions import EmptyCollectionError
from dep_tools.grids import PACIFIC_GRID_10
from dep_tools.loaders import OdcLoader
from dep_tools.namers import S3ItemPath
from dep_tools.processors import S2Processor
from dep_tools.searchers import PystacSearcher
from dep_tools.stac_utils import StacCreator
from dep_tools.task import AwsStacTask as Task
from dep_tools.writers import AwsDsCogWriter
from odc.stac import configure_s3_access
from typing_extensions import Annotated
from utils import do_prediction, make_indices, mask_deeps, mask_land, SDBProcessor, S2_BANDS
from xarray import DataArray, Dataset


def get_logger(region_code: str) -> Logger:
    """Set up a simple logger"""
    console = StreamHandler()
    time_format = "%Y-%m-%d %H:%M:%S"
    console.setFormatter(
        Formatter(
            fmt=f"%(asctime)s %(levelname)s ({region_code}):  %(message)s",
            datefmt=time_format,
        )
    )

    log = getLogger("GEOMAD")
    log.addHandler(console)
    log.setLevel(INFO)
    return log


def main(
    model_zip_uri: Annotated[str, typer.Option()],
    tile_id: Annotated[str, typer.Option()],
    version: Annotated[str, typer.Option()],
    output_bucket: str = "dep-public-staging",
    memory_limit: str = "50GB",
    n_workers: int = 2,
    threads_per_worker: int = 32,
    overwrite: Annotated[bool, typer.Option()] = False,
    cloud_cover_lessthan: Annotated[int, typer.Option()] = 100,
    datetime: Annotated[str, typer.Option()] = "2024",
) -> None:
    log = get_logger(tile_id)
    log.info("Starting processing")

    grid = PACIFIC_GRID_10
    catalog = "https://earth-search.aws.element84.com/v1"
    collection = "sentinel-2-l2a"

    # Download the model and unzip it
    model_zip = "models/" + model_zip_uri.split("/")[-1]

    if not Path(model_zip).exists():
        log.info(f"Downloading model from {model_zip_uri}")
        r = requests.get(model_zip_uri)
        with open(model_zip, "wb") as f:
            f.write(r.content)

        log.info("Unzipping model")
        with ZipFile(model_zip, "r") as zip_ref:
            zip_ref.extractall()

    # Open the model
    model = joblib.load(model_zip.replace(".zip", ".joblib"))

    tile_index = tuple(int(i) for i in tile_id.split(","))
    geobox = grid.tile_geobox(tile_index)

    # Make sure we can access S3
    log.info("Configuring S3 access")
    configure_s3_access(cloud_defaults=True)
    client = boto3.client("s3")

    itempath = S3ItemPath(
        bucket=output_bucket,
        sensor="s2",
        dataset_id="sdb",
        version=version,
        time=datetime,
    )
    stac_document = itempath.stac_path(tile_id)

    # If we don't want to overwrite, and the destination file already exists, skip it
    if not overwrite and object_exists(output_bucket, stac_document, client=client):
        log.info(f"Item already exists at {stac_document}")
        # This is an exit with success
        raise typer.Exit()

    searcher = PystacSearcher(
        catalog=catalog,
        collections=[collection],
        datetime=datetime,
        query={"eo:cloud_cover": {"lt": cloud_cover_lessthan}},
    )

    loader = OdcLoader(
        bands=S2_BANDS,
        chunks={"x": 3201, "y": 3201},
        groupby="solar_day",
        fail_on_error=False,
    )

    processor = SDBProcessor(
        model=model,
        preprocessor_args={
            "mask_clouds": True,
        },
    )

    # Custom writer so we write multithreaded
    writer = AwsDsCogWriter(itempath, write_multithreaded=True)

    # STAC making thing
    stac_creator = StacCreator(
        itempath=itempath, remote=True, make_hrefs_https=True, with_raster=True
    )

    try:
        with Client(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            memory_limit=memory_limit,
        ):
            log.info(
                (
                    f"Started dask client with {n_workers} workers "
                    f"and {threads_per_worker} threads with "
                    f"{memory_limit} memory"
                )
            )
            paths = Task(
                itempath=itempath,
                id=tile_index,
                area=geobox,
                searcher=searcher,
                loader=loader,
                processor=processor,
                writer=writer,
                logger=log,
                stac_creator=stac_creator,
            ).run()
    except EmptyCollectionError:
        log.info("No items found for this tile")
        raise typer.Exit()  # Exit with success
    except Exception as e:
        log.exception(f"Failed to process with error: {e}")
        raise typer.Exit(code=1)

    log.info(
        f"Completed processing. Wrote {len(paths)} items to https://{output_bucket}.s3.us-west-2.amazonaws.com/{stac_document}"
    )


if __name__ == "__main__":
    typer.run(main)
