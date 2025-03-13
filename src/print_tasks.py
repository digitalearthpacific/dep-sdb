import typer
import geopandas as gpd
import sys
import json

from dep_tools.aws import object_exists
from boto3 import client
from dep_tools.namers import S3ItemPath

COASTLINE_GRID = "https://dep-public-staging.s3.us-west-2.amazonaws.com/dep_ls_coastlines/raw/buffered_coastline_grid.gpkg"


app = typer.Typer()


@app.command()
def print_tasks(
    overwrite: bool = False,
    limit: int = None,
    output_bucket: str = "dep-public-staging",
    output_prefix: str = None,
    datetime: str = None,
    version: str = None,
):
    # Load the coastline grid
    coastlines = gpd.read_file(COASTLINE_GRID)

    tasks = [tile_id for tile_id in coastlines["tile_id"]]
    if limit is not None:
        tasks = tasks[:limit]

    if not overwrite:
        s3_client = client("s3")
        valid_tasks = []

        for task in tasks:
            itempath = S3ItemPath(
                bucket=output_bucket,
                sensor="s2",
                dataset_id="sdb",
                version=version,
                time=datetime,
            )
            stac_path = itempath.stac_path(task["tile-id"].split(","))

            if output_prefix is not None:
                stac_path = f"{output_prefix}/{stac_path}"

            if not object_exists(output_bucket, stac_path, client=s3_client):
                valid_tasks.append(task)

        tasks = valid_tasks

    json.dump(tasks, sys.stdout)


if __name__ == "__main__":
    app()
