import json
import sys

import requests
import typer
from boto3 import client
from dep_tools.aws import object_exists
from dep_tools.namers import S3ItemPath

TILES_LIST = "https://dep-public-staging.s3.us-west-2.amazonaws.com/dep_ls_coastlines/raw/non_hawaii_tiles.txt"

app = typer.Typer()


@app.command()
def print_tasks(
    overwrite: bool = False,
    limit: int = None,
    output_bucket: str = "dep-public-staging",
    output_prefix: str = None,
    datetime: str = "2025",
    version: str = "0.0.0",
):
    # Load the coastline non-hawaii tiles list
    tile_ids = requests.get(TILES_LIST).text.splitlines()

    if limit is not None:
        tile_ids = tile_ids[:limit]

    if not overwrite:
        s3_client = client("s3")
        valid_tile_ids = []

        for tile_id in tile_ids:
            itempath = S3ItemPath(
                bucket=output_bucket,
                sensor="s2",
                dataset_id="sdb",
                version=version,
                time=datetime,
            )
            stac_path = itempath.stac_path(tile_id)

            if output_prefix is not None:
                stac_path = f"{output_prefix}/{stac_path}"

            if not object_exists(output_bucket, stac_path, client=s3_client):
                valid_tile_ids.append(tile_id)

        tile_ids = valid_tile_ids

    json.dump(tile_ids, sys.stdout)


if __name__ == "__main__":
    app()
