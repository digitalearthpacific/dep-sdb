import geopandas as gpd
import glob

path = "/Volumes/SACHIN4TB/PRNI_DATA/data/"
countries = [
    "CK",
    "FJ",
    "FM",
    "FP",
    "KI",
    "MH",
    "NC",
    "NR",
    "NU",
    "PG",
    "SB",
    "TO",
    "TV",
    "VU",
    "WS",
]
countries = ["FJ"]


def reproject(gdf):
    reprojected = gdf.to_crs(epsg="4326")
    return reprojected

def fix_columns(gdf):
    gdf.columns = gdf.columns.str.lower()
    gdf.rename(columns={i: "depth" for i in gdf.columns if i.startswith("dept")})
    if "depth" not in gdf.columns.to_list():
        print(gdf.geometry)
        gdf = gdf.rename(columns={gdf.columns[2]: "depth"})
    return gdf


for country in countries:
    print("Processing: " + country)
    cpath = path + country + "/Bathy_shp/"
    files = glob.glob(cpath + "**/*.shp", recursive=True)
    # print(files)
    gdf_list = [gpd.read_file(file) for file in files]
    gdf_list = [fix_columns(gdf) for gdf in gdf_list]
    gdf_list = [reproject(gdf) for gdf in gdf_list]
    gdf = gpd.pd.concat(gdf_list, axis=0)
    # print(gdf.columns)
    gdf = gdf.filter(items=["depth", "geometry"])
    gdf = gdf.dropna()
    gdf.to_file("data/" + country + ".gpkg", driver="GPKG", layer="prni")
