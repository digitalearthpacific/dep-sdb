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
    ##"NU",
    "PG",
    "SB",
    "TO",
    "TV",
    "VU",
    "WS",
]
# countries = ["CK"]


def reproject(gdf, country):
    # print(gdf.columns)
    if country == "FJ":
        gdf = gdf.set_crs(epsg="3460", allow_override=True)
    if country == "KI":
        gdf = gdf.set_crs(epsg="32760", allow_override=True)
    if country == "TV":
        gdf = gdf.set_crs(epsg="32760", allow_override=True)
    reprojected = gdf.to_crs(epsg="4326")
    return reprojected


def fix_columns(gdf):
    gdf.columns = gdf.columns.str.lower()
    if gdf.geometry.name != "geometry":
        gdf.rename(columns={gdf.geometry.name: "geometry"})
    gdf.rename(columns={i: "depth" for i in gdf.columns if i.startswith("dep")})
    if "depth" not in gdf.columns.to_list():
        gdf = gdf.rename(columns={gdf.columns[2]: "depth"})
    return gdf


def remove_non_point_type(gdf_list):
    gdf_list_point = []
    for gdf in gdf_list:
        if gdf.geom_type[0] == "Point":
            gdf_list_point.append(gdf)
    return gdf_list_point


for country in countries:
    print("Processing: " + country)
    cpath = path + country + "/Bathy_shp/"
    files = glob.glob(cpath + "**/*.shp", recursive=True)
    gdf_list = [gpd.read_file(file) for file in files]
    gdf_list = remove_non_point_type(gdf_list)
    gdf_list = [fix_columns(gdf) for gdf in gdf_list]
    gdf_list = [reproject(gdf, country) for gdf in gdf_list]
    gdf = gpd.pd.concat(gdf_list, axis=0)
    gdf = gdf.filter(items=["depth", "geometry"])
    gdf = gdf.dropna()
    gdf.to_file("data/" + country + ".gpkg", driver="GPKG", layer="prni")
