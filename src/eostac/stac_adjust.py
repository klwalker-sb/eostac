import subprocess
import tempfile
from pathlib import Path

from geosample import QuadTree

import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
from tqdm.auto import tqdm


def create_polygon(x: float, y: float, dist: float) -> Polygon:
    left = x - dist
    right = x + dist
    bottom = y - dist
    top = y + dist

    return Polygon(
        [
            (left, bottom),
            (left, top),
            (right, top),
            (right, bottom),
            (left, bottom)
        ]
    )


def adjust_data(args):
    df = gpd.read_file(args.geometry)

    # Get random x,y coordinates
    x_min, y_min, x_max, y_max = df.total_bounds
    rng = np.random.default_rng(100)
    x = rng.uniform(x_min, x_max, 50_000)
    y = rng.uniform(y_min, y_max, 50_000)
    # Convert them to a points GeoSeries
    points = gpd.GeoSeries(gpd.points_from_xy(x, y, crs=df.crs))
    # Clips the points to the geometry of interest
    points = points[points.within(df.unary_union)]
    # Get a spatially balanced sample
    qt = QuadTree(points)
    qt.split_recursive(max_samples=10)
    df = gpd.GeoDataFrame(
        qt.sample(
            n=args.n_samples, random_state=100
        ),
        columns=['geometry'],
        crs=points.crs
    )
    df = df.to_crs('epsg:4326')
    df = df.assign(unique_id=range(len(df.index)))
    # Create polygon bounding boxes around the points
    df = df.assign(geometry=[create_polygon(geom.x, geom.y, args.dist) for geom in df.geometry])
    if args.freq is None:
        dates = [(args.start_date, args.end_date)]
    else:
        dates = [
            (
                f"{int(args.start_date.split('-')[0])+i}-{args.start_date.split('-')[1]}-{args.start_date.split('-')[2]}",
                f"{int(args.end_date.split('-')[0])+i}-{args.end_date.split('-')[1]}-{args.end_date.split('-')[2]}"
            )
            for i in range(0, args.freq_repeat)
        ]

    if not args.regress:
        with tempfile.TemporaryDirectory() as tmp:
            points_file = Path(tmp) / 'points.gpkg'
            df.to_file(points_file)
            for row in tqdm(df.itertuples(), total=len(df.index), position=0, desc='Sample'):
                id_path = str(Path(args.out_path) / f"{row.unique_id:06d}")
                for start_date, end_date in dates:
                    com = "python stac_downloader.py " \
                          f"--start-date {start_date} " \
                          f"--end-date {end_date} " \
                          f"--out-path {id_path} " \
                          f"--bounds {points_file} " \
                          f"--bounds-query 'unique_id=={row.unique_id}' " \
                          "--cloud-cover 5 " \
                          "--workers 1 " \
                          "--threads 8 " \
                          "--max-items -1 " \
                          "--tqdm-item-position 1 " \
                          "--tqdm-extra-position 2"

                    subprocess.call(com, shell=True)
    else:
        for row in tqdm(df.itertuples(), total=len(df.index), position=0, desc='Sample'):
            id_path = str(Path(args.out_path) / f"{row.unique_id:06d}")
            com = "python stac_post.py " \
                  f"--project-path {id_path} " \
                  "--get-coeffs"

            subprocess.call(com, shell=True)
