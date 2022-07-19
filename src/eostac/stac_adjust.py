import subprocess
import tempfile
from pathlib import Path
import enum

from geosample import QuadTree
import numpy as np
import geopandas as gpd
from shapely.geometry import box, Polygon
from tqdm.auto import tqdm


class UniqueIdentifers(enum.Enum):
    unique_id = 'unique_id'


def create_polygon(x: float, y: float, dist: float) -> Polygon:
    left = x - dist
    right = x + dist
    bottom = y - dist
    top = y + dist

    return box(left, bottom, right, top)


def adjust_data(args):
    df_region = gpd.read_file(args.geometry)

    # Get random x,y coordinates
    x_min, y_min, x_max, y_max = df_region.total_bounds
    rng = np.random.default_rng(100)
    x = rng.uniform(x_min, x_max, 50_000)
    y = rng.uniform(y_min, y_max, 50_000)
    # Convert them to a points GeoSeries
    points = gpd.GeoSeries(gpd.points_from_xy(x, y, crs=df_region.crs))
    # Clips the points to the geometry of interest
    points = points[points.within(df_region.unary_union)]
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
    df = df.assign(**{UniqueIdentifers.unique_id.value: range(len(df.index))})
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
                id_path = str(Path(args.out_path) / f"{getattr(row, UniqueIdentifers.unique_id.value):06d}")
                for start_date, end_date in dates:
                    com = "eostac download " \
                          f"--start-date {start_date} " \
                          f"--end-date {end_date} " \
                          f"--out-path {id_path} " \
                          f"--bounds {points_file} " \
                          f"--bounds-query '{UniqueIdentifers.unique_id.value}=={row.unique_id}' " \
                          f"--resolution {args.resolution} " \
                          "--cloud-cover 5 " \
                          f"--workers {args.workers} " \
                          f"--threads {args.threads} " \
                          "--max-items -1 " \
                          "--chunksize 256 " \
                          "--tqdm-item-position 1 " \
                          "--tqdm-extra-position 2"

                    if args.epsg is not None:
                        com += f" --epsg {args.epsg}"

                    if args.errors_only:
                        com += ' --errors-only'

                    subprocess.call(com, shell=True)
    else:
        for row in tqdm(df.itertuples(), total=len(df.index), position=0, desc='Sample'):
            id_path = str(Path(args.out_path) / f"{row.unique_id:06d}")
            com = "eostac brdf " \
                  f"--start-date {args.start_date} " \
                  f"--end-date {args.end_date} " \
                  f"--project-path {id_path} " \
                  f"--workers {args.workers} " \
                  f"--threads {args.threads} " \
                  "--get-coeffs"

            subprocess.call(com, shell=True)
