from pathlib import Path
import typing as T

from .stac_utils import read_df, open_netcdf, smooth

import pandas as pd
import rasterio as rio
from rasterio.drivers import driver_from_extension
from tqdm.auto import tqdm
import psutil


def get_file_list(brdf_path: Path, start_date: str, end_date: str):
    df = read_df(brdf_path / 'scene.info')
    df['file_path'] = brdf_path / df.out_id.map(str)
    df['file_path_exists'] = df.file_path.apply(lambda x: x.is_file())
    df = df.loc[df.file_path_exists]
    df = df.loc[start_date:end_date]
    file_list = df.file_path.map(str).values.tolist()
    time_names = df.date.dt.strftime('%Y-%m-%d').values.tolist()

    return file_list, time_names


def smooth_series(
    brdf_grid_path: Path,
    filename: T.Union[str, Path],
    vi: str = 'evi2',
    bands: T.Sequence[str] = None,
    read_chunks: int = 512,
    proc_chunks: int = 256,
    n_workers: int = psutil.cpu_count(),
    memory_limit: str = 'auto',
    periods: T.Sequence[tuple] = None,
    freq: str = 'W',
    target_day: str = '7-1'
):
    """Smooths a time series of BRDF images

    Args:
        brdf_grid_path: The path to time series images for a single grid.
        filename: The output file.
        vi: The vegetation index to apply.
        bands: The bands needed to compute ``vi``.
        read_chunks: The input chunk size.
        proc_chunks: The processing block size.
        n_workers: The number of parallel workers.
        memory_limit: The memory limit (in GB) per worker.
        periods: A sequence of start/end periods to process.
            E.g.:
                periods = [
                    ('2020-4-1', '2021-11-1')
                ]
        freq: The time frequency to interpolate and smooth to.
            Reference:
                https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases
        target_day: The target month and day of the time series.
    """
    if not bands:
        bands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
    filename = Path(filename)
    for start_end in periods:
        start_date, end_date = start_end
        target_range = pd.date_range(start_date, end_date, freq=freq)
        target_slice = slice(
            f"{start_date.split('-')[0]}-{target_day}",
            f"{end_date.split('-')[0]}-{target_day}"
        )
        file_list, time_names = get_file_list(
            brdf_path=Path(brdf_grid_path),
            start_date=start_date,
            end_date=end_date
        )
        with open_netcdf(
            file_list,
            chunks=read_chunks,
            n_workers=n_workers,
            threads_per_worker=1,
            memory_limit=memory_limit
        ) as src:
            profile = dict(
                driver=driver_from_extension(filename),
                width=src.darray.gw.ncols,
                height=src.darray.gw.nrows,
                nodata=0,
                blockxsize=src.darray.gw.col_chunks,
                blockysize=src.darray.gw.row_chunks,
                crs=src.darray.gw.crs_to_pyproj,
                transform=src.darray.gw.transform,
                compress='lzw',
                tiled=True
            )
            data_windows = list(src.darray.gw.windows(
                row_chunks=proc_chunks, col_chunks=proc_chunks
            ))
            for w in tqdm(data_windows):
                i = w.row_off
                j = w.col_off
                # Load the data for the chunk
                darray = src.load(
                    bands=bands,
                    coord_slice=(
                        slice(i, i+w.height),
                        slice(j, j+w.width)
                    ),
                    compute=False
                )
                # Smooth the data
                ysmooth = smooth(
                    darray,
                    vi=vi,
                    date_range=target_range,
                    s=0.01,
                    k=3
                )
                # Write to file
                ysmooth = ysmooth.sel(time=target_slice).values[::10]
                profile['count'] = ysmooth.shape[0]
                profile['dtype'] = ysmooth.dtype.name
                mode = 'w' if not filename.is_file() else 'r+'
                with rio.open(filename, mode=mode, **profile) as dst:
                    dst.write(
                        ysmooth,
                        indexes=list(range(1, profile['count']+1)),
                        window=w
                    )


def smooth_data(args):
    print(
        "View processing diagnostics at http://localhost:8787/status"
    )
    smooth_series(
        brdf_grid_path=args.project_path,
        filename=args.out_path,
        vi=args.vi,
        bands=args.bands,
        read_chunks=args.read_chunks,
        proc_chunks=args.proc_chunks,
        n_workers=args.workers,
        memory_limit=args.memory_limit,
        periods=[(args.start_date, args.end_date)],
        freq=args.freq,
        target_day=args.target_day
    )
