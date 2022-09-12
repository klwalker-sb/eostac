import itertools
import enum
from pathlib import Path
import typing as T
import fnmatch
from datetime import datetime
import concurrent.futures

from .errors import CorruptedError
from geowombat.core import ndarray_to_xarray

import numpy as np
from scipy.interpolate import BSpline, splrep
import pandas as pd
from dask.delayed import Delayed
from dask.distributed import LocalCluster, Client
import xarray as xr
from sklearn.linear_model import TheilSenRegressor
import attr
import matplotlib.pyplot as plt
import seaborn as sns
import json
from affine import Affine
from tqdm.auto import tqdm
import psutil


NODATA = 32768
DOWN_SCALE = 1e4
UP_SCALE = 1e-4
NODOWNLOAD = -999


class LandsatBands(enum.Enum):
    coastal = 'coastal'
    blue = 'blue'
    green = 'green'
    red = 'red'
    nir = 'nir08'
    swir1 = 'swir16'
    swir2 = 'swir22'
    qa = 'qa_pixel'


LANDSAT_LIKE_BAND_NAMES = [
    LandsatBands('blue').name,
    LandsatBands('green').name,
    LandsatBands('red').name,
    LandsatBands('nir08').name,
    LandsatBands('swir16').name,
    LandsatBands('swir22').name
]


class Sentinel2Bands(enum.Enum):
    coastal = 'B01'
    blue = 'B02'
    green = 'B03'
    red = 'B04'
    nir1 = 'B05'
    nir2 = 'B06'
    nir3 = 'B07'
    nir = 'B08'
    rededge = 'B8A'
    water = 'B09'
    cirrus = 'B10'
    swir1 = 'B11'
    swir2 = 'B12'
    scl = 'SCL'


def get_lin_coefficients(
    X: np.ndarray,
    y: np.ndarray,
    scale_factor: float,
    plot_title: str,
    show_plot: bool = False,
    out_fig: T.Union[str, Path] = None
):
    sns.set_style('whitegrid')
    fig, axes = plt.subplots(2, 3, figsize=(8, 4), dpi=100)
    axes = axes.flatten()
    sns.despine(fig=fig)

    colors = ['blue', 'green', 'red', 'darkred', 'orange', 'purple']
    lin_model = TheilSenRegressor(n_jobs=1)

    coeffs = {}
    for band_x, band_y, ax, title, color, in zip(X, y, axes, LANDSAT_LIKE_BAND_NAMES, colors):
        xsite = band_x.flatten().astype('float32')
        ysite = band_y.flatten().astype('float32')
        idx = np.where(
            (xsite != NODATA)
            & (ysite != NODATA)
            & (xsite > 0)
            & (ysite > 0)
            & (np.abs(ysite - xsite) < 3_000)
        )
        xsite = xsite[idx] * scale_factor
        ysite = ysite[idx] * scale_factor

        reg = lin_model.fit(
            xsite[:, None],
            ysite
        )

        # yhat = X*beta + alpha
        yhat = xsite*float(reg.coef_) + float(reg.intercept_)

        ax.scatter(
            xsite,
            ysite,
            s=10,
            c='k',
            ec='none',
            alpha=0.5
        )
        ax.scatter(
            yhat,
            ysite,
            s=10,
            c=color,
            ec='white',
            lw=0.1,
            alpha=0.5
        )
        # ax.hexbin(
        #     yhat,
        #     (10 ** ysite) * UP_SCALE,
        #     gridsize=10,
        #     cmap='plasma'
        # )
        ax.plot(
            [0, 1],
            [0, 1],
            color='grey',
            lw=1,
            ls='-',
            solid_capstyle='round'
        )
        ax.set_title(title)
        plt.suptitle(plot_title)

        data_min = np.clip(np.concatenate((yhat, ysite)).min() - 0.05, 0, 1)
        data_max = np.clip(np.concatenate((yhat, ysite)).max() + 0.05, 0, 1)

        ax.set_xlim(data_min, data_max)
        ax.set_ylim(data_min, data_max)

        coeffs[title] = {
            'beta': float(reg.coef_),
            'alpha': float(reg.intercept_)
        }

    if out_fig is not None:
        plt.tight_layout()
        plt.savefig(out_fig, dpi=300)

    if show_plot:
        plt.show()

    return coeffs


class SensorNames(enum.Enum):
    landsat5 = 'lt05'
    landsat7 = 'le07'
    landsat8 = 'lc08'
    landsat9 = 'lc09'
    sentinel2 = 's2a'
    sentinel2a = 's2a'
    sentinel2b = 's2b'
    sentinel2c = 's2c'


class GWSensorNameLookup(enum.Enum):
    l5 = SensorNames.landsat5
    l7 = SensorNames.landsat7
    l8 = SensorNames.landsat8
    l9 = SensorNames.landsat9
    s2 = SensorNames.sentinel2a
    s2a = SensorNames.sentinel2a
    s2b = SensorNames.sentinel2b
    s2c = SensorNames.sentinel2c


def get_landsat_sensor_name(sensor_id: str):
    for sensor_name in SensorNames:
        if fnmatch.fnmatch(sensor_id.lower(), f'{sensor_name.value}*'):
            return sensor_name.value

    return None


def save_dict(data_file: T.Union[str, Path], data: dict):
    with open(str(data_file).replace('.npz', '.json'), mode='w') as f:
        json.dump(data, f, indent=4)


def load_dict(data_file: T.Union[str, Path]) -> dict:
    with open(str(data_file), mode='r') as f:
        data = json.load(f)

    return data


def save_array(array_file: T.Union[str, Path], **kwargs):
    with open(str(array_file), mode='wb') as f:
        np.savez_compressed(f, **kwargs)


def load_array(array_file: T.Union[str, Path]) -> np.ndarray:
    with open(array_file, mode='rb') as f:
        data = np.load(str(array_file))

    return data


class SensorPathsEnum(enum.Enum):
    l5_to_l8_coefficients = 'coefficients_l5_to_l8.npz'
    l7_to_l8_coefficients = 'coefficients_l7_to_l8.npz'
    l9_to_l8_coefficients = 'coefficients_l9_to_l8.npz'
    s2_to_l8_coefficients = 'coefficients_s2_to_l8.npz'
    landsat = 'landsat'
    sentinel2 = 'sentinel2'


@attr.s
class SensorPaths(object):
    out_path: T.Union[str, Path] = attr.ib(default=None)

    @staticmethod
    def mkdir(p: Path):
        p.mkdir(parents=True, exist_ok=True)

    def get_path(self, coeffs: str) -> Path:
        return Path(self.out_path) / getattr(SensorPathsEnum, coeffs).value

    @property
    def l5_to_l8_coefficients(self) -> Path:
        return self.get_path('l5_to_l8_coefficients')

    @property
    def l7_to_l8_coefficients(self) -> Path:
        return self.get_path('l7_to_l8_coefficients')

    @property
    def l9_to_l8_coefficients(self) -> Path:
        return self.get_path('l9_to_l8_coefficients')

    @property
    def s2_to_l8_coefficients(self) -> Path:
        return self.get_path('s2_to_l8_coefficients')

    @property
    def landsat(self) -> Path:
        lpath = Path(self.out_path) / SensorPathsEnum.landsat.value
        self.mkdir(lpath)

        return lpath

    @property
    def sentinel2(self) -> Path:
        s2path = Path(self.out_path) / SensorPathsEnum.sentinel2.value
        self.mkdir(s2path)

        return s2path


def read_df(scene_info: Path) -> pd.DataFrame:
    """Reads the scene info DataFrame
    """
    df = pd.read_pickle(scene_info).set_index('date')
    df.index.rename(None, inplace=True)
    df = df.assign(date=df.index)

    return df.sort_index()


def pickle_scene_info(
    df: pd.DataFrame,
    date_pos: int,
    scene_info: Path
) -> tuple:
    """Pickles scene information to/from file
    """
    df['date'] = pd.DatetimeIndex(df.id.str.split('_').str[date_pos])
    df_download = df.copy()
    if scene_info.is_file():
        df_info = pd.read_pickle(scene_info)
        # Set the index as unique ids
        df = df.set_index('id', drop=False)
        df.index.rename(None, inplace=True)
        df_info = df_info.set_index('id', drop=False)
        df_info.index.rename(None, inplace=True)
        if (len(df_info.index) == 1) or (len(df_info.index.difference(df.index)) > 0):
            df = pd.concat((df_info, df), axis=0).reset_index(drop=True)
        df = df.drop_duplicates(subset=['id', 'date'])
    df.to_pickle(scene_info)

    return df, df_download


def assign_crs(data: xr.DataArray) -> xr.DataArray:
    return data.assign_attrs(crs=data.gw.crs_to_pyproj.to_proj4())


def rechunk(data: xr.DataArray, chunks: int):
    """Re-chunks a ``DataArray`` if the chunk sizes are not in intervals of 16
    """
    if not (data.gw.row_chunks % 16 == 0) or not (data.gw.col_chunks % 16 == 0):
        if data.gw.nrows >= chunks:
            data = data.chunk(chunks={'band': 1, 'y': chunks, 'x': chunks})
        else:
            chunk_size = 16.0
            while True:
                if (chunk_size < data.gw.nrows) and (chunk_size < data.gw.ncols):
                    break
                chunk_size *= 0.5
                if chunk_size <= 2:
                    break
            data = data.chunk(chunks={'band': 1, 'y': int(chunk_size), 'x': int(chunk_size)})

    return data


def _scale(data: xr.DataArray) -> xr.DataArray:
    return data * UP_SCALE


def expand_time(dataset):
    """`open_mfdataset` preprocess function
    """
    attrs = dataset.attrs.copy()
    attrs['transform'] = Affine(*attrs['transform'])
    attrs['res'] = tuple(attrs['res'])
    # Get the date
    file_date = datetime.strptime(Path(dataset.encoding['source']).stem.split('_')[3], '%Y%m%d')
    darray = (
        dataset
        .to_array()
        .rename({'variable': 'band'})
        .sel(band=LANDSAT_LIKE_BAND_NAMES)
        .assign_coords(time=file_date, y=dataset.y, x=dataset.x)
        .expand_dims('time')
        .transpose('time', 'band', 'y', 'x')
        .where(lambda x: x != x.nodatavals[0])
    )

    return darray.map_blocks(_scale, template=darray).assign_attrs(**attrs)


def get_netcdf_encoding(
    chunksize: tuple,
    bands: T.Sequence[int],
    dtype: str,
    encoding: dict
) -> dict:
    """Gets encodings for NetCDF files
    """
    all_encodings = {}
    for band_name in bands:
        encode_dict = {
            'chunksizes': chunksize,
            'dtype': dtype
        }
        encode_dict.update(encoding)
        all_encodings[band_name] = encode_dict

    return all_encodings


def save_netcdf(
    datasets: T.Sequence[xr.Dataset],
    filenames: T.Sequence[T.Union[str, Path]],
    **kwargs
) -> Delayed:
    """Saves a collection of xarray Datasets as NetCDF files
    """
    xr.save_mfdataset(
        datasets,
        filenames,
        mode='w',
        **kwargs
    )
    # Check output files
    for fn in filenames:
        error_file = fn.parent.parent / f'{fn.stem}_ERROR.txt'
        try:
            with xr.open_dataset(fn, chunks={'band': -1, 'y': 1024, 'x': 1024}) as ds:
                res = ds.to_array(dim='band')
        except Exception as e:
            if fn.is_file():
                fn.unlink()
            with open(error_file, mode='w') as f:
                f.write(str(e))


class open_netcdf(object):
    """Opens a collection of files as a xarray DataArray

    Dask dashboard:
        http://localhost:8787/status
    """
    def __init__(
        self,
        file_list: T.Sequence[T.Union[str, Path]],
        chunks: int = 512,
        n_workers: int = psutil.cpu_count(),
        threads_per_worker: int = 1,
        processes: bool = True,
        memory_limit: str = 'auto'
    ):
        self.file_list = file_list
        self.chunks = chunks
        self.n_workers = n_workers
        self.threads_per_worker = threads_per_worker
        self.processes = processes
        self.memory_limit = memory_limit

    def __repr__(self) -> str:
        return str(self.darray)

    def __enter__(self):
        self.cluster = LocalCluster(
            n_workers=self.n_workers,
            threads_per_worker=self.threads_per_worker,
            processes=self.processes,
            memory_limit=self.memory_limit
        )
        self.client = Client(self.cluster)
        darray = xr.open_mfdataset(
            self.file_list,
            concat_dim='time',
            chunks={
                'time': len(self.file_list),
                'band': len(LANDSAT_LIKE_BAND_NAMES),
                'y': self.chunks,
                'x': self.chunks
            },
            combine='nested',
            engine='h5netcdf',
            preprocess=expand_time,
            parallel=True
        )
        self.attrs = darray.attrs.copy()
        self.darray = (
            (
                darray
                .groupby('time')
                .mean(skipna=True)
                .chunk({'time': -1, 'band': 1})
            )
            .assign_attrs(**darray.attrs)
        )

        return self

    def close(self):
        self.darray.close()
        self.client.close()
        self.cluster.close()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def load(
        self,
        date_range: T.Optional[pd.DatetimeIndex] = None,
        bands: T.Optional[T.Sequence[str]] = None,
        coord_slice: T.Optional[tuple] = None,
        compute: T.Optional[bool] = False
    ):
        darray = None
        if date_range is not None:
            darray = (
                self.darray
                .interpolate_na(dim='time')
                .bfill(dim='time')
                .ffill(dim='time')
                .interp(time=date_range, method='cubic')
            )
        if bands is not None:
            if darray is None:
                darray = self.darray.sel(band=bands)
            else:
                darray = darray.sel(band=bands)
        if coord_slice is not None:
            # Add time and bands so slice = (time, bands, rows, columns)
            coord_slice = (
                slice(0, None),
                slice(0, None)
            ) + coord_slice
            if darray is None:
                darray = self.darray[coord_slice]
            else:
                darray = darray[coord_slice]

        if darray is None:
            darray = self.darray.assign_attrs(**self.attrs)
        else:
            darray = darray.assign_attrs(**self.attrs)
        if compute:
            darray = darray.load()

        return darray


def get_interp_grids(darray: xr.DataArray, date_range: pd.DatetimeIndex):
    t = pd.DatetimeIndex(darray.time.values)
    days = [1]
    for tidx in range(1, len(t)):
        days.append(days[-1] + (t[tidx] - t[tidx-1]).days)
    df = pd.DataFrame(data=days, columns=['day'], index=t)
    df = df.reindex(date_range).reset_index().rename(columns={'index': 'date'})
    xsparse = df[~df.day.isna()].index.values
    xnew = df.index.values

    return df, xsparse, xnew


def detect_outliers(y: np.ndarray, nstd: float = 1.0):
    """Detects outliers
    """
    d = np.diff(y)
    d = d / y[1:] * 100.0
    mu = d.mean()
    std = d.std()
    outliers = np.where(
        (d < mu-std*nstd) | (d > mu+std*nstd)
    )
    if outliers[0].shape[0] > 0:
        outliers = (outliers[0] + 1).tolist()
    else:
        outliers = []

    return outliers


def _fit_bspline(
    xsparse: np.ndarray,
    xnew: np.ndarray,
    ydata: np.ndarray,
    s: float,
    k: int,
    weights: np.ndarray
):
    tck = splrep(xsparse, ydata, s=s, k=k, w=weights)

    return BSpline(*tck)(xnew)


def set_outliers_weights(ysparse: np.ndarray) -> T.Tuple[np.ndarray, np.ndarray]:
    weights = np.ones_like(ysparse) + 1
    weights[ysparse < np.percentile(ysparse, 10)] = 5

    outliers_a = detect_outliers(ysparse, nstd=1.0)
    outliers_b = detect_outliers(1-ysparse, nstd=1.0)
    outliers_lower = np.array(sorted(list(set(outliers_a + outliers_b))))
    if outliers_lower.shape[0] > 0:
        weights[outliers_lower] = 0.5
    outliers_c = detect_outliers(ysparse, nstd=2.0)
    outliers_d = detect_outliers(1-ysparse, nstd=2.0)
    outliers_upper = np.array(sorted(list(set(outliers_c + outliers_d))))
    if outliers_upper.shape[0] > 0:
        weights[outliers_upper] = 0.25

    outliers = np.array(
        sorted(list(set(
            outliers_lower.tolist() + outliers_upper.tolist()
        )))
    )

    return outliers, weights


def rmse_score(ytrue: np.ndarray, ypred: np.ndarray) -> np.ndarray:
    return ((ypred - ytrue)**2).mean()**0.5


def fit_bspline(
    *args,
    srange: tuple = None,
    step: float = 0.05,
    tol: float = 1e-3
):
    y, xsparse, xnew, s, k = list(itertools.chain(*args))

    if srange is None:
        srange = (0.01, 0.2)
    ysparse = y[xsparse]
    outliers, weights = set_outliers_weights(ysparse)

    s = srange[0]
    stop = srange[1]
    min_score = 1e9
    last_score = None
    best_s = s
    while True:
        yhat = _fit_bspline(xsparse, xnew, ysparse, s, k, weights)
        if outliers.shape[0] > 0:
            rmse = rmse_score(ysparse[~outliers], yhat[~outliers])
        else:
            rmse = rmse_score(ysparse, yhat)
        if last_score is not None:
            if abs(rmse - last_score) < tol:
                break
            if rmse > last_score:
                break
        if rmse < min_score:
            min_score = rmse
            best_s = s
            last_score = rmse

        if s == srange[0]:
            s += (step - s)
        else:
            s += step
        if s > stop:
            break

    return np.uint16(_fit_bspline(xsparse, xnew, ysparse, best_s, k, weights)*DOWN_SCALE)


def smooth(
    darray: xr.DataArray,
    vi: str,
    date_range: pd.DatetimeIndex,
    s: float = 0.1,
    k: int = 3,
    return_raw: T.Optional[bool] = False,
    n_workers: int = psutil.cpu_count()
):
    """Smooths an image along the 'time' dimension
    """
    attrs = darray.attrs.copy()
    # Calculate an spectral index, reshape, and fill nans
    yraw = (
        getattr(darray.gw, vi)(sensor='l7')
        .squeeze()
        .stack(z=('y', 'x'))
        .transpose()
    )
    if return_raw:
        yraw = yraw.load()

    df, xsparse, xnew = get_interp_grids(yraw, date_range)
    # Interpolate to the target range
    yinterp = (
        yraw
        .chunk({'z': 256, 'time': -1})
        .interpolate_na(dim='time')
        .bfill(dim='time')
        .ffill(dim='time')
        .interp(time=date_range, method='cubic')
    )
    if not return_raw:
        yinterp = yinterp.load()
    # Smooth the data
    iters = (
        (y, xsparse, xnew, s, k) for y in yinterp.values
    )
    ysmooth = []
    pbar_chunk = 500
    total_iters = yinterp.shape[0]
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        with tqdm(total=total_iters, position=1, desc='Smooth', leave=False) as pbar:
            for i, result in enumerate(executor.map(fit_bspline, iters, chunksize=1_000)):
                ysmooth.append(result)
                if i % pbar_chunk == 0:
                    if total_iters - i < pbar_chunk:
                        pbar.update(total_iters-i)
                    else:
                        pbar.update(pbar_chunk)
    ysmooth = np.array(ysmooth)
    ysmooth = np.ascontiguousarray(
        ysmooth.T.reshape(
            ysmooth.shape[1], darray.gw.nrows, darray.gw.ncols
        )
    )
    ysmooth = (
        ndarray_to_xarray(
            darray,
            ysmooth,
            band_names=df.date.values
        )
        .rename({'band': 'time'})
    )

    # ysmooth = (
    #     xr.apply_ufunc(
    #         fit_bspline,
    #         yinterp,
    #         input_core_dims=[['time']],
    #         output_core_dims=[['time']],
    #         exclude_dims=set(('time',)),
    #         kwargs={
    #             'xsparse': xsparse,
    #             'xnew': xnew,
    #             's': s,
    #             'k': k
    #         },
    #         dask='forbidden',
    #         vectorize=True,
    #         dask_gufunc_kwargs={
    #             'time': xnew.shape[0]
    #         },
    #         output_dtypes=[np.uint16]
    #     )
    #     .assign_coords(time=('time', df.date))
    #     .assign_attrs(**attrs)
    # )

    if return_raw:
        return yraw.unstack(), ysmooth.unstack()
    else:
        return ysmooth
