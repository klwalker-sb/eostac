import enum
from pathlib import Path
import typing as T
import fnmatch
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.linear_model import TheilSenRegressor
import attr
import matplotlib.pyplot as plt
import seaborn as sns
import json
from affine import Affine


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


def save_netcdf(datasets: T.Sequence[xr.Dataset], filenames: T.Sequence[T.Union[str, Path]]):
    return xr.save_mfdataset(
        datasets,
        filenames,
        mode='w',
        format='NETCDF4',
        engine='h5netcdf',
        compute=False
    )


def open_netcdf(
    file_list: T.Sequence[T.Union[str, Path]],
    chunks: int = 512
) -> xr.DataArray:
    darray = xr.open_mfdataset(
        file_list,
        concat_dim='time',
        chunks={
            'time': len(file_list),
            'band': len(LANDSAT_LIKE_BAND_NAMES),
            'y': chunks,
            'x': chunks
        },
        combine='nested',
        engine='h5netcdf',
        preprocess=expand_time,
        parallel=True
    )
    attrs = darray.attrs.copy()

    # Group dates
    return (
        darray
        .groupby('time')
        .mean(skipna=True)
        .chunk({'time': -1, 'band': -1})
        .assign_attrs(**attrs)
    )
