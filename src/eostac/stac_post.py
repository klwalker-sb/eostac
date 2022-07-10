from pathlib import Path
import tempfile
from collections import namedtuple
import enum
from dataclasses import dataclass
import typing as T
import warnings

from .stac_downloader import NODATA
from .stac_downloader import LandsatParams, Sentinel2Params
from .stac_utils import (
    SensorPaths,
    SensorNames,
    get_landsat_sensor_name,
    save_array,
    load_array,
    save_dict,
    load_dict,
    get_lin_coefficients,
    DOWN_SCALE,
    UP_SCALE,
    LandsatBands,
    Sentinel2Bands
)

import geowombat as gw
from geowombat.radiometry import landsat_pixel_angles, sentinel_pixel_angles
from geowombat.radiometry import BRDF, LinearAdjustments, RadTransforms
# from geowombat.ml import predict
import numpy as np
import pandas as pd
import xarray as xr
import rasterio as rio
from affine import Affine
import attr
# import matplotlib.pyplot as plt


BR = BRDF()
LA = LinearAdjustments()
RT = RadTransforms()


# if data_s2 is not None:
#     data_s2 = data_s2.assign_coords(band=S2_VIS_BAND_NAMES + ['SCL'])
#     data_s2_cloud_mask = s2_l2a_clouds(data_s2)
#     data_s2 = (
#         xr.where(data_s2_cloud_mask == NODATA, NODATA, data_s2)
#     ).assign_attrs(**data_s2.attrs)
#     data_s2 = (
#         data_s2
#         .sel(band=S2_VIS_BAND_NAMES)
#         .assign_coords(time=np.array(data_s2.time.values, dtype='datetime64[D]'))
#     ).assign_attrs(**data_s2.attrs)
#     # Bandpass adjustments on S2
#     la = LinearAdjustments()
#     data_s2 = la.bandpass(data_s2, sensor='s2a', to='l8', src_nodata=NODATA, dst_nodata=NODATA)
#     data_l = (
#         data_l
#         .assign_coords(time=np.array(data_l.time.values, dtype='datetime64[D]'))
#     ).assign_attrs(**data_l.attrs)
#     # Combine Landsat + Sentinel
#     data = gw.merge_stac(data_l, data_s2)
#     # Scale from [0,10000] to [0,1]
#     data = scale_data(data)
#
#     for time_coord, time in zip(
#         data.time.values,
#         pd.DatetimeIndex(data.time.values).round('D').to_pydatetime()
#     ):
#         data.sel(time=time_coord).transpose('band', 'y', 'x').gw.to_raster(
#             ProjectPaths.main_path / f"{time.strftime('%Y-%m-%d')}.tif",
#             n_workers=8,
#             n_threads=1,
#             compress='lzw',
#             overwrite=True
#         )
#     # s2_l1c_clouds(data_s2)


class FileFormat(enum.Enum):
    netcdf = 'netcdf'
    geotiff = 'geotiff'


IO_PARAMETERS = {
    FileFormat.geotiff: {
        'method': 'to_raster',
        'extension': '.tif',
        'kwargs': {
            'compress': 'lzw',
            'overwrite': True,
            'tqdm_kwargs': {}
        }
    },
    FileFormat.netcdf: {
        'method': 'to_netcdf',
        'extension': '.nc',
        'kwargs': {
            'zlib': True,
            'complevel': 5,
            'overwrite': True
        }
    }
}


@attr.s
class Parameters(object):
    """General parameters
    """
    file_format: FileFormat = attr.ib(default=FileFormat.netcdf)
    out_path: T.Union[str, Path, None] = attr.ib(default=None)
    l57_angles_path: str = attr.ib(default='/mnt/c/Users/jbgra/Documents/data/nn/ESPA/landsat_angles')
    l8_angles_path: str = attr.ib(default='/mnt/c/Users/jbgra/Documents/data/nn/ESPA/l8_angles')
    subsample: int = attr.ib(default=10)
    resampling: str = attr.ib(default='bilinear')
    apply_bandpass: bool = attr.ib(default=False)
    coeffs_path: T.Union[str, Path, None] = attr.ib(default=None)
    return_data: bool = attr.ib(default=False)
    n_workers: int = attr.ib(default=1, converter=int)
    threads_per_worker: int = attr.ib(default=1, converter=int)
    show_plot: bool = attr.ib(default=False)

    def __attrs_post_init__(self):
        if self.out_path is not None:
            self.out_path = Path(self.out_path)
            self.out_path.mkdir(parents=True, exist_ok=True)

    def get_out_file(self, name: str):
        return self.out_path / f"{name}{self.io['extension']}"

    @property
    def io(self):
        return IO_PARAMETERS[self.file_format]


def create_image_reference(filename: Path, out_dir: Path, chunks: int):
    """Creates a temporary image reference for pixel angle creation
    """
    RefInfo = namedtuple('RefInfo', 'bounds crs res')

    with gw.open(filename, chunks=chunks) as src:
        ref_info = RefInfo(
            bounds=src.gw.bounds,
            crs=src.crs,
            res=src.gw.celly
        )
        transform = Affine(
            src.gw.celly, 0.0, src.gw.left, 0.0, -src.gw.celly, src.gw.top
        )
        profile = {
            'transform': transform,
            'crs': src.crs,
            'width': src.gw.ncols,
            'height': src.gw.nrows,
            'count': 1,
            'driver': 'GTiff',
            'blockxsize': 512,
            'blockysize': 512,
            'tiled': True,
            'dtype': 'uint8'
        }

    temp_file = str(out_dir / Path(filename).name)
    with rio.open(temp_file, mode='w', **profile) as dst:
        pass

    return temp_file, ref_info


@dataclass
class Data:
    sensor_lookup: dict
    l5_to_l8_coeffs_path: Path
    l7_to_l8_coeffs_path: Path
    l9_to_l8_coeffs_path: Path
    s2_to_l8_coeffs_path: Path
    landsat_path: Path
    landsat_files: T.Sequence[str]
    landsat_dates: T.Sequence[object]
    landsat_df: pd.DataFrame
    sentinel2_path: Path
    sentinel2_files: T.Sequence[str]
    sentinel2_dates: T.Sequence[object]
    sentinel2_df: pd.DataFrame


def apply_bandpass(coeffs: dict, data: xr.DataArray) -> xr.DataArray:
    """Applies a band-pass calibration adjustment
    """
    attrs = data.attrs.copy()
    alphas = np.array(
        [coeffs[band]['intercept'] for band in data.band.values], dtype='float64'
    )
    betas = np.array(
        [coeffs[band]['slope'] for band in data.band.values], dtype='float64'
    )

    alphas = xr.DataArray(
        data=alphas,
        coords={'band': data.band.values},
        dims='band'
    )

    betas = xr.DataArray(
        data=betas,
        coords={'band': data.band.values},
        dims='band'
    )

    # Apply the linear bandpass adjustment
    return (alphas + betas * data).assign_attrs(**attrs)


def landsat_brdf(
    data: xr.DataArray,
    scene_info: pd.DataFrame,
    params: Parameters,
    data_obj: T.Optional[Data] = None
):
    """Calculates BRDF for Landsat
    """
    values = []
    for row in scene_info.rename(columns={'mtl.txt': 'metadata'}).itertuples():
        if params.out_path is not None:
            if params.get_out_file(row.id).is_file():
                continue
        data_slice = scale_data(
            data.sel(time=row.date)
        )
        metadata = Path(row.metadata)
        angle_coeff = row.ang
        filename = metadata.parent / f'{row.id}.tif'
        assert filename.is_file() and metadata.is_file(), 'The file and metadata must both exist.'
        meta = RT.get_landsat_coefficients(str(metadata))

        with tempfile.TemporaryDirectory() as tmp_dir:
            ref_image, ref_info = create_image_reference(
                filename,
                Path(tmp_dir),
                data.gw.row_chunks
            )

            angle_info = landsat_pixel_angles(
                angle_coeff,
                ref_image,
                str(tmp_dir),
                meta.sensor,
                l57_angles_path=params.l57_angles_path,
                l8_angles_path=params.l8_angles_path,
                subsample=params.subsample,
                resampling=params.resampling,
                num_threads=params.threads_per_worker,
                verbose=0
            )

            if meta.sensor == 'l8':
                rad_sensor = 'l8l7'
            else:
                rad_sensor = meta.sensor

            open_kwargs = {
                'chunks': data.gw.row_chunks,
                'resampling': 'bilinear'
            }

            with gw.config.update(
                sensor=rad_sensor,
                ref_bounds=ref_info.bounds,
                ref_crs=ref_info.crs,
                ref_res=ref_info.res,
                ignore_warnings=True
            ):
                with gw.open(angle_info.sza, **open_kwargs) as sza, \
                    gw.open(angle_info.vza, **open_kwargs) as vza, \
                    gw.open(angle_info.saa, **open_kwargs) as saa, \
                    gw.open(angle_info.vaa, **open_kwargs) as vaa:
                    # BRDF normalization
                    sr_brdf = BR.norm_brdf(
                        data_slice,
                        sza, saa, vza, vaa,
                        sensor=rad_sensor,
                        wavelengths=data.band.values.tolist(),
                        out_range=DOWN_SCALE,
                        src_nodata=NODATA,
                        dst_nodata=NODATA
                    )
                    if params.apply_bandpass:
                        if data_obj is not None:
                            coeffs = None
                            if meta.sensor.lower() in ['l5', 'l7']:
                                try:
                                    coeffs = load_dict(str(data_obj.l7_to_l8_coeffs_path).replace('.npz', '.json'))
                                except FileNotFoundError:
                                    coeffs = None

                            if coeffs is not None:
                                sr_brdf = apply_bandpass(coeffs, sr_brdf)

                        # if meta.sensor.lower() in ['l5', 'l7']:
                        #     # Linearly adjust to Landsat 8
                        #     sr_brdf = LA.bandpass(
                        #         sr_brdf,
                        #         meta.sensor,
                        #         to='l8',
                        #         scale_factor=UP_SCALE,
                        #         src_nodata=NODATA,
                        #         dst_nodata=NODATA
                        #     )
                    if params.out_path is not None:
                        # Write to file
                        getattr(sr_brdf.gw, params.io['method'])(
                            params.get_out_file(row.id),
                            **params.io['kwargs']
                        )
                    else:
                        if params.return_data:
                            values.append(sr_brdf.data.compute(num_workers=params.n_workers))

    if params.return_data:
        return values


def sentinel2_brdf(
    data: xr.DataArray,
    scene_info: pd.DataFrame,
    params: Parameters,
    data_obj: T.Optional[Data] = None
):
    """Calculates BRDF for Sentinel-2
    """
    values = []
    for row in scene_info.itertuples():
        if params.out_path is not None:
            if params.get_out_file(row.id).is_file():
                continue
        data_slice = scale_data(
            data.sel(time=row.date)
        )

        metadata = Path(row.metadata)
        filename = metadata.parent / f'{row.id}.tif'
        assert filename.is_file() and metadata.is_file()

        with tempfile.TemporaryDirectory() as tmp_dir:
            ref_image, ref_info = create_image_reference(
                filename,
                Path(tmp_dir),
                data.gw.row_chunks
            )
            angle_info = sentinel_pixel_angles(
                metadata,
                ref_image,
                outdir=str(tmp_dir),
                nodata=-32768,
                overwrite=False,
                verbose=0
            )
            rad_sensor = angle_info.sensor + 'l7'
            bandpass_sensor = angle_info.sensor
            open_kwargs = {
                'chunks': data.gw.row_chunks,
                'resampling': 'bilinear'
            }
            with gw.open(angle_info.sza, **open_kwargs) as sza, \
                gw.open(angle_info.vza, **open_kwargs) as vza, \
                gw.open(angle_info.saa, **open_kwargs) as saa, \
                gw.open(angle_info.vaa, **open_kwargs) as vaa:
                # BRDF normalization
                sr_brdf = BR.norm_brdf(
                    data_slice,
                    sza, saa, vza, vaa,
                    sensor=rad_sensor,
                    wavelengths=data.band.values.tolist(),
                    out_range=DOWN_SCALE,
                    src_nodata=NODATA,
                    dst_nodata=NODATA
                )
                if params.apply_bandpass:
                    if data_obj is not None:
                        try:
                            coeffs = load_dict(str(data_obj.s2_to_l8_coeffs_path).replace('.npz', '.json'))
                        except FileNotFoundError:
                            coeffs = None

                        if coeffs is not None:
                            sr_brdf = apply_bandpass(coeffs, sr_brdf)

                # Linearly adjust to Landsat 8
                # if params.apply_bandpass:
                #     sr_brdf = LA.bandpass(
                #         sr_brdf,
                #         bandpass_sensor.lower(),
                #         to='l8',
                #         scale_factor=UP_SCALE,
                #         src_nodata=NODATA,
                #         dst_nodata=NODATA
                #     )
                if params.out_path is not None:
                    # Write to file
                    getattr(sr_brdf.gw, params.io['method'])(
                        params.get_out_file(row.id),
                        **params.io['kwargs']
                    )
                else:
                    if params.return_data:
                        values.append(sr_brdf.data.compute(num_workers=params.n_workers))

    if params.return_data:
        return values


def scale_data(data: xr.DataArray, scale_factor: float = UP_SCALE):
    """Scales data to/from [0,10_000] and [0,1]
    """
    return (
        xr.where(
            (data > 0) & (data < NODATA), data * scale_factor, NODATA
        )
        .astype('float64')
    ).assign_attrs(**data.attrs)


def read_df(scene_info: Path) -> pd.DataFrame:
    """Reads the scene info DataFrame
    """
    df = pd.read_pickle(scene_info).set_index('date')
    df.index.rename(None, inplace=True)
    df = df.assign(date=df.index)

    return df.sort_index()


def setup_paths(args) -> Data:
    sensor_lookup = {
        SensorNames.landsat5.value: 'l5_to_l8_coeffs_path',
        SensorNames.landsat7.value: 'l7_to_l8_coeffs_path',
        SensorNames.sentinel2.value: 's2_to_l8_coeffs_path'
    }

    if args.coeffs_path is None:
        project_parent = Path(args.project_path).parent
    else:
        project_parent = Path(args.coeffs_path)

    l5_to_l8_coeffs_path = SensorPaths(project_parent).l5_to_l8_coefficients
    l7_to_l8_coeffs_path = SensorPaths(project_parent).l7_to_l8_coefficients
    l9_to_l8_coeffs_path = SensorPaths(project_parent).l9_to_l8_coefficients
    s2_to_l8_coeffs_path = SensorPaths(project_parent).s2_to_l8_coefficients

    landsat_path = SensorPaths(args.project_path).landsat
    sentinel2_path = SensorPaths(args.project_path).sentinel2
    landsat_scene_info = landsat_path / 'scene.info'
    sentinel2_scene_info = sentinel2_path / 'scene.info'
    time_slice = slice(args.start_date, args.end_date)

    landsat_df = pd.DataFrame([])
    sentinel2_df = pd.DataFrame([])
    if landsat_scene_info.is_file():
        landsat_df = read_df(landsat_scene_info)
        landsat_df = landsat_df[time_slice]
    if sentinel2_scene_info.is_file():
        sentinel2_df = read_df(sentinel2_scene_info)
        sentinel2_df = sentinel2_df[time_slice]

    # Get full image paths
    if not landsat_df.empty:
        landsat_lists = [(str(landsat_path / f"{row.id}.tif"), row.date) for row in landsat_df.itertuples()]
        landsat_files, landsat_dates = zip(*landsat_lists)
    else:
        warnings.warn('No Landsat scenes matched the time query.')
        landsat_files = []
        landsat_dates = []
    if not sentinel2_df.empty:
        sentinel2_lists = [(str(sentinel2_path / f"{row.id}.tif"), row.date) for row in sentinel2_df.itertuples()]
        sentinel2_files, sentinel2_dates = zip(*sentinel2_lists)
    else:
        warnings.warn('No Sentinel-2 scenes matched the time query.')
        sentinel2_files = []
        sentinel2_dates = []

    return Data(
        sensor_lookup=sensor_lookup,
        l5_to_l8_coeffs_path=l5_to_l8_coeffs_path,
        l7_to_l8_coeffs_path=l7_to_l8_coeffs_path,
        l9_to_l8_coeffs_path=l9_to_l8_coeffs_path,
        s2_to_l8_coeffs_path=s2_to_l8_coeffs_path,
        landsat_path=landsat_path,
        landsat_files=list(landsat_files),
        landsat_dates=list(landsat_dates),
        landsat_df=landsat_df,
        sentinel2_path=sentinel2_path,
        sentinel2_files=list(sentinel2_files),
        sentinel2_dates=list(sentinel2_dates),
        sentinel2_df=sentinel2_df
    )


def load_data(args):
    data_obj = setup_paths(args)
    params = Parameters(
        file_format=FileFormat(args.file_format),
        out_path=args.out_path,
        apply_bandpass=args.apply_bandpass,
        coeffs_path=args.coeffs_path,
        return_data=True if args.get_coeffs else False,
        n_workers=args.workers,
        threads_per_worker=args.threads,
        show_plot=args.show_plot
    )
    landsat_params = LandsatParams(
        out_path=data_obj.landsat_path
    )
    sentinel2_params = Sentinel2Params(
        out_path=data_obj.sentinel2_path
    )

    if args.get_coeffs:
        if data_obj.landsat_df.empty:
            data_obj.landsat_df = pd.DataFrame([], columns=['id', 'metadata', 'ang', 'date', 'sensor'])
        else:
            data_obj.landsat_df = (
                data_obj.landsat_df
                .rename(columns={'mtl.txt': 'metadata'})
                .loc[:, ['id', 'metadata', 'ang', 'date']]
            )
            data_obj.landsat_df = data_obj.landsat_df.assign(
                sensor=data_obj.landsat_df.apply(lambda x: get_landsat_sensor_name(x.id), axis=1)
            )
        if data_obj.sentinel2_df.empty:
            data_obj.sentinel2_df = pd.DataFrame([], columns=['id', 'metadata', 'ang', 'date', 'sensor'])
        else:
            data_obj.sentinel2_df = (
                data_obj.sentinel2_df
                .assign(ang=0)
                .assign(sensor=SensorNames.sentinel2.value)
                .loc[:, ['id', 'metadata', 'ang', 'date', 'sensor']]
            )
        df = pd.concat((data_obj.landsat_df, data_obj.sentinel2_df), axis=0)

        # There must be Landsat 8/9 to calibrate on
        if (SensorNames.landsat8.value not in df.sensor.values) \
                and (SensorNames.landsat9.value not in df.sensor.values):
            warnings.warn('There are no Landsat 8/9 images to calibrate to')
            return

        if not df.empty:
            df = df.assign(date_str=df.date.dt.strftime('%Y-%m-%d'))
            df = df[df.duplicated(subset='date_str', keep=False)]

        # Check for non-Landsat 8/9 scenes within +/- N days
        #     landsat_non_l8 = df[df.sensor.str.startswith('l') & ~df.sensor.str.startswith(SensorNames.landsat8.value)]
        #     landsat_l8 = df.query(f"sensor == '{SensorNames.landsat8.value}'")
        #     if not landsat_non_l8.empty and not landsat_l8.empty:
        #         diffs = np.abs(
        #             landsat_non_l8.date.apply(lambda x: x.toordinal()).values
        #             - landsat_l8.date.apply(lambda x: x.toordinal()).values
        #         )
        #         day_diff_match = np.where((diffs > 0) & (diffs < args.max_day_diff))[0]
        #         if day_diff_match.shape[0] > 0:
        #             import ipdb
        #             ipdb.set_trace()

        # Get date pairs
        date_pairs = []
        if not df.empty:
            for date in df.date_str.unique():
                df_pair = df.query(f"date_str == '{date}'")
                if (SensorNames.landsat8.value in df_pair.sensor.values) \
                        or (SensorNames.landsat9.value in df_pair.sensor.values):
                    date_pairs.append(df_pair)

        if date_pairs:
            def apply_landsat_brdf(
                data_array: xr.DataArray,
                dataframe: pd.DataFrame,
                sensor_enum: SensorNames,
                lin_var: str = 'X'
            ) -> dict:
                xy_dict[lin_var][sensor_enum.value] = landsat_brdf(
                    data_array, dataframe.query(f"sensor == '{sensor_enum.value}'"), params
                )

                return xy_dict

            for df_pair in date_pairs:
                xy_dict = {
                    'X': {
                        SensorNames.sentinel2.value: [],
                        SensorNames.landsat5.value: [],
                        SensorNames.landsat7.value: []
                    },
                    'y': {
                        SensorNames.landsat8.value: [],
                        SensorNames.landsat9.value: []
                    },
                    'coeffs_path': {
                        SensorNames.sentinel2.value: '',
                        SensorNames.landsat5.value: '',
                        SensorNames.landsat7.value: ''
                    }
                }

                if SensorNames.sentinel2.value in df_pair.sensor.values:
                    sensor_name = SensorNames.sentinel2
                    with gw.open(
                        data_obj.sentinel2_files,
                        time_names=data_obj.sentinel2_dates,
                        band_names=[
                            Sentinel2Bands(band).name for band in sentinel2_params.bands
                            if band != Sentinel2Bands.scl.value
                        ]
                    ) as data_s2:
                        xy_dict['X'][sensor_name.value] = sentinel2_brdf(
                            data_s2, df_pair.query(f"sensor == '{sensor_name.value}'"), params
                        )
                    xy_dict['coeffs_path'][sensor_name.value] = getattr(
                        data_obj, data_obj.sensor_lookup[
                            df_pair.query(f"sensor == '{sensor_name.value}'").sensor.values[0]
                        ]
                    )
                if SensorNames.landsat5.value in df_pair.sensor.values:
                    sensor_name = SensorNames.landsat5
                    with gw.open(
                        data_obj.landsat_files,
                        time_names=data_obj.landsat_dates,
                        band_names=[
                            LandsatBands(band).name for band in landsat_params.bands if band != LandsatBands.qa.value
                        ]
                    ) as data_l:
                        xy_dict = apply_landsat_brdf(data_l, df_pair, sensor_name)
                        xy_dict['coeffs_path'][sensor_name.value] = getattr(
                            data_obj, data_obj.sensor_lookup[
                                df_pair.query(f"sensor == '{sensor_name.value}'").sensor.values[0]
                            ]
                        )
                if SensorNames.landsat7.value in df_pair.sensor.values:
                    sensor_name = SensorNames.landsat7
                    with gw.open(
                        data_obj.landsat_files,
                        time_names=data_obj.landsat_dates,
                        band_names=[
                            LandsatBands(band).name for band in landsat_params.bands if band != LandsatBands.qa.value
                        ]
                    ) as data_l:
                        xy_dict = apply_landsat_brdf(data_l, df_pair, sensor_name)
                        xy_dict['coeffs_path'][sensor_name.value] = getattr(
                            data_obj, data_obj.sensor_lookup[
                                df_pair.query(f"sensor == '{sensor_name.value}'").sensor.values[0]
                            ]
                        )
                if SensorNames.landsat8.value in df_pair.sensor.values:
                    sensor_name = SensorNames.landsat8
                    with gw.open(
                        data_obj.landsat_files,
                        time_names=data_obj.landsat_dates,
                        band_names=[
                            LandsatBands(band).name for band in landsat_params.bands if band != LandsatBands.qa.value
                        ]
                    ) as data_l:
                        xy_dict = apply_landsat_brdf(data_l, df_pair, sensor_name, lin_var='y')
                if SensorNames.landsat9.value in df_pair.sensor.values:
                    sensor_name = SensorNames.landsat9
                    with gw.open(
                        data_obj.landsat_files,
                        time_names=data_obj.landsat_dates,
                        band_names=[
                            LandsatBands(band).name for band in landsat_params.bands if band != LandsatBands.qa.value
                        ]
                    ) as data_l:
                        xy_dict = apply_landsat_brdf(data_l, df_pair, sensor_name, lin_var='y')

                if max([len(y) for y in xy_dict['y'].values()]) == 0:
                    continue

                y = None
                counter = 0
                for yl in xy_dict['y'].values():
                    if yl:
                        yl = yl[0].reshape(yl[0].shape[0], -1)
                        if y is not None:
                            y = y + yl
                        else:
                            y = yl.astype('float32')
                        counter += 1
                if y is not None:
                    y /= float(counter)

                for sensor_name in SensorNames:
                    if sensor_name.value in xy_dict['X']:
                        if len(xy_dict['X'][sensor_name.value]) > 0:
                            X = xy_dict['X'][sensor_name.value][0]
                            X = X.reshape(X.shape[0], -1)
                            coeffs_path = Path(xy_dict['coeffs_path'][sensor_name.value])
                            if coeffs_path.is_file():
                                coeffs_data = load_array(coeffs_path)
                                X = np.concatenate((coeffs_data['X'], X), axis=1)
                                y_sensor = np.concatenate((coeffs_data['y'], y), axis=1)
                            else:
                                y_sensor = y.copy()
                            save_array(coeffs_path, X=X, y=y_sensor)

            if data_obj.l5_to_l8_coeffs_path.is_file():
                coeffs_data = load_array(data_obj.l5_to_l8_coeffs_path)
                coeffs = get_lin_coefficients(
                    coeffs_data['X'],
                    coeffs_data['y'],
                    scale_factor=UP_SCALE,
                    plot_title='L5 to L8',
                    show_plot=params.show_plot,
                    out_fig=str(data_obj.l5_to_l8_coeffs_path).replace('.npz', '.png')
                )
                save_dict(str(data_obj.l5_to_l8_coeffs_path).replace('.npz', '.json'), coeffs)
            if data_obj.l7_to_l8_coeffs_path.is_file():
                coeffs_data = load_array(data_obj.l7_to_l8_coeffs_path)
                coeffs = get_lin_coefficients(
                    coeffs_data['X'],
                    coeffs_data['y'],
                    scale_factor=UP_SCALE,
                    plot_title='L7 to L8',
                    show_plot=params.show_plot,
                    out_fig=str(data_obj.l7_to_l8_coeffs_path).replace('.npz', '.png')
                )
                save_dict(str(data_obj.l7_to_l8_coeffs_path).replace('.npz', '.json'), coeffs)
            if data_obj.s2_to_l8_coeffs_path.is_file():
                coeffs_data = load_array(data_obj.s2_to_l8_coeffs_path)
                coeffs = get_lin_coefficients(
                    coeffs_data['X'],
                    coeffs_data['y'],
                    scale_factor=UP_SCALE,
                    plot_title='S2 to L8',
                    show_plot=params.show_plot,
                    out_fig=str(data_obj.s2_to_l8_coeffs_path).replace('.npz', '.png')
                )
                save_dict(str(data_obj.s2_to_l8_coeffs_path).replace('.npz', '.json'), coeffs)

    else:
        with gw.open(
            data_obj.sentinel2_files,
            time_names=data_obj.sentinel2_dates,
            band_names=[Sentinel2Bands(band).name for band in sentinel2_params.bands if band != Sentinel2Bands.scl.value]
        ) as data_s2:
            sentinel2_brdf(
                data_s2,
                data_obj.sentinel2_df,
                params,
                data_obj=data_obj
            )
        with gw.open(
            data_obj.landsat_files,
            time_names=data_obj.landsat_dates,
            band_names=[LandsatBands(band).name for band in landsat_params.bands if band != LandsatBands.qa.value]
        ) as data_l:
            landsat_brdf(
                data_l,
                data_obj.landsat_df,
                params,
                data_obj=data_obj
            )
