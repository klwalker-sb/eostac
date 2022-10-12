from pathlib import Path
import enum
import typing as T
from abc import ABC
import concurrent.futures

from .errors import CorruptedError, DownloadError
from .stac_utils import (
    LandsatBands,
    Sentinel2Bands,
    SensorPaths,
    pickle_scene_info,
    rechunk,
    NODATA,
    NODOWNLOAD
)

import geowombat as gw
from geowombat.core.stac import open_stac

import pandas as pd
import geopandas as gpd
from rasterio.enums import Resampling
import xarray as xr
import pyproj
from shapely.geometry import CAP_STYLE, JOIN_STYLE
# from s2cloudless import S2PixelCloudDetector
import attr
from tqdm.auto import tqdm


@attr.s
class Parameters(object):
    """General parameters
    """
    bounds: T.Union[str, Path, T.Sequence[float]] = attr.ib(default=None)
    bounds_query: str = attr.ib(default=None)
    bounds_buffer: int = attr.ib(default=0, converter=int)
    out_path: Path = attr.ib(default=None, converter=Path)
    start_date: str = attr.ib(default=None)
    end_date: str = attr.ib(default=None)
    epsg: T.Union[int, None] = attr.ib(default=None)
    resolution: float = attr.ib(default=10, converter=float)
    chunksize: int = attr.ib(default=512, converter=int)
    cloud_cover_perc: int = attr.ib(default=90)
    n_workers: int = attr.ib(default=1, converter=int)
    threads_per_worker: int = attr.ib(default=1, converter=int)
    view_asset_keys: bool = attr.ib(default=False)
    max_items: T.Union[int, None] = attr.ib(default=100)
    tqdm_item_position: int = attr.ib(default=0)
    tqdm_extra_position: int = attr.ib(default=1)
    errors_only: bool = attr.ib(default=False)

    def __attrs_post_init__(self):
        # Open a file if passed as a path
        if not isinstance(self.bounds, gpd.GeoDataFrame):
            self.bounds = gpd.read_file(self.bounds)
        # Query a geometry feature subset
        if self.bounds_query is not None:
            self.bounds = self.bounds.query(self.bounds_query)
        # Buffer the bounds
        if self.bounds_buffer != 0:
            buffer_geom = self.bounds.buffer(
                self.bounds_buffer, cap_style=CAP_STYLE.square, join_style=JOIN_STYLE.mitre
            )
            self.bounds = self.bounds.assign(geometry=buffer_geom)
        self.original_crs = self.bounds.crs
        self.proj_bounds = self.bounds.total_bounds.flatten().tolist()

        # Translate to WGS 84 lat/lon
        if self.bounds.crs != pyproj.CRS.from_epsg(4326):
            self.bounds = self.bounds.to_crs('epsg:4326')


class SensorParams(ABC):
    """Sensor parameters meta class
    """
    bands: T.List[str]
    collection: str
    stac_catalog: str
    mask_data: bool
    epsg: T.Union[int, None]
    resolution: T.Union[float, int]
    resampling: Resampling
    extra_assets: T.List[str]
    out_path: Path


@attr.s
class LandsatParams(SensorParams):
    """Landsat-specific parameters
    """
    bands: T.List[str] = attr.ib(default=[
        LandsatBands.blue.value,
        LandsatBands.green.value,
        LandsatBands.red.value,
        LandsatBands.nir.value,
        LandsatBands.swir1.value,
        LandsatBands.swir2.value,
        LandsatBands.qa.value
    ])
    collection: str = attr.ib(default='landsat_c2_l2')
    stac_catalog: str = attr.ib(default='microsoft')
    mask_data: bool = attr.ib(default=True)
    epsg: T.Union[int, None] = attr.ib(default=None)
    resolution: T.Union[float, int] = attr.ib(default=10.0)
    resampling: Resampling = attr.ib(default=Resampling.cubic)
    extra_assets: T.List[str] = attr.ib(default=['ang', 'mtl.txt', 'mtl.xml'])
    out_path: Path = attr.ib(default='.')

    def __attrs_post_init__(self):
        self.out_path.mkdir(parents=True, exist_ok=True)


@attr.s
class Sentinel2Params(SensorParams):
    """Sentinel-2-specific parameters
    """
    bands: T.List[str] = attr.ib(default=[
        Sentinel2Bands.blue.value,
        Sentinel2Bands.green.value,
        Sentinel2Bands.red.value,
        Sentinel2Bands.nir.value,
        Sentinel2Bands.swir1.value,
        Sentinel2Bands.swir2.value,
        Sentinel2Bands.scl.value
    ])
    collection: str = attr.ib(default='sentinel_s2_l2a')
    stac_catalog: str = attr.ib(default='element84')
    mask_data: bool = attr.ib(default=False)
    epsg: T.Union[int, None] = attr.ib(default=None)
    resolution: T.Union[float, int] = attr.ib(default=10.0)
    resampling: Resampling = attr.ib(default=Resampling.cubic)
    extra_assets: T.List[str] = attr.ib(default=['metadata'])
    out_path: Path = attr.ib(default='.')

    def __attrs_post_init__(self):
        self.out_path.mkdir(parents=True, exist_ok=True)


class S2SceneClassificationLegend(enum.Enum):
    """Sentinel-2 Scene Classification Layer legend
    """
    saturated_or_defective = 1
    dark_area_pixels = 2
    cloud_shadows = 3
    vegetation = 4
    bare_soils = 5
    water = 6
    unclassified = 7
    cloud_medium_probability = 8
    cloud_high_probability = 9
    thin_cirrus = 10
    snow_ice = 11


S2CLOUDLESS_BAND_NAMES = ['coastal', 'blue', 'red', 'nir1', 'nir', 'rededge', 'water', 'cirrus', 'swir1', 'swir2']
S2CLOUDLESS_BANDS = [getattr(Sentinel2Bands, band).value for band in S2CLOUDLESS_BAND_NAMES]
S2_VIS_BAND_NAMES = ['blue', 'green', 'red']
S2_VIS_BANDS = [getattr(Sentinel2Bands, band).value for band in S2_VIS_BAND_NAMES]


def s2_l2a_clouds(data: xr.DataArray) -> xr.DataArray:
    """Data masker for Sentinel-2 Level 2A
    """
    replace = {
        old_value: NODATA for old_value in [
            S2SceneClassificationLegend.saturated_or_defective.value,
            S2SceneClassificationLegend.dark_area_pixels.value,
            S2SceneClassificationLegend.cloud_shadows.value,
            S2SceneClassificationLegend.cloud_medium_probability.value,
            S2SceneClassificationLegend.cloud_high_probability.value,
            S2SceneClassificationLegend.thin_cirrus.value,
            S2SceneClassificationLegend.snow_ice.value
        ]
    }

    return data.sel(band=Sentinel2Bands('SCL').name).gw.replace(replace)


# def s2_l1c_clouds(data: xr.DataArray):
#     """Data masker for Sentinel-2 Level 1C
#     """
#     data = data.assign_coords(
#         band=S2CLOUDLESS_BAND_NAMES
#     )
#
#     kwargs = dict(
#         threshold=0.4,
#         average_over=4,
#         dilation_size=5,
#         all_bands=False
#     )
#     cloud_detector = S2PixelCloudDetector(**kwargs)
#     # Get the S2Cloudless bands'
#     # B01, B02, B04, B05, B08, B8A, B09, B10, B11, B12
#     data_cloudless = data.sel(band=S2CLOUDLESS_BAND_NAMES)
#
#     X = Stackerizer(
#         stack_dims=('y', 'x', 'time'), direction='stack'
#     ).fit_transform(data_cloudless)


def write_stack(
    row_id: str,
    data: xr.DataArray,
    out_path: Path,
    threads_per_worker: int,
    tqdm_extra_position: int
):
    """Writes a single time band stack to file
    """
    out_name = out_path / f"{row_id}.tif"
    if not out_name.is_file():
        # Index by scene id because it is always unique
        error_file = out_path.parent.parent / f'{Path(out_name).parent.parent.name}_{Path(out_name).stem}_ERROR.txt'
        scene_idx = data.id.values.tolist().index(row_id)
        try:
            (
                data
                .isel(time=scene_idx)
                .squeeze()
                .transpose('band', 'y', 'x')
                .gw.save(
                    out_name,
                    overwrite=True,
                    compression='lzw',
                    num_workers=threads_per_worker,
                    log_progress=False,
                    tqdm_kwargs={
                        'position': tqdm_extra_position+2,
                        'desc': 'Chunks'
                    }
                )
            )
            # Ensure the file is not corrupted
            if out_name.is_file():
                try:
                    with gw.open(out_name) as src:
                        if (src.min().data.compute() == NODATA) & (src.max().data.compute() == NODATA):
                            raise ValueError('The image is empty.')
                    # Successful open at this point
                    # If there was previously an error file, remove it
                    if error_file.is_file():
                        error_file.unlink()
                except Exception as e:
                    raise CorruptedError(f'The downloaded file is corrupted because of {e}.')
            else:
                raise DownloadError('No file was created')

        except (CorruptedError, DownloadError) as e:
            if not e.endswith('The image is empty.'):
                # Remove the possibly incomplete or corrupted file
                if out_name.is_file():
                    out_name.unlink()
                with open(error_file, mode='w') as f:
                    f.write(str(e))


def write_time_stack(
    data: xr.DataArray,
    out_path: Path,
    n_workers: int,
    threads_per_worker: int,
    df: pd.DataFrame,
    tqdm_extra_position: int,
    desc: str,
    chunks: int,
    errors_only: bool
) -> None:
    """Writes each DataArray time step to file
    """
    # Check for chunk sizes in intervals of 16
    data = rechunk(data, chunks)

    if errors_only:
        df['error_file'] = out_path.parent / (f'{out_path.parent.name}_' + df.id.map(str) + '_ERROR.txt')
        df['error_file_exists'] = df.error_file.apply(lambda x: x.is_file())
        df = df.loc[df.error_file_exists]
        if df.empty:
            return

    data = (
        data
        .gw.assign_nodata_attrs(NODATA)
        .assign_attrs(**{
            'time': pd.DatetimeIndex(data.time.values).round('D'),
            'crs': data.gw.crs_to_pyproj.to_wkt()
        })
    )

    # from tqdm.dask import TqdmCallback
    # with TqdmCallback(desc=desc):
    #     # Download the data into memory
    #     data.load(num_workers=threads_per_worker)

    if n_workers == 1:
        for row in tqdm(
            df.sort_values('date').itertuples(),
            position=tqdm_extra_position,
            desc=desc
        ):
            write_stack(
                row.id,
                data,
                out_path,
                threads_per_worker,
                tqdm_extra_position
            )

    else:
        data_gen = (
            (
                row.id,
                data,
                out_path,
                threads_per_worker,
                tqdm_extra_position
            ) for row in df.sort_values('date').itertuples()
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(write_stack, *args) for args in data_gen]
            with tqdm(
                total=len(df.index),
                position=tqdm_extra_position,
                desc=desc
            ) as pbar:
                for future in concurrent.futures.as_completed(futures):
                    future.result()
                    pbar.update(1)


def download_landsat(params: Parameters, landsat_params: LandsatParams) -> int:
    """Downloads Landsat data
    """
    data, df = open_stac(
        stac_catalog=landsat_params.stac_catalog,
        bounds=params.bounds,
        proj_bounds=None if params.original_crs == pyproj.CRS.from_epsg(4326) else params.proj_bounds,
        collection=landsat_params.collection,
        bands=landsat_params.bands,
        cloud_cover_perc=params.cloud_cover_perc,
        chunksize=params.chunksize,
        start_date=params.start_date,
        end_date=params.end_date,
        mask_data=landsat_params.mask_data,
        resolution=landsat_params.resolution,
        epsg=landsat_params.epsg,
        view_asset_keys=params.view_asset_keys,
        resampling=landsat_params.resampling,
        extra_assets=landsat_params.extra_assets,
        out_path=landsat_params.out_path,
        max_items=params.max_items,
        tqdm_item_position=params.tqdm_item_position,
        tqdm_extra_position=params.tqdm_extra_position
    )
    if data is None:
        return NODOWNLOAD
    else:
        data = data.assign_coords(band=[LandsatBands(band).name for band in data.band.values])
        data = data.assign_attrs(scene_ids=df.id.values.tolist())

        # Save the scene info
        scene_info = landsat_params.out_path / 'scene.info'
        df, df_download = pickle_scene_info(df, date_pos=3, scene_info=scene_info)

        # Write each date to file
        write_time_stack(
            data,
            landsat_params.out_path,
            n_workers=params.n_workers,
            threads_per_worker=params.threads_per_worker,
            df=df_download,
            tqdm_extra_position=params.tqdm_extra_position,
            desc='Landsat scenes',
            chunks=params.chunksize,
            errors_only=params.errors_only
        )

        return int(data.epsg.values)


def download_sentinel2(params: Parameters, sentinel2_params: Sentinel2Params):
    """Downloads Sentinel-2 data
    """
    data, df = open_stac(
        stac_catalog=sentinel2_params.stac_catalog,
        bounds=params.bounds,
        proj_bounds=None if params.original_crs == pyproj.CRS.from_epsg(4326) else params.proj_bounds,
        collection=sentinel2_params.collection,
        bands=sentinel2_params.bands,
        cloud_cover_perc=params.cloud_cover_perc,
        chunksize=params.chunksize,
        start_date=params.start_date,
        end_date=params.end_date,
        bounds_query=params.bounds_query,
        resolution=sentinel2_params.resolution,
        epsg=sentinel2_params.epsg,
        nodata_fill=NODATA,
        resampling=sentinel2_params.resampling,
        extra_assets=sentinel2_params.extra_assets,
        out_path=sentinel2_params.out_path,
        max_items=params.max_items,
        tqdm_item_position=params.tqdm_item_position,
        tqdm_extra_position=params.tqdm_extra_position
    )
    if data is not None:
        s2_bands = [Sentinel2Bands(band).name for band in data.band.values]
        data = data.assign_coords(band=s2_bands)
        data = data.assign_attrs(scene_ids=df.id.values.tolist())

        # Mask clouds using the ESA sen2cor scene classification layer
        data_s2_cloud_mask = s2_l2a_clouds(data)
        data = (
            (
                xr.where(data_s2_cloud_mask == NODATA, NODATA, data)
            )
            .sel(band=[band for band in s2_bands if band != Sentinel2Bands('SCL').name])
            .assign_attrs(**data.attrs)
        )

        # Save the scene info
        scene_info = sentinel2_params.out_path / 'scene.info'
        df, df_download = pickle_scene_info(df, date_pos=2, scene_info=scene_info)

        # Write each date to file
        write_time_stack(
            data,
            sentinel2_params.out_path,
            n_workers=params.n_workers,
            threads_per_worker=params.threads_per_worker,
            df=df_download,
            tqdm_extra_position=params.tqdm_extra_position,
            desc='Sentinel-2 scenes',
            chunks=params.chunksize,
            errors_only=params.errors_only
        )


def download_data(args):
    params = Parameters(
        bounds=args.bounds,
        bounds_query=args.bounds_query,
        bounds_buffer=args.bounds_buffer,
        out_path=args.out_path,
        start_date=args.start_date,
        end_date=args.end_date,
        epsg=args.epsg,
        resolution=args.resolution,
        cloud_cover_perc=args.cloud_cover_perc,
        chunksize=args.chunksize,
        n_workers=args.workers,
        threads_per_worker=args.threads,
        view_asset_keys=args.view_asset_keys,
        max_items=None if args.max_items == -1 else args.max_items,
        tqdm_item_position=args.tqdm_item_position,
        tqdm_extra_position=args.tqdm_extra_position,
        errors_only=args.errors_only
    )
    # Download Landsat
    landsat_params = LandsatParams(
        out_path=SensorPaths(out_path=params.out_path).landsat,
        epsg=params.epsg,
        resolution=params.resolution
    )
    epsg = download_landsat(params, landsat_params)

    if epsg != NODOWNLOAD:
        # Download Sentinel-2
        sentinel2_params = Sentinel2Params(
            out_path=SensorPaths(out_path=params.out_path).sentinel2,
            epsg=epsg,
            resolution=params.resolution
        )
        download_sentinel2(params, sentinel2_params)
